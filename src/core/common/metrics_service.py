"""
Centralized metrics service:
- records rolling-window samples via a bounded event deque
- reports local metrics to the leader from a dedicated thread
- optionally runs the leader aggregation / cluster-view loop on a dedicated thread
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.common.bunny_talk import BunnyTalk
from src.core.database.base import DatabaseBase
from src.core.models.cluster import Metrics, MetricsBucket, MetricsPayload, NodeMetricSeries, NodeView, WindowSample

_METRICS_LEADER_QUEUE = "metrics-leader"
_DEFAULT_REPORT_INTERVAL_S = 10.0
_DEFAULT_LEADER_LOOP_INTERVAL_S = 5.0
_DEFAULT_CLUSTER_VIEW_WINDOWS = {30, 60}
_DEFAULT_METRICS_NODE_EXPIRY_SECONDS = 30
_DEFAULT_EVENT_BUFFER_MAXLEN = 100_000
_LIVE_BUCKET_SECONDS = 5
_1M_BUCKET_SECONDS = 60
_5M_BUCKET_SECONDS = 300
_PENDING_BUCKET_MAX_AGE_S = 86_400  # drop buckets older than 24 hours from the retry queue
_1M_RETENTION_S = 86_400            # keep 1m buckets for 24 hours
_5M_RETENTION_S = 604_800           # keep 5m buckets for 7 days
_TRIM_INTERVAL_S = 3_600            # trim old metric buckets once per hour

_NormalizedKey = tuple[str, ...]
MetricDims = Union[str, tuple[str, ...], list[str]]


class _Agg(Enum):
    AVG = "avg"
    SUM = "sum"


@dataclass(frozen=True)
class _MetricEvent:
    key: _NormalizedKey
    value: float
    agg: _Agg
    n: Optional[int]
    wall_ts: float


class MetricsService:
    def __init__(
        self,
        amqp_url: str,
        logger: Logger,
        hostname: str,
        source: str,
        role: str,
        windows: list[int],
        database: DatabaseBase,
        *,
        report_interval_s: float = _DEFAULT_REPORT_INTERVAL_S,
        leader_loop_interval_s: float = _DEFAULT_LEADER_LOOP_INTERVAL_S,
        cluster_view_windows: Optional[set[int]] = None,
        metrics_node_expiry_seconds: int = _DEFAULT_METRICS_NODE_EXPIRY_SECONDS,
        last_used_ttl_seconds: Optional[int] = None,
    ) -> None:
        if not amqp_url:
            raise ValueError("amqp_url must not be empty")
        if database is None:
            raise ValueError("database must not be None")
        if not windows:
            raise ValueError("windows must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        if not role:
            raise ValueError("role must not be empty")
        if not hostname:
            raise ValueError("hostname must not be empty")
        self._amqp_url = amqp_url
        self._database = database
        self._logger = logger
        self._hostname = hostname
        self._source = source
        self._role = role
        self._windows: list[int] = sorted(windows)
        self._max_window: int = max(windows)
        self._cluster_view_windows = cluster_view_windows or set(_DEFAULT_CLUSTER_VIEW_WINDOWS)
        self._report_interval_s = report_interval_s
        self._leader_loop_interval_s = leader_loop_interval_s
        self._metrics_node_expiry_seconds = metrics_node_expiry_seconds
        self._last_used_ttl_seconds = last_used_ttl_seconds

        self._events: deque[_MetricEvent] = deque(maxlen=_DEFAULT_EVENT_BUFFER_MAXLEN)

        self._state_lock = threading.Lock()
        self._data: dict[_NormalizedKey, deque[MetricsBucket]] = {}
        self._agg: dict[_NormalizedKey, _Agg] = {}
        self._last_seen: dict[_NormalizedKey, float] = {}
        self._last_used: dict[_NormalizedKey, float] = {}
        self._cluster_payloads: Dict[str, MetricsPayload] = {}
        self._cluster_view: Dict[str, NodeView] = {}
        self._cluster_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}  # hostname -> source -> meta
        self._cluster_series: Dict[str, Dict[str, List[NodeMetricSeries]]] = {}  # hostname -> source -> series
        self._published_meta: Dict[str, Any] = {}
        self._is_leader = False

        self._last_flushed_1m_start: int = 0
        self._last_flushed_5m_start: int = 0
        # 1m buckets waiting to be rolled up into 5m buckets.
        self._pending_1m_for_5m: list[MetricsBucket] = []
        # Buckets added here are complete and must not be mutated after insertion.
        self._pending_1m: list[MetricsBucket] = []

        self._last_trimmed_at: float = 0.0

        self._main_loop: asyncio.AbstractEventLoop | None = None

        self._report_thread: threading.Thread | None = None
        self._leader_thread: threading.Thread | None = None
        self._report_stop = threading.Event()
        self._leader_stop = threading.Event()

    def record(
        self,
        key: str,
        value: float = 1.0,
        *,
        dims: Optional[MetricDims] = None,
        n: Optional[int] = None,
    ) -> None:
        nk = _normalize_metric(key, dims)
        agg = _Agg.AVG if n is not None else _Agg.SUM
        self._record_event(nk, value, agg, 1 if n is None else n)

    def publish_meta(self, meta: Dict[str, Any]) -> None:
        with self._state_lock:
            self._published_meta = dict(meta)

    def mark_last_used(self, dims: MetricDims) -> None:
        now = time.time()
        nk = _normalize_metric("last_used", dims)
        with self._state_lock:
            self._last_used[nk[1:]] = now

    def last_used_snapshot(self) -> list[tuple[tuple[str, ...], float]]:
        now = time.time()
        with self._state_lock:
            self._trim_last_used_locked(now)
            return sorted(self._last_used.items())

    def start_reporting(self) -> None:
        if self._report_thread is not None and self._report_thread.is_alive():
            return
        self._main_loop = asyncio.get_event_loop()
        self._report_stop = threading.Event()
        self._report_thread = threading.Thread(
            target=self._run_report_thread,
            name=f"metrics-report-{self._source}",
            daemon=True,
        )
        self._report_thread.start()

    def start_leader_loop(self) -> None:
        if self._leader_thread is not None and self._leader_thread.is_alive():
            return
        self._main_loop = asyncio.get_event_loop()
        self._leader_stop = threading.Event()
        self._leader_thread = threading.Thread(
            target=self._run_leader_thread,
            name=f"metrics-leader-{self._source}",
            daemon=True,
        )
        self._leader_thread.start()

    async def stop_reporting(self) -> None:
        thread = self._report_thread
        if thread is None:
            return
        self._report_stop.set()
        await asyncio.to_thread(thread.join, self._report_interval_s + 5.0)
        if thread.is_alive():
            self._logger.error("MetricsService: report thread did not stop in time; it may still be running")
        else:
            self._report_thread = None

    async def stop_leader_loop(self) -> None:
        thread = self._leader_thread
        if thread is None:
            return
        self._leader_stop.set()
        await asyncio.to_thread(thread.join, self._leader_loop_interval_s + 5.0)
        if thread.is_alive():
            self._logger.error("MetricsService: leader thread did not stop in time; it may still be running")
        else:
            self._leader_thread = None
        with self._state_lock:
            self._is_leader = False

    async def stop(self) -> None:
        await self.stop_reporting()
        await self.stop_leader_loop()

    def last_seen(self, key: str, dims: Optional[MetricDims] = None) -> float | None:
        with self._state_lock:
            return self._last_seen.get(_normalize_metric(key, dims))

    def is_leader(self) -> bool:
        with self._state_lock:
            return self._is_leader

    @property
    def leader_queue(self) -> str:
        return _METRICS_LEADER_QUEUE

    def cluster_snapshot(self) -> Dict[str, MetricsPayload]:
        with self._state_lock:
            return {
                hostname: payload.model_copy(deep=True)
                for hostname, payload in self._cluster_payloads.items()
            }

    def snapshot(self) -> dict[_NormalizedKey, dict[int, dict[str, float | int | None]]]:
        snap, _ = self._drain_and_process()
        return snap

    def snapshot_as_node_series(self) -> list[NodeMetricSeries]:
        snap, _ = self._drain_and_process()
        with self._state_lock:
            last_seen = dict(self._last_seen)
        return self._snap_to_series(snap, last_seen)

    def _snap_to_series(
        self,
        snap: dict[_NormalizedKey, dict[int, dict[str, float | int | None]]],
        last_seen: dict[_NormalizedKey, float],
    ) -> list[NodeMetricSeries]:
        return [
            NodeMetricSeries(
                key=k[0],
                dims=list(k[1:]),
                windows={win: WindowSample(value=d["value"], n=d["n"]) for win, d in windows.items()},
                last_seen=last_seen.get(k),
            )
            for k, windows in snap.items()
        ]

    def _drain_and_process(
        self,
        collect_1m: bool = False,
    ) -> tuple[dict[_NormalizedKey, dict[int, dict[str, float | int | None]]], list[MetricsBucket]]:
        """Drain pending events then, under a single lock: snapshot windows, optionally collect
        completed 1-minute buckets, trim stale raw buckets."""
        self._drain_events_into_state()
        now = time.time()
        with self._state_lock:
            snap = self._snapshot_locked(now)
            completed = self._collect_completed_1m_buckets_locked(now) if collect_1m else []
            self._trim_buckets_locked(now)
        return snap, completed

    def _run_report_thread(self) -> None:
        asyncio.run(self._report_thread_main())

    def _run_leader_thread(self) -> None:
        asyncio.run(self._leader_thread_main())

    async def _report_thread_main(self) -> None:
        bunny_talk = await BunnyTalk.create(self._logger, self._amqp_url)
        pending_flush: "concurrent.futures.Future[None] | None" = None
        try:
            while not self._report_stop.is_set():
                snap, completed = self._drain_and_process(collect_1m=True)
                with self._state_lock:
                    last_seen = dict(self._last_seen)
                series = self._snap_to_series(snap, last_seen)
                try:
                    payload = self._build_payload(series)
                    await bunny_talk.talk(
                        routing_key=_METRICS_LEADER_QUEUE,
                        payload=payload,
                        start_trace=True,
                    )
                except Exception as e:
                    self._logger.warning(f"MetricsService: Could not report to metrics leader: {e}")
                now = time.time()
                completed_5m = self._collect_completed_5m_buckets_from_1m(completed, now)
                cutoff = now - _PENDING_BUCKET_MAX_AGE_S
                self._pending_1m = [b for b in self._pending_1m if b.bucket_start >= cutoff]
                if completed:
                    self._pending_1m.extend(completed)
                if completed_5m:
                    self._pending_1m.extend(completed_5m)
                if completed or completed_5m:
                    if pending_flush is None or pending_flush.done():
                        pending_flush = self._schedule_flush()
                elif self._pending_1m and (pending_flush is None or pending_flush.done()):
                    pending_flush = self._schedule_flush()
                if await _wait_for_stop(self._report_stop, self._report_interval_s):
                    break
            # Drain on shutdown: wait for any in-flight flush, then do a final one.
            if pending_flush is not None and not pending_flush.done():
                try:
                    pending_flush.result()
                except Exception:
                    pass
            if self._pending_1m:
                self._schedule_flush()
        finally:
            await bunny_talk.close()

    def _schedule_flush(self) -> "concurrent.futures.Future[None]":
        """Schedule _flush_pending_1m on the main event loop and return its Future."""
        return asyncio.run_coroutine_threadsafe(self._flush_pending_1m(), self._main_loop)

    async def _flush_pending_1m(self) -> None:
        batch = list(self._pending_1m)
        try:
            await self._database.append_metric_buckets(self._hostname, self._source, batch)
            # Remove exactly the items we flushed, as the list may have been modified by age eviction
            # or appends while we were awaiting the database.
            batch_ids = {id(b) for b in batch}
            self._pending_1m = [b for b in self._pending_1m if id(b) not in batch_ids]
        except Exception as e:
            self._logger.warning(f"MetricsService: Could not flush 1m metric buckets: {e}")

    async def _trim_metric_buckets(self) -> None:
        now = time.time()
        try:
            await self._database.trim_metric_buckets(
                bucket_seconds=_1M_BUCKET_SECONDS,
                cutoff=now - _1M_RETENTION_S,
            )
        except Exception as e:
            self._logger.warning(f"MetricsService: Could not trim 1m metric buckets: {e}")
        try:
            await self._database.trim_metric_buckets(
                bucket_seconds=_5M_BUCKET_SECONDS,
                cutoff=now - _5M_RETENTION_S,
            )
        except Exception as e:
            self._logger.warning(f"MetricsService: Could not trim 5m metric buckets: {e}")

    async def _leader_thread_main(self) -> None:
        bunny_talk = await BunnyTalk.create(self._logger, self._amqp_url)
        leader_listener = None
        try:
            while not self._leader_stop.is_set():
                try:
                    try_leader = True
                    try:
                        await bunny_talk.talk(
                            routing_key=_METRICS_LEADER_QUEUE,
                            payload=MetricsPayload(probe=True, hostname=self._hostname),
                            start_trace=True,
                        )
                        try_leader = False
                    except Exception:
                        pass

                    if try_leader and leader_listener is None:
                        leader_listener = await bunny_talk.listen(
                            routing_key=_METRICS_LEADER_QUEUE,
                            handler=self._metrics_signal,
                            auto_delete=True,
                            exclusive=True,
                            robust=False,
                        )
                        with self._state_lock:
                            self._is_leader = True
                        self._logger.info("MetricsService: I'm the leader.")
                except Exception as e:
                    leader_listener = None
                    with self._state_lock:
                        self._is_leader = False
                    self._logger.debug(f"MetricsService: Not the leader: {e}")

                now = time.time()
                if self.is_leader() and now - self._last_trimmed_at >= _TRIM_INTERVAL_S:
                    self._last_trimmed_at = now
                    asyncio.run_coroutine_threadsafe(self._trim_metric_buckets(), self._main_loop)

                if await _wait_for_stop(self._leader_stop, self._leader_loop_interval_s):
                    break
        finally:
            with self._state_lock:
                self._is_leader = False
            await bunny_talk.close()

    def _build_payload(self, series: list[NodeMetricSeries]) -> MetricsPayload:
        with self._state_lock:
            meta = dict(self._published_meta)
        return MetricsPayload(
            probe=False,
            hostname=self._hostname,
            source=self._source,
            role=self._role,
            metrics=series,
            meta=meta,
        )

    async def _metrics_signal(self, payload: MetricsPayload) -> Optional[Metrics]:
        if payload.probe:
            return None

        if payload.query_view:
            window = payload.query_window
            key_allow: Optional[frozenset[str]] = None
            if payload.query_keys is not None and len(payload.query_keys) > 0:
                key_allow = frozenset(payload.query_keys)
            with self._state_lock:
                if self._hostname in self._cluster_view:
                    self._cluster_view[self._hostname].is_leader = True
                nodes = {}
                for hostname, node in self._cluster_view.items():
                    node_copy = node.model_copy(deep=True)
                    filtered: List[NodeMetricSeries] = []
                    for s in node_copy.metrics:
                        if key_allow is not None and s.key not in key_allow:
                            continue
                        if window is not None:
                            filtered.append(
                                NodeMetricSeries(
                                    key=s.key,
                                    dims=s.dims,
                                    windows={w: v for w, v in s.windows.items() if w == window},
                                    last_seen=s.last_seen,
                                )
                            )
                        else:
                            filtered.append(s.model_copy(deep=True))
                    node_copy.metrics = filtered
                    nodes[hostname] = node_copy
            return Metrics(nodes=nodes)

        hostname = payload.hostname
        now = time.time()
        source = payload.source
        role = payload.role

        if source is None:
            self._logger.warning(f"Metrics payload from {hostname} is missing source, dropping")
            return None
        if role is None:
            self._logger.warning(f"Metrics payload from {hostname} is missing role, dropping")
            return None

        incoming_meta = payload.meta or {}
        incoming_metrics = payload.metrics or []
        wire_metrics = [
            NodeMetricSeries(
                key=s.key,
                dims=list(s.dims),
                windows={w: v for w, v in s.windows.items() if w in self._cluster_view_windows},
                last_seen=s.last_seen,
            )
            for s in incoming_metrics
        ]
        with self._state_lock:
            host_series = self._cluster_series.setdefault(hostname, {})
            host_series[source] = wire_metrics
            merged_metrics = self._merge_cluster_view_metrics(host_series)
            host_meta = self._cluster_meta.setdefault(hostname, {})
            host_meta[source] = incoming_meta
            merged_meta = {}
            for src_meta in host_meta.values():
                merged_meta.update(src_meta)
            self._cluster_view[hostname] = NodeView(
                role=role,
                last_seen=now,
                meta=merged_meta,
                metrics=merged_metrics,
            )
            self._cluster_payloads[hostname] = MetricsPayload(
                probe=False,
                hostname=hostname,
                source=source,
                role=role,
                metrics=incoming_metrics,
                meta=incoming_meta,
            )
            self._cleanup_stale_metrics_locked(now)
        return None

    @staticmethod
    def _merge_cluster_view_metrics(
        host_series: Dict[str, List[NodeMetricSeries]],
    ) -> List[NodeMetricSeries]:
        merged: Dict[Tuple[str, ...], NodeMetricSeries] = {}
        for series_list in host_series.values():
            for series in series_list:
                merged[_series_identity(series)] = series
        return list(merged.values())

    def _collect_completed_1m_buckets_locked(self, now: float) -> list[MetricsBucket]:
        current_1m_start = int(now // 60) * 60
        if current_1m_start <= self._last_flushed_1m_start:
            return []
        result: list[MetricsBucket] = []
        for k, agg in self._agg.items():
            merged: dict[int, MetricsBucket] = {}
            for bucket in self._data[k]:
                slot = int(bucket.bucket_start // 60) * 60
                if slot < self._last_flushed_1m_start or slot >= current_1m_start:
                    continue
                if slot not in merged:
                    merged[slot] = MetricsBucket(
                        key=k[0],
                        dims=list(k[1:]),
                        bucket_start=slot,
                        bucket_seconds=60,
                        value=0.0,
                        n=0 if agg is _Agg.AVG else None,
                    )
                merged[slot].value += bucket.value
                if agg is _Agg.AVG and bucket.n is not None:
                    merged[slot].n = (merged[slot].n or 0) + bucket.n
            result.extend(merged.values())
        low = self._last_flushed_1m_start
        has_raw_in_completed = False
        for k in self._data:
            for bucket in self._data[k]:
                slot = int(bucket.bucket_start // 60) * 60
                if low <= slot < current_1m_start:
                    has_raw_in_completed = True
                    break
            if has_raw_in_completed:
                break
        if result or not has_raw_in_completed:
            self._last_flushed_1m_start = current_1m_start
        return result

    def _collect_completed_5m_buckets_from_1m(
        self, completed_1m: list[MetricsBucket], now: float
    ) -> list[MetricsBucket]:
        current_5m_start = int(now // _5M_BUCKET_SECONDS) * _5M_BUCKET_SECONDS
        if current_5m_start <= self._last_flushed_5m_start:
            return []

        cutoff = current_5m_start - _PENDING_BUCKET_MAX_AGE_S
        self._pending_1m_for_5m = [b for b in self._pending_1m_for_5m if b.bucket_start >= cutoff]
        self._pending_1m_for_5m.extend(completed_1m)

        merged: dict[tuple, MetricsBucket] = {}
        is_avg: dict[tuple, bool] = {}
        low_5m = self._last_flushed_5m_start
        for bucket in self._pending_1m_for_5m:
            slot = int(bucket.bucket_start // _5M_BUCKET_SECONDS) * _5M_BUCKET_SECONDS
            if slot < low_5m or slot >= current_5m_start:
                continue
            identity = (bucket.key, *bucket.dims, slot)
            avg = bucket.n is not None
            if identity not in merged:
                is_avg[identity] = avg
                merged[identity] = MetricsBucket(
                    key=bucket.key,
                    dims=list(bucket.dims),
                    bucket_start=slot,
                    bucket_seconds=_5M_BUCKET_SECONDS,
                    value=0.0,
                    n=0 if avg else None,
                )
            merged[identity].value += bucket.value
            if is_avg[identity] and bucket.n is not None:
                merged[identity].n = (merged[identity].n or 0) + bucket.n

        has_1m_in_completed_5m = any(
            low_5m
            <= int(b.bucket_start // _5M_BUCKET_SECONDS) * _5M_BUCKET_SECONDS
            < current_5m_start
            for b in self._pending_1m_for_5m
        )
        if merged or not has_1m_in_completed_5m:
            self._pending_1m_for_5m = [
                b for b in self._pending_1m_for_5m
                if int(b.bucket_start // _5M_BUCKET_SECONDS) * _5M_BUCKET_SECONDS >= current_5m_start
            ]
            self._last_flushed_5m_start = current_5m_start
        return list(merged.values())

    def _record_event(self, k: _NormalizedKey, value: float, agg: _Agg, n: Optional[int]) -> None:
        self._events.append(
            _MetricEvent(
                key=k,
                value=value,
                agg=agg,
                n=n,
                wall_ts=time.time(),
            )
        )

    def _drain_events_into_state(self) -> None:
        drained: list[_MetricEvent] = []
        while True:
            try:
                drained.append(self._events.popleft())
            except IndexError:
                break
        if not drained:
            return
        with self._state_lock:
            for event in drained:
                existing = self._agg.get(event.key)
                if existing is None:
                    self._agg[event.key] = event.agg
                    self._data[event.key] = deque()
                elif existing is not event.agg:
                    self._logger.error(
                        f"MetricsService: key {event.key!r} recorded as both "
                        f"{existing.value!r} and {event.agg.value!r}; event dropped"
                    )
                    continue
                self._last_seen[event.key] = event.wall_ts
                bucket_start = _bucket_start(event.wall_ts)
                buckets = self._data[event.key]
                if buckets and buckets[-1].bucket_start == bucket_start:
                    bucket = buckets[-1]
                else:
                    bucket = MetricsBucket(
                        key=event.key[0],
                        dims=list(event.key[1:]),
                        bucket_start=bucket_start,
                        bucket_seconds=_LIVE_BUCKET_SECONDS,
                        value=0.0,
                        n=0 if event.agg is _Agg.AVG else None,
                    )
                    buckets.append(bucket)
                bucket.value += event.value
                if event.agg is _Agg.AVG:
                    if bucket.n is None:
                        bucket.n = 0
                    bucket.n += 0 if event.n is None else event.n

    def _snapshot_locked(self, now: float) -> dict[_NormalizedKey, dict[int, dict[str, float | int | None]]]:
        current_bucket_start = _bucket_start(now)
        result: dict[_NormalizedKey, dict[int, dict[str, float | int | None]]] = {}

        for k, agg in self._agg.items():
            buckets = self._data[k]
            windows: dict[int, dict[str, float | int | None]] = {}
            for win in self._windows:
                bucket_cut = current_bucket_start - win + _LIVE_BUCKET_SECONDS
                total = 0.0
                count = 0
                for bucket in reversed(buckets):
                    if bucket.bucket_start < bucket_cut:
                        break
                    total += bucket.value
                    if agg is _Agg.AVG and bucket.n is not None:
                        count += bucket.n

                windows[win] = {
                    "value": total,
                    "n": count if agg is _Agg.AVG else None,
                }

            result[k] = windows

        return result

    def _trim_buckets_locked(self, now: float) -> None:
        current_bucket_start = _bucket_start(now)
        window_cutoff = current_bucket_start - self._max_window + _LIVE_BUCKET_SECONDS
        # Never trim buckets that belong to a 1m slot not yet flushed to DB.
        trim_cutoff = min(window_cutoff, self._last_flushed_1m_start)
        empty_keys = []
        for k, buckets in self._data.items():
            while buckets and buckets[0].bucket_start < trim_cutoff:
                buckets.popleft()
            if not buckets:
                empty_keys.append(k)
        for k in empty_keys:
            del self._data[k]
            del self._agg[k]
            self._last_seen.pop(k, None)
        self._trim_last_used_locked(now)

    def _trim_last_used_locked(self, now: float) -> None:
        if self._last_used_ttl_seconds is None:
            return
        cutoff = now - self._last_used_ttl_seconds
        stale_keys = [k for k, ts in self._last_used.items() if ts < cutoff]
        for k in stale_keys:
            del self._last_used[k]

    def _cleanup_stale_metrics_locked(self, now: float) -> None:
        stale_view = [
            hostname for hostname, data in self._cluster_view.items()
            if data.last_seen < now - self._metrics_node_expiry_seconds
        ]
        for hostname in stale_view:
            del self._cluster_view[hostname]
            self._cluster_payloads.pop(hostname, None)
            self._cluster_meta.pop(hostname, None)
            self._cluster_series.pop(hostname, None)
            self._logger.info(f"MetricsService: Expired cluster view for {hostname} (stale)")


def _normalize_metric(key: str, dims: Optional[MetricDims]) -> _NormalizedKey:
    if dims is None:
        return (key,)
    if isinstance(dims, str):
        return (key, dims)
    return (key, *tuple(dims))


def _series_identity(series: NodeMetricSeries) -> Tuple[str, ...]:
    return (series.key, *series.dims)


def _bucket_start(ts: float) -> int:
    return int(ts // _LIVE_BUCKET_SECONDS) * _LIVE_BUCKET_SECONDS


async def _wait_for_stop(stop_event: threading.Event, timeout_s: float) -> bool:
    return await asyncio.to_thread(stop_event.wait, timeout_s)
