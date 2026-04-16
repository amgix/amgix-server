import math
import time
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, List, Tuple

from src.core.common.bunny_talk import BunnyTalk
from src.core.common.metrics_definitions import MetricKey
from src.core.models.cluster import MetricsPayload

NODE_META_LOAD_MODELS = "load_models"
NODE_META_AT_CAPACITY = "at_capacity"
NODE_META_LOADED_MODELS = "loaded_models"
NODE_META_MODEL_LAST_USED = "model_last_used"
NODE_META_MODEL_KEY = "model_key"
NODE_META_LOADED_AT = "loaded_at"
NODE_META_LAST_USED_AT = "last_used_at"


@dataclass(frozen=True)
class _PendingSignal:
    load: bool
    sent_at: float


class ModelRebalancer:
    def __init__(
        self,
        bunny_talk: BunnyTalk,
        logger: Logger,
        windows: list[int],
        *,
        leader_loop_interval_s: float,
        model_idle_grace_seconds: int,
        target_availability_pct: int,
        node_queue_prefix: str,
    ) -> None:
        if not windows:
            raise ValueError("windows must not be empty")
        self._bunny_talk = bunny_talk
        self._logger = logger
        self._windows = sorted(windows)
        self._leader_loop_interval_s = leader_loop_interval_s
        self._model_idle_grace_seconds = model_idle_grace_seconds
        self._target_availability_pct = target_availability_pct
        self._node_queue_prefix = node_queue_prefix
        self._cluster_metrics: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {}
        self._pending_signals: Dict[Tuple[str, Tuple[str, str, str]], _PendingSignal] = {}

    async def rebalance(self, snapshot: Dict[str, MetricsPayload]) -> None:
        self._cluster_metrics = self._build_cluster_metrics(snapshot)
        self._reconcile_pending_signals()
        await self._rebalance_models()

    def _build_cluster_metrics(
        self,
        snapshot: Dict[str, MetricsPayload],
    ) -> Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]]:
        cluster_metrics: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {}
        now = time.time()
        for hostname, payload in snapshot.items():
            if payload.source != "router":
                continue
            incoming_meta = payload.meta if isinstance(payload.meta, dict) else {}
            loaded_models_meta = _meta_loaded_models(incoming_meta)
            model_last_used_meta = _meta_model_last_used(incoming_meta)
            incoming_metrics = payload.metrics or []
            rebalance_metrics: Dict[tuple, Dict[str, Dict[str, float]]] = {}
            for series in incoming_metrics:
                if series.key == MetricKey.EMBED_BATCHES_ORIGIN and len(series.dims) == 3:
                    mk = tuple(series.dims)
                    rebalance_metrics[mk] = {
                        str(win): {"value": sample.value, "n": sample.n}
                        for win, sample in series.windows.items()
                    }
            cluster_metrics[hostname] = {
                "metrics": rebalance_metrics,
                "model_last_used": {
                    tuple(entry[NODE_META_MODEL_KEY]): float(entry[NODE_META_LAST_USED_AT])
                    for entry in model_last_used_meta
                },
                "loaded_models": [
                    (tuple(entry[NODE_META_MODEL_KEY]), float(entry[NODE_META_LOADED_AT]))
                    for entry in loaded_models_meta
                ],
                "load_models": _meta_bool(incoming_meta, NODE_META_LOAD_MODELS),
                "capacity": 0 if _meta_bool(incoming_meta, NODE_META_AT_CAPACITY) else 1 if len(loaded_models_meta) > 0 else 2,
                "last_seen": now,
            }
        return cluster_metrics

    async def _rebalance_models(self) -> None:
        start_ns = time.monotonic_ns()
        self._logger.debug("ModelRebalancer: Rebalancing models --------------------------------")

        async def _send_signal(hostname: str, model_key: tuple[str, str, str], load: bool) -> None:
            if self._should_skip_pending_signal(hostname, model_key, load):
                return
            node_queue_name = f"{self._node_queue_prefix}-{hostname}"
            try:
                self._logger.debug(f"ModelRebalancer: Sending signal to {hostname} for {model_key}: load={load}")
                await self._bunny_talk.talk(node_queue_name, load=load, model_key=list(model_key), start_trace=True)
                self._pending_signals[(hostname, model_key)] = _PendingSignal(load=load, sent_at=time.time())
            except Exception as e:
                self._logger.warning(f"ModelRebalancer: Failed to send signal to {hostname} for {model_key} (load={load}): {e}")

        st_host_count = 0
        st_available_count = 0
        available_st_hosts: Dict[str, int] = {}

        for hostname, data in self._cluster_metrics.items():
            if data.get("load_models", True):
                st_host_count += 1
                if data["capacity"] > 0:
                    available_st_hosts[hostname] = data["capacity"]
                    st_available_count += 1

        st_reservations = max(1, math.floor(self._target_availability_pct * st_host_count / 100))
        st_direction = st_available_count - st_reservations

        model_list = set(
            model_tuple[0]
            for host_data in self._cluster_metrics.values()
            for model_tuple in host_data["loaded_models"]
            if host_data["loaded_models"]
        )

        max_window = max(self._windows)
        cluster_rps: dict[tuple, dict[str, float]] = {}
        cluster_last_used: dict[tuple, float] = {}
        for host_data in self._cluster_metrics.values():
            for mk, windows in host_data["metrics"].items():
                if mk not in cluster_rps:
                    cluster_rps[mk] = {}
                for win_str, d in windows.items():
                    window = int(win_str)
                    total = d.get("value", 0.0)
                    cluster_rps[mk][win_str] = cluster_rps[mk].get(win_str, 0.0) + (
                        total / window if window > 0 else 0.0
                    )
            for mk, ts in host_data.get("model_last_used", {}).items():
                if ts > cluster_last_used.get(mk, 0.0):
                    cluster_last_used[mk] = ts

        total_rps = 0.0
        models: Dict[tuple, Dict[str, Any]] = {}
        for model_key in model_list:
            metrics = cluster_rps.get(model_key, {})
            weighted_rps = 0.0
            for window in self._windows:
                window_rps = metrics.get(str(window), 0.0)
                weight = max_window / window
                weighted_rps += window_rps * weight

            models[model_key] = {
                "hosts": sorted(
                    [
                        (k, v["capacity"], model_tuple[1])
                        for k, v in self._cluster_metrics.items()
                        for model_tuple in v["loaded_models"]
                        if model_tuple[0] == model_key
                    ],
                    key=lambda x: x[1],
                ),
                "host_count": 0,
                "weighted_rps": weighted_rps,
                "proportion": 0.0,
            }
            total_rps += weighted_rps

        best_add_st = None
        best_score_st = -1.0
        second_best_score_st = -1.0
        best_remove_st = None
        best_remove_score_st = float("inf")
        second_best_remove_score_st = float("inf")

        for model_key, data in models.items():
            data["host_count"] = len(data["hosts"])
            data["proportion"] = data["weighted_rps"] / total_rps if total_rps > 0 else 0.0
            data["score"] = data["proportion"] / (data["host_count"] + 1)
            data["target_hosts"] = [
                (host, capacity)
                for host, capacity in available_st_hosts.items()
                if host not in (h for h, _, _ in data["hosts"])
            ]

            if data["weighted_rps"] > 0 and data["target_hosts"] and data["score"] > best_score_st:
                second_best_score_st = best_score_st
                best_score_st = data["score"]
                best_add_st = (model_key, data)
            elif data["weighted_rps"] > 0 and data["target_hosts"] and data["score"] > second_best_score_st:
                second_best_score_st = data["score"]

            if data["weighted_rps"] > 0 and data["host_count"] > 1:
                remove_score = data["proportion"] / data["host_count"]
                if remove_score < best_remove_score_st:
                    second_best_remove_score_st = best_remove_score_st
                    best_remove_score_st = remove_score
                    best_remove_st = (model_key, data)
                elif remove_score < second_best_remove_score_st:
                    second_best_remove_score_st = remove_score

        current_time = time.time()
        for model_key, data in [(k, v) for k, v in models.items() if v["weighted_rps"] == 0]:
            last_active = cluster_last_used.get(model_key, 0.0)
            idle_seconds = current_time - last_active
            if idle_seconds <= self._model_idle_grace_seconds:
                continue
            if data["host_count"] > 1:
                hostname = data["hosts"][0][0]
                self._logger.debug(
                    f"ModelRebalancer: Unloading model {model_key} from {hostname} "
                    f"(idle {int(idle_seconds)}s, no rps)."
                )
                await _send_signal(hostname, model_key, False)
            elif data["host_count"] == 1:
                hostname, _, load_timestamp = data["hosts"][0]
                if current_time - load_timestamp > self._model_idle_grace_seconds:
                    self._logger.debug(
                        f"ModelRebalancer: Unloading model {model_key} from {hostname} "
                        f"(idle {int(idle_seconds)}s, loaded {int(current_time - load_timestamp)}s ago, no rps)."
                    )
                    await _send_signal(hostname, model_key, False)

        capable_hosts_count = sum(1 for _, d in self._cluster_metrics.items() if d.get("load_models", True))
        if capable_hosts_count < 3:
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
            self._logger.debug(
                f"ModelRebalancer: Skipping demand-based rebalancing: only {capable_hosts_count} capable host(s)."
            )
            self._logger.debug(f"ModelRebalancer: Done rebalancing models in {elapsed_ms:.3f}ms ---------- ")
            return

        if st_direction == 0:
            swap_add_st = None
            swap_add_score_st = -1.0
            swap_remove_st = None
            swap_remove_score_st = float("inf")

            for m_key, m_data in models.items():
                if m_data["weighted_rps"] == 0:
                    continue
                if m_data["score"] > swap_add_score_st:
                    swap_add_score_st = m_data["score"]
                    swap_add_st = (m_key, m_data)
                if m_data["host_count"] > 1:
                    rm_score = m_data["proportion"] / m_data["host_count"]
                    if rm_score < swap_remove_score_st:
                        swap_remove_score_st = rm_score
                        swap_remove_st = (m_key, m_data)

            if swap_add_st and swap_remove_st and swap_add_st[0] != swap_remove_st[0]:
                add_key, add_data = swap_add_st
                remove_key, remove_data = swap_remove_st
                add_hosts = set(h for h, _, _ in add_data["hosts"])
                remove_host_tuple = next(
                    ((h, cap, ts) for h, cap, ts in remove_data["hosts"] if h not in add_hosts),
                    None,
                )
                if remove_host_tuple:
                    hostname, _, load_ts = remove_host_tuple
                    current_time = time.time()
                    if not load_ts or current_time - load_ts >= self._leader_loop_interval_s * 2:
                        self._logger.debug(
                            f"ModelRebalancer: ST swap: unloading {remove_key} from {hostname} "
                            f"to make room for {add_key}"
                        )
                        await _send_signal(hostname, remove_key, False)

        if st_direction > 0 and best_add_st:
            model_key, data = best_add_st
            hostname = max(data["target_hosts"], key=lambda x: x[1])[0]
            self._logger.debug(f"ModelRebalancer: Adding {model_key} to {hostname} (score: {best_score_st:.3f})")
            await _send_signal(hostname, model_key, True)
        elif st_direction < 0 and best_remove_st:
            model_key, data = best_remove_st
            hostname = min(data["hosts"], key=lambda x: x[1])[0]
            current_time = time.time()
            _, _, load_timestamp = next((h for h in data["hosts"] if h[0] == hostname), (None, None, None))
            if load_timestamp and current_time - load_timestamp < self._leader_loop_interval_s * 2:
                self._logger.debug(
                    f"ModelRebalancer: Skipping removal of {model_key} from {hostname} "
                    f"(too recent: {int(current_time - load_timestamp)}s)"
                )
            else:
                self._logger.debug(
                    f"ModelRebalancer: Removing {model_key} from {hostname} (score: {best_remove_score_st:.3f})"
                )
                await _send_signal(hostname, model_key, False)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        self._logger.debug(f"ModelRebalancer: Done rebalancing models in {elapsed_ms:.3f}ms ---------- ")

    def _pending_signal_timeout_seconds(self) -> float:
        return max(self._leader_loop_interval_s * 3.0, 15.0)

    def _reconcile_pending_signals(self) -> None:
        now = time.time()
        timeout_s = self._pending_signal_timeout_seconds()
        cleared: List[Tuple[str, Tuple[str, str, str]]] = []
        expired: List[Tuple[str, Tuple[str, str, str], _PendingSignal]] = []
        for key, pending in self._pending_signals.items():
            hostname, model_key = key
            host_data = self._cluster_metrics.get(hostname)
            if host_data is None:
                cleared.append(key)
                continue
            is_loaded = _host_has_loaded_model(host_data, model_key)
            if is_loaded == pending.load:
                cleared.append(key)
                continue
            if now - pending.sent_at > timeout_s:
                expired.append((hostname, model_key, pending))
                cleared.append(key)
        for key in cleared:
            self._pending_signals.pop(key, None)
        for hostname, model_key, pending in expired:
            self._logger.warning(
                "ModelRebalancer: Timed out waiting for %s confirmation from %s for %s; allowing retry.",
                "load" if pending.load else "unload",
                hostname,
                model_key,
            )

    def _should_skip_pending_signal(self, hostname: str, model_key: tuple[str, str, str], load: bool) -> bool:
        pending = self._pending_signals.get((hostname, model_key))
        if pending is None or pending.load != load:
            return False
        if time.time() - pending.sent_at > self._pending_signal_timeout_seconds():
            return False
        self._logger.debug(
            f"ModelRebalancer: Suppressing duplicate signal to {hostname} for {model_key}: load={load}"
        )
        return True


def _meta_bool(meta: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = meta.get(key)
    return bool(value) if value is not None else default


def _host_has_loaded_model(host_data: Dict[str, Any], model_key: Tuple[str, str, str]) -> bool:
    return any(loaded_key == model_key for loaded_key, _ in host_data.get("loaded_models", []))


def _meta_loaded_models(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = meta.get(NODE_META_LOADED_MODELS)
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_key = item.get(NODE_META_MODEL_KEY)
        raw_loaded_at = item.get(NODE_META_LOADED_AT)
        if not isinstance(raw_key, list) or not all(isinstance(part, str) for part in raw_key):
            continue
        if not isinstance(raw_loaded_at, (int, float)):
            continue
        out.append(
            {
                NODE_META_MODEL_KEY: list(raw_key),
                NODE_META_LOADED_AT: float(raw_loaded_at),
            }
        )
    return out


def _meta_model_last_used(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = meta.get(NODE_META_MODEL_LAST_USED)
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_key = item.get(NODE_META_MODEL_KEY)
        raw_last_used_at = item.get(NODE_META_LAST_USED_AT)
        if not isinstance(raw_key, list) or not all(isinstance(part, str) for part in raw_key):
            continue
        if not isinstance(raw_last_used_at, (int, float)):
            continue
        out.append(
            {
                NODE_META_MODEL_KEY: list(raw_key),
                NODE_META_LAST_USED_AT: float(raw_last_used_at),
            }
        )
    return out
