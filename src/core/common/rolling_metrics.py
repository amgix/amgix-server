"""
Generic rolling-window metrics collector.

Usage:
    metrics = RollingMetrics(windows=[5, 30, 60, 300])

    # Record values — aggregation strategy declared at record time.
    metrics.record_rate(("batches", model_key))          # count/sec
    metrics.record_avg(("inference_ms", model_key), ms)  # mean over window
    metrics.record_sum(("errors", model_key))             # raw count in window

    # Get all windows for all tracked keys.
    snap = metrics.snapshot()
    # snap[("inference_ms", model_key)][30] == {"value": 210.3, "n": 42}
"""

import time
from collections import deque
from enum import Enum
from typing import Union

MetricKey = Union[str, tuple[str, ...]]
_NormalizedKey = tuple[str, ...]


class _Agg(Enum):
    AVG = "avg"
    RATE = "rate"
    SUM = "sum"


class RollingMetrics:
    """
    Thread-safe rolling-window metrics collector.

    Each key tracks one stream of float samples. The aggregation strategy
    (avg, rate, sum) is declared on first use and cannot change for that key.

    - avg:  mean(values) over the window
    - rate: sum(values) / window_seconds  (record 1.0 per event for events/sec,
            or a magnitude like bytes for bytes/sec)
    - sum:  sum(values) over the window  (raw count or total magnitude)
    """

    def __init__(self, windows: list[int]) -> None:
        if not windows:
            raise ValueError("windows must not be empty")
        self._windows: list[int] = sorted(windows)
        self._max_window: int = max(windows)
        self._data: dict[_NormalizedKey, deque[tuple[float, float]]] = {}
        self._agg: dict[_NormalizedKey, _Agg] = {}
        self._last_seen: dict[_NormalizedKey, float] = {}

    # ------------------------------------------------------------------
    # Public record API
    # ------------------------------------------------------------------

    def record_avg(self, key: MetricKey, value: float) -> None:
        """Record a measurement to be averaged over each window."""
        self._record(_normalize(key), value, _Agg.AVG)

    def record_rate(self, key: MetricKey, value: float = 1.0) -> None:
        """Record an event (or magnitude) contributing to a per-second rate."""
        self._record(_normalize(key), value, _Agg.RATE)

    def record_sum(self, key: MetricKey, value: float = 1.0) -> None:
        """Record an increment to be summed over each window."""
        self._record(_normalize(key), value, _Agg.SUM)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[_NormalizedKey, dict[int, dict[str, float | int]]]:
        """
        Return aggregated results for every tracked key and every window.

        Shape:
            {
                ("inference_ms", "model_x"): {
                    30:  {"value": 210.3, "n": 42},
                    300: {"value": 198.7, "n": 401},
                },
                ...
            }

        "value" is computed according to the aggregation strategy registered
        for that key. "n" is always the raw sample count in the window.
        """
        now = time.monotonic()
        cutoff = now - self._max_window
        result: dict[_NormalizedKey, dict[int, dict[str, float | int]]] = {}

        for k, agg in list(self._agg.items()):
            dq = self._data[k]

            # Prune samples older than the longest window.
            # deque.popleft() is GIL-atomic; no lock needed for the deque itself.
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            windows: dict[int, dict[str, float | int]] = {}
            for win in self._windows:
                win_cut = now - win
                n = 0
                total = 0.0
                for ts, val in reversed(dq):
                    if ts < win_cut:
                        break
                    n += 1
                    total += val

                if agg is _Agg.AVG:
                    computed = total / n if n > 0 else 0.0
                elif agg is _Agg.RATE:
                    computed = total / win if win > 0 else 0.0
                else:  # SUM
                    computed = total

                windows[win] = {"value": computed, "n": n}

            result[k] = windows

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def last_seen(self, key: MetricKey) -> float | None:
        """Return the UTC timestamp (time.time()) of the most recent record for this key, or None."""
        return self._last_seen.get(_normalize(key))

    def _record(self, k: _NormalizedKey, value: float, agg: _Agg) -> None:
        mono = time.monotonic()
        existing = self._agg.get(k)
        if existing is None:
            self._agg[k] = agg
            self._data[k] = deque()
        elif existing is not agg:
            raise ValueError(
                f"Key {k!r} was registered as {existing.value!r}; "
                f"cannot record as {agg.value!r}"
            )
        self._last_seen[k] = time.time()
        self._data[k].append((mono, value))


def _normalize(key: MetricKey) -> _NormalizedKey:
    return (key,) if isinstance(key, str) else tuple(key)
