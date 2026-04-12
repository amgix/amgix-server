"""HTTP request metrics for the API process (rolling windows, shipped to encoder leader)."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.common.rolling_metrics import RollingMetrics
from src.core.models.cluster import NodeMetricSeries, WindowSample

API_METRIC_WINDOWS = [30, 60]

_rolling = RollingMetrics(API_METRIC_WINDOWS)

_CLASSIFIED_OPERATIONS: dict[str, tuple[str, str]] = {
    "upsert_document": ("api_async_upload", "api_async_upload_ms"),
    "upsert_document_sync": ("api_sync_upload", "api_sync_upload_ms"),
    "upsert_documents_bulk": ("api_bulk_upload", "api_bulk_upload_ms"),
    "search": ("api_search", "api_search_ms"),
}


def record_api_http_request(request: Request, duration_ms: float) -> None:
    _rolling.record_rate(("api_requests",))
    _rolling.record_avg(("api_request_ms",), duration_ms)
    route = request.scope.get("route")
    op_id = getattr(route, "operation_id", None) if route is not None else None
    if op_id and op_id in _CLASSIFIED_OPERATIONS:
        rate_key, ms_key = _CLASSIFIED_OPERATIONS[op_id]
        _rolling.record_rate((rate_key,))
        _rolling.record_avg((ms_key,), duration_ms)


def _record_api_http_status_errors(status_code: int) -> None:
    if 400 <= status_code <= 499:
        _rolling.record_rate(("api_error_4xx",))
    elif status_code >= 500:
        _rolling.record_rate(("api_error_5xx",))


def snapshot_as_node_metric_series() -> list[NodeMetricSeries]:
    return [
        NodeMetricSeries(
            key=list(k),
            windows={win: WindowSample(value=d["value"], n=d["n"]) for win, d in windows.items()},
            last_seen=_rolling.last_seen(k),
        )
        for k, windows in _rolling.snapshot().items()
    ]


class ApiMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter_ns()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            _rolling.record_rate(("api_error_5xx",))
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            record_api_http_request(request, elapsed_ms)
            if response is not None:
                _record_api_http_status_errors(response.status_code)
