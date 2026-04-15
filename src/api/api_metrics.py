"""HTTP request metrics for the API process (rolling windows, shipped to encoder leader)."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from logging import Logger

from src.core.common.metrics_definitions import MetricKey
from src.core.common.metrics_service import MetricsService
from src.core.database.base import DatabaseBase

API_METRIC_WINDOWS = [30, 60]

api_metrics: MetricsService


def init_api_metrics_service(
    amqp_url: str,
    logger: Logger,
    hostname: str,
    database: DatabaseBase,
) -> MetricsService:
    global api_metrics
    api_metrics = MetricsService(
        amqp_url=amqp_url,
        logger=logger,
        hostname=hostname,
        source="api",
        role="api",
        windows=API_METRIC_WINDOWS,
        database=database,
    )
    return api_metrics

_CLASSIFIED_OPERATIONS: dict[str, tuple[MetricKey, MetricKey]] = {
    "upsert_document":       (MetricKey.API_ASYNC_UPLOAD, MetricKey.API_ASYNC_UPLOAD_MS),
    "upsert_document_sync":  (MetricKey.API_SYNC_UPLOAD,  MetricKey.API_SYNC_UPLOAD_MS),
    "upsert_documents_bulk": (MetricKey.API_BULK_UPLOAD,  MetricKey.API_BULK_UPLOAD_MS),
    "search":                (MetricKey.API_SEARCH,        MetricKey.API_SEARCH_MS),
}


def record_api_http_request(request: Request, duration_ms: float) -> None:
    api_metrics.record(MetricKey.API_REQUESTS)
    api_metrics.record(MetricKey.API_REQUEST_MS, duration_ms, n=1)
    route = request.scope.get("route")
    op_id = getattr(route, "operation_id", None) if route is not None else None
    if op_id and op_id in _CLASSIFIED_OPERATIONS:
        rate_key, ms_key = _CLASSIFIED_OPERATIONS[op_id]
        api_metrics.record(rate_key)
        api_metrics.record(ms_key, duration_ms, n=1)


def _record_api_http_status_errors(status_code: int) -> None:
    if 400 <= status_code <= 499:
        api_metrics.record(MetricKey.API_ERROR_4XX)
    elif status_code >= 500:
        api_metrics.record(MetricKey.API_ERROR_5XX)


class ApiMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        t0 = time.perf_counter_ns()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            api_metrics.record(MetricKey.API_ERROR_5XX)
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            record_api_http_request(request, elapsed_ms)
            if response is not None:
                _record_api_http_status_errors(response.status_code)
