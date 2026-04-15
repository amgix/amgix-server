from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MetricKey(StrEnum):
    API_REQUESTS = "api_requests"
    API_REQUEST_MS = "api_request_ms"
    API_ASYNC_UPLOAD = "api_async_upload"
    API_ASYNC_UPLOAD_MS = "api_async_upload_ms"
    API_SYNC_UPLOAD = "api_sync_upload"
    API_SYNC_UPLOAD_MS = "api_sync_upload_ms"
    API_BULK_UPLOAD = "api_bulk_upload"
    API_BULK_UPLOAD_MS = "api_bulk_upload_ms"
    API_SEARCH = "api_search"
    API_SEARCH_MS = "api_search_ms"
    API_ERROR_4XX = "api_error_4xx"
    API_ERROR_5XX = "api_error_5xx"
    EMBED_BATCHES_ORIGIN = "embed_batches_origin"
    EMBED_PASSAGES_ORIGIN = "embed_passages_origin"
    EMBED_INFERENCE_ORIGIN_MS = "embed_inference_origin_ms"
    EMBED_INFERENCE_ORIGIN_ERRORS = "embed_inference_origin_errors"
    EMBED_BATCHES = "embed_batches"
    EMBED_PASSAGES = "embed_passages"
    EMBED_INFERENCE_MS = "embed_inference_ms"
    EMBED_HOPS = "embed_hops"
    INDEX_QUEUE_DOCS_SKIPPED_STALE = "index_queue_docs_skipped_stale"
    INDEX_QUEUE_DOCS_NEW = "index_queue_docs_new"
    INDEX_QUEUE_DOCS_UPDATED = "index_queue_docs_updated"
    INDEX_QUEUE_FAILED = "index_queue_failed"
    INDEX_QUEUE_REQUEUED = "index_queue_requeued"
    INDEX_QUEUE_JOB_MS = "index_queue_job_ms"
    INDEX_BULK_BATCHES = "index_bulk_batches"
    INDEX_BULK_BATCH_SIZE = "index_bulk_batch_size"
    INDEX_BULK_FAILED = "index_bulk_failed"
    INDEX_BULK_REQUEUED = "index_bulk_requeued"
    INDEX_BULK_JOB_MS = "index_bulk_job_ms"


@dataclass(frozen=True)
class MetricDefinition:
    unit: str
    description: str


METRIC_DEFINITIONS: dict[MetricKey, MetricDefinition] = {
    MetricKey.API_REQUESTS: MetricDefinition(
        unit="req",
        description="Total HTTP requests handled by the API process.",
    ),
    MetricKey.API_REQUEST_MS: MetricDefinition(
        unit="ms",
        description="Total request latency accumulated across all API requests.",
    ),
    MetricKey.API_ASYNC_UPLOAD: MetricDefinition(
        unit="req",
        description="Total async single-document upload requests handled by the API process.",
    ),
    MetricKey.API_ASYNC_UPLOAD_MS: MetricDefinition(
        unit="ms",
        description="Total latency accumulated across async single-document upload requests.",
    ),
    MetricKey.API_SYNC_UPLOAD: MetricDefinition(
        unit="req",
        description="Total sync single-document upload requests handled by the API process.",
    ),
    MetricKey.API_SYNC_UPLOAD_MS: MetricDefinition(
        unit="ms",
        description="Total latency accumulated across sync single-document upload requests.",
    ),
    MetricKey.API_BULK_UPLOAD: MetricDefinition(
        unit="req",
        description="Total bulk upload requests handled by the API process.",
    ),
    MetricKey.API_BULK_UPLOAD_MS: MetricDefinition(
        unit="ms",
        description="Total latency accumulated across bulk upload requests.",
    ),
    MetricKey.API_SEARCH: MetricDefinition(
        unit="req",
        description="Total search requests handled by the API process.",
    ),
    MetricKey.API_SEARCH_MS: MetricDefinition(
        unit="ms",
        description="Total latency accumulated across search requests.",
    ),
    MetricKey.API_ERROR_4XX: MetricDefinition(
        unit="err",
        description="Total API responses with 4xx status codes.",
    ),
    MetricKey.API_ERROR_5XX: MetricDefinition(
        unit="err",
        description="Total API responses with 5xx status codes.",
    ),
    MetricKey.EMBED_BATCHES_ORIGIN: MetricDefinition(
        unit="batch",
        description="Total embedding request batches originating on this encoder node.",
    ),
    MetricKey.EMBED_PASSAGES_ORIGIN: MetricDefinition(
        unit="passage",
        description="Total passages from embedding requests originating on this encoder node.",
    ),
    MetricKey.EMBED_INFERENCE_ORIGIN_MS: MetricDefinition(
        unit="ms",
        description="Total end-to-end embedding latency accumulated on the originating encoder node.",
    ),
    MetricKey.EMBED_INFERENCE_ORIGIN_ERRORS: MetricDefinition(
        unit="err",
        description="Total failed embedding requests originating on this encoder node.",
    ),
    MetricKey.EMBED_BATCHES: MetricDefinition(
        unit="batch",
        description="Total embedding request batches executed on this node.",
    ),
    MetricKey.EMBED_PASSAGES: MetricDefinition(
        unit="passage",
        description="Total passages embedded on this node.",
    ),
    MetricKey.EMBED_INFERENCE_MS: MetricDefinition(
        unit="ms",
        description="Total model inference latency accumulated on this node.",
    ),
    MetricKey.EMBED_HOPS: MetricDefinition(
        unit="hop",
        description="Total routing hops accumulated for embedding requests handled on this node.",
    ),
    MetricKey.INDEX_QUEUE_DOCS_SKIPPED_STALE: MetricDefinition(
        unit="doc",
        description="Total queued documents skipped because their queue entries were stale.",
    ),
    MetricKey.INDEX_QUEUE_DOCS_NEW: MetricDefinition(
        unit="doc",
        description="Total new documents indexed from single-document queue jobs.",
    ),
    MetricKey.INDEX_QUEUE_DOCS_UPDATED: MetricDefinition(
        unit="doc",
        description="Total existing documents updated from single-document queue jobs.",
    ),
    MetricKey.INDEX_QUEUE_FAILED: MetricDefinition(
        unit="job",
        description="Total single-document queue jobs marked as failed.",
    ),
    MetricKey.INDEX_QUEUE_REQUEUED: MetricDefinition(
        unit="job",
        description="Total single-document queue jobs requeued for retry.",
    ),
    MetricKey.INDEX_QUEUE_JOB_MS: MetricDefinition(
        unit="ms",
        description="Total processing latency accumulated across single-document queue jobs.",
    ),
    MetricKey.INDEX_BULK_BATCHES: MetricDefinition(
        unit="batch",
        description="Total bulk indexing jobs processed.",
    ),
    MetricKey.INDEX_BULK_BATCH_SIZE: MetricDefinition(
        unit="doc",
        description="Total documents accumulated across processed bulk indexing jobs.",
    ),
    MetricKey.INDEX_BULK_FAILED: MetricDefinition(
        unit="batch",
        description="Total bulk indexing jobs marked as failed.",
    ),
    MetricKey.INDEX_BULK_REQUEUED: MetricDefinition(
        unit="batch",
        description="Total bulk indexing jobs requeued for retry.",
    ),
    MetricKey.INDEX_BULK_JOB_MS: MetricDefinition(
        unit="ms",
        description="Total processing latency accumulated across bulk indexing jobs.",
    ),
}
