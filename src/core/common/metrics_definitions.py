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
    API_ASYNC_DELETE = "api_async_delete"
    API_ASYNC_DELETE_MS = "api_async_delete_ms"
    API_SYNC_DELETE = "api_sync_delete"
    API_SYNC_DELETE_MS = "api_sync_delete_ms"
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
        description="HTTP requests handled by the API process.",
    ),
    MetricKey.API_REQUEST_MS: MetricDefinition(
        unit="ms",
        description="Sum of HTTP request durations in milliseconds on the API process.",
    ),
    MetricKey.API_ASYNC_UPLOAD: MetricDefinition(
        unit="req",
        description="Async single-document upload requests handled by the API process.",
    ),
    MetricKey.API_ASYNC_UPLOAD_MS: MetricDefinition(
        unit="ms",
        description="Sum of request durations in milliseconds for async single-document uploads on the API process.",
    ),
    MetricKey.API_SYNC_UPLOAD: MetricDefinition(
        unit="req",
        description="Sync single-document upload requests handled by the API process.",
    ),
    MetricKey.API_SYNC_UPLOAD_MS: MetricDefinition(
        unit="ms",
        description="Sum of request durations in milliseconds for sync single-document uploads on the API process.",
    ),
    MetricKey.API_BULK_UPLOAD: MetricDefinition(
        unit="req",
        description="Bulk upload requests handled by the API process.",
    ),
    MetricKey.API_BULK_UPLOAD_MS: MetricDefinition(
        unit="ms",
        description="Sum of request durations in milliseconds for bulk uploads on the API process.",
    ),
    MetricKey.API_SEARCH: MetricDefinition(
        unit="req",
        description="Search requests handled by the API process.",
    ),
    MetricKey.API_SEARCH_MS: MetricDefinition(
        unit="ms",
        description="Sum of request durations in milliseconds for search on the API process.",
    ),
    MetricKey.API_ASYNC_DELETE: MetricDefinition(
        unit="req",
        description="Async single-document delete requests handled by the API process.",
    ),
    MetricKey.API_ASYNC_DELETE_MS: MetricDefinition(
        unit="ms",
        description="Sum of request durations in milliseconds for async single-document deletes on the API process.",
    ),
    MetricKey.API_SYNC_DELETE: MetricDefinition(
        unit="req",
        description="Sync single-document delete requests handled by the API process.",
    ),
    MetricKey.API_SYNC_DELETE_MS: MetricDefinition(
        unit="ms",
        description="Sum of request durations in milliseconds for sync single-document deletes on the API process.",
    ),
    MetricKey.API_ERROR_4XX: MetricDefinition(
        unit="err",
        description="API responses with a 4xx status code on the API process.",
    ),
    MetricKey.API_ERROR_5XX: MetricDefinition(
        unit="err",
        description="API responses with a 5xx status code on the API process.",
    ),
    MetricKey.EMBED_BATCHES_ORIGIN: MetricDefinition(
        unit="batch",
        description="Embedding batches for requests that entered the pipeline on this encoder node.",
    ),
    MetricKey.EMBED_PASSAGES_ORIGIN: MetricDefinition(
        unit="passage",
        description="Passages in embedding batches for requests that entered the pipeline on this encoder node.",
    ),
    MetricKey.EMBED_INFERENCE_ORIGIN_MS: MetricDefinition(
        unit="ms",
        description="Sum of end-to-end embedding durations in milliseconds for requests that entered the pipeline on this encoder node.",
    ),
    MetricKey.EMBED_INFERENCE_ORIGIN_ERRORS: MetricDefinition(
        unit="err",
        description="Embedding requests that failed after entering the pipeline on this encoder node.",
    ),
    MetricKey.EMBED_BATCHES: MetricDefinition(
        unit="batch",
        description="Embedding batches executed on this encoder node.",
    ),
    MetricKey.EMBED_PASSAGES: MetricDefinition(
        unit="passage",
        description="Passages embedded on this encoder node.",
    ),
    MetricKey.EMBED_INFERENCE_MS: MetricDefinition(
        unit="ms",
        description="Sum of model inference durations in milliseconds on this encoder node.",
    ),
    MetricKey.EMBED_HOPS: MetricDefinition(
        unit="hop",
        description="Sum of routing hop counts for embedding work handled on this encoder node.",
    ),
    MetricKey.INDEX_QUEUE_DOCS_SKIPPED_STALE: MetricDefinition(
        unit="doc",
        description="Single-document queue documents skipped because the queue entry was stale.",
    ),
    MetricKey.INDEX_QUEUE_DOCS_NEW: MetricDefinition(
        unit="doc",
        description="New documents indexed from the single-document queue.",
    ),
    MetricKey.INDEX_QUEUE_DOCS_UPDATED: MetricDefinition(
        unit="doc",
        description="Existing documents updated from the single-document queue.",
    ),
    MetricKey.INDEX_QUEUE_FAILED: MetricDefinition(
        unit="job",
        description="Single-document queue jobs marked failed.",
    ),
    MetricKey.INDEX_QUEUE_REQUEUED: MetricDefinition(
        unit="job",
        description="Single-document queue jobs requeued for retry.",
    ),
    MetricKey.INDEX_QUEUE_JOB_MS: MetricDefinition(
        unit="ms",
        description="Sum of job durations in milliseconds for single-document queue jobs.",
    ),
    MetricKey.INDEX_BULK_BATCHES: MetricDefinition(
        unit="batch",
        description="Bulk indexing batches processed.",
    ),
    MetricKey.INDEX_BULK_BATCH_SIZE: MetricDefinition(
        unit="doc",
        description="Sum of document counts across bulk indexing batches (numerator for average batch size).",
    ),
    MetricKey.INDEX_BULK_FAILED: MetricDefinition(
        unit="batch",
        description="Bulk indexing batches marked failed.",
    ),
    MetricKey.INDEX_BULK_REQUEUED: MetricDefinition(
        unit="batch",
        description="Bulk indexing batches requeued for retry.",
    ),
    MetricKey.INDEX_BULK_JOB_MS: MetricDefinition(
        unit="ms",
        description="Sum of batch durations in milliseconds for bulk indexing.",
    ),
}
