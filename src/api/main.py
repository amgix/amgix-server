from __future__ import annotations

import os
import logging
import asyncio
import pathlib
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, HTTPException, Path, Query, Request, APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from typing import Annotated
import traceback


from src.api import api_metrics as api_metrics_module
from src.api.api_metrics import ApiMetricsMiddleware, init_api_metrics_service
from src.api.prometheus_metrics import metrics_to_prometheus_text
from src.core.common.bunny_talk import BunnyTalk
from src.core.common.lock_manager import LockService, LockClient
from src.core.common.logging_config import configure_logging
from src.core.models.cluster import Metrics, MetricTrend, MetricsBucket, MetricsPayload
from src.core.common.metrics_definitions import METRIC_DEFINITIONS, MetricKey

_VALID_METRIC_KEY_STR = {k.value for k in MetricKey}
from src.core.models.document import (
    CollectionStatsResponse,
    Document,
    DocumentStatusResponse,
    QueueDocument,
    QueueInfo,
    SearchResult,
)
from src.core.models.vector import CollectionConfig, CollectionConfigInternal, VectorConfigInternal, SearchQuery, SearchQueryWithVectors, ModelValidationResponse
from src.core.models.vector import VectorData
from src.core.common import (
    VectorType, get_real_collection_name, get_user_collection_name,
    APP_NAME, MAX_BULK_UPLOAD, MAX_COLLECTION_NAME_LENGTH,
    RPC_TIMEOUT_SECONDS,APP_PREFIX
)
from src.core.database.common import (
    get_connected_database,
    validate_metadata_types,
    AmgixValidationError,
)
from src.core.database.base import AmgixNotFound
from pydantic import Field, BaseModel
from starlette.concurrency import run_in_threadpool

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


# Collection name validation type
CollectionName = Annotated[str, Path(..., 
    regex=r"^[a-zA-Z0-9_-]+$",
    min_length=1,
    max_length=MAX_COLLECTION_NAME_LENGTH,
    description="Collection name (alphanumeric, underscores, hyphens only)"
)]

class OkResponse(BaseModel):
    ok: bool

# 218 = partial readiness (some index/query probes not ready); 200 = fully ready
HTTP_STATUS_PARTIAL_READY = 218

READINESS_PING_TIMEOUT_SEC = 2.0


async def _readiness_ping(routing_key: str) -> bool:
    if _bunny_talk is None:
        return False
    try:
        await _bunny_talk.rpc(routing_key, timeout=READINESS_PING_TIMEOUT_SEC)
        return True
    except Exception:
        return False


class ReadyResponse(BaseModel):
    database: bool
    rabbitmq: bool
    index: bool
    query: bool
    ready: bool

class VersionResponse(BaseModel):
    version: str


class MetricDefinitionItem(BaseModel):
    key: str = Field(..., description="Stable metric identifier")
    unit: str = Field(..., description="Unit label (e.g. req, ms, batch)")
    description: str = Field(..., description="Description of this metric")


def _database_kind_label(connection_string: str) -> str:
    """Short product label from the DB URL scheme (no host or credentials exposed)."""
    scheme = connection_string.split("://", 1)[0].lower() if "://" in connection_string else ""
    known = {
        "qdrant": "Qdrant",
        "mariadb": "MariaDB",
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
    }
    if scheme in known:
        return known[scheme]
    return scheme.capitalize() if scheme else "Database"


class SystemInfoResponse(BaseModel):
    amgix_version: str = Field(..., description="API / deployment version string")
    database_kind: str = Field(
        ...,
        description="Database product derived from configured URL scheme (no connection string)",
    )
    database_version: str = Field(..., description="Version reported by the database backend after probe")
    database_features: Dict[str, bool] = Field(
        ...,
        description="Feature flags detected at probe time (e.g. dense vector support)",
    )
    rabbitmq_version: str = Field(
        ...,
        description="AMQP broker version from Connection.Start server_properties (e.g. RabbitMQ)",
    )
    collection_count: int = Field(..., description="Number of user collections", ge=0)


class CollectionExistsResponse(BaseModel):
    exists: bool


class BulkUploadRequest(BaseModel):
    documents: List[Document] = Field(..., max_length=MAX_BULK_UPLOAD)


AMGIX_VERSION = os.getenv("AMGIX_VERSION", "1.0.0-dev")
HOSTNAME = os.getenv('HOSTNAME', 'unknown')
AMGIX_DATABASE_URL = os.getenv("AMGIX_DATABASE_URL", "qdrant://localhost:6334")
AMGIX_AMQP_URL = os.getenv("AMGIX_AMQP_URL", f"pyamqp://guest:guest@rabbitmq//")
RPC_EXCHANGE = f"{APP_PREFIX}-rpc"

if "?" not in AMGIX_AMQP_URL:
    AMGIX_AMQP_URL = f"{AMGIX_AMQP_URL}?name={APP_PREFIX}-api-{HOSTNAME}"

# Lazy initialization of database
_database = None

# Lazy initialization of BunnyTalk
_bunny_talk = None
# Filled at startup from BunnyTalk.get_broker_version() (AMQP server_properties).
_rabbitmq_broker_version: str | None = None

DASHBOARD_STATIC_DIR = pathlib.Path(__file__).resolve().parent / "dashboard"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for database connection on startup."""
    global _database, _bunny_talk, _rabbitmq_broker_version

    logger.info(f"Starting {APP_NAME} API v{AMGIX_VERSION}")

    # Initialize BunnyTalk for RPC calls
    _bunny_talk = await BunnyTalk.create(logger, AMGIX_AMQP_URL)
    _rabbitmq_broker_version = _bunny_talk.get_broker_version()
    logger.info(f"RabbitMQ broker version: {_rabbitmq_broker_version or 'unknown'}")

    # Create and start lock service
    logger.info("Starting lock service...")
    lock_service = LockService(logger, _bunny_talk)
    await lock_service.startup()
    logger.info("Starting lock service... done")
    
    # Create lock client
    lock_client = LockClient(logger, _bunny_talk)

    _database = await get_connected_database(AMGIX_DATABASE_URL, logger=logger)

    async with lock_client.acquire("database-configure", timeout=30.0):
        await _database.configure()

    await _database.check_features()

    api_metrics = init_api_metrics_service(
        amqp_url=AMGIX_AMQP_URL,
        logger=logger,
        hostname=HOSTNAME,
        database=_database,
    )
    api_metrics.start_reporting()

    yield

    logger.info(f"Stopping {APP_NAME} API")

    await api_metrics.stop_reporting()

    # Cleanup on shutdown
    if _bunny_talk:
        await _bunny_talk.close()

app = FastAPI(
    title=f"{APP_NAME} API", 
    version=AMGIX_VERSION, 
    description=f"{APP_NAME} (Amgix). Open-Source Hybrid Search System",
    lifespan=lifespan)



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all exceptions with full details."""
    # Log the full exception details
    logger.error(f"Unhandled exception in {request.method} {request.url}")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Full traceback:")
    logger.error(traceback.format_exc())
    _bunny_talk.log_trace_context(f"Error context for {request.method} {request.url}")

    # Determine status code and detail based on exception type
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        detail = f"HTTP error: {exc.detail}"
    elif isinstance(exc, AmgixNotFound):
        status_code = 404
        detail = str(exc)
    elif isinstance(exc, AmgixValidationError):
        status_code = 400
        detail = str(exc)
    else:
        status_code = 500
        detail = f"Internal server error: {str(exc)}"
    
    # Return a generic error response (don't expose internal details to client)
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail}
    )


# -------------------------
# Shared endpoints (v1 = default)
# -------------------------
shared_router = APIRouter()

@shared_router.get("/version", operation_id="version")
async def version() -> VersionResponse:
    """Return the system version.

    Returns:
        A `VersionResponse` object with the system version.
    """
    return VersionResponse(version=AMGIX_VERSION)


@shared_router.get("/system/info", operation_id="system_info")
async def system_info() -> SystemInfoResponse:
    """Summarize deployment and infrastructure (no connection URLs)."""
    cached = _database.get_cached_database_info()
    db_version = cached.version if cached else "unknown"
    db_features = dict(cached.features) if cached else {}
    real_names = await _database.list_collections()
    collection_count = len(real_names)
    rmq_version = _rabbitmq_broker_version or "unknown"
    return SystemInfoResponse(
        amgix_version=AMGIX_VERSION,
        database_kind=_database_kind_label(_database.connection_string),
        database_version=db_version,
        database_features=db_features,
        rabbitmq_version=rmq_version,
        collection_count=collection_count,
    )


@shared_router.get("/health/check", operation_id="health_check")
async def health() -> OkResponse:
    """Check API service responsiveness.

    This endpoint returns a simple 'ok' status to indicate that the API service
    is running and able to respond to requests.

    Returns:
        An `OkResponse` object with the 'ok' field set to True, confirming the service's responsiveness.
    """
    return OkResponse(ok=True)


@shared_router.get(
    "/health/ready",
    operation_id="health_ready",
    responses={
        200: {"description": "Fully ready", "model": ReadyResponse},
        218: {"description": "Partial ready (some index/query probes not ready)", "model": ReadyResponse},
        503: {"description": "Service Unavailable (infra down or all encoder roles down)", "model": ReadyResponse},
    },
)
async def readiness_check() -> ReadyResponse:
    """Check if service is ready to handle requests.

    Runs four probes: database, rabbitmq, index workers, query workers.
    Returns 200 if all pass (fully ready), 218 if some fail (partial ready).
    Response body always includes all four probe results and a ready flag.
    """
    db_healthy = False
    try:
        db_healthy = await _database.is_connected()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    rabbitmq_healthy = False
    try:
        rabbitmq_healthy = (
            not _bunny_talk.connection.is_closed
            and not _bunny_talk.publish_channel.is_closed
        )
    except Exception as e:
        logger.error(f"RabbitMQ health check failed: {e}")

    index_healthy = False
    query_healthy = False
    if rabbitmq_healthy:
        index_healthy, query_healthy = await asyncio.gather(
            _readiness_ping("ping-encoder"),
            _readiness_ping("ping-rpc"),
        )

    ready = db_healthy and rabbitmq_healthy and index_healthy and query_healthy
    body = ReadyResponse(
        database=db_healthy,
        rabbitmq=rabbitmq_healthy,
        index=index_healthy,
        query=query_healthy,
        ready=ready,
    )
    # 503 if infra down (db/rabbit) or no encoder role at all; 218 if one encoder role up (partial); 200 if all up
    if not db_healthy or not rabbitmq_healthy or (not index_healthy and not query_healthy):
        return JSONResponse(status_code=503, content=body.model_dump())
    if ready:
        return JSONResponse(status_code=200, content=body.model_dump())
    return JSONResponse(status_code=HTTP_STATUS_PARTIAL_READY, content=body.model_dump())


@shared_router.get("/metrics/current", operation_id="metrics_current")
async def metrics_current(
    window: int = Query(
        default=60,
        description="Aggregation window in seconds — 30 or 60.",
    ),
    keys: Optional[List[str]] = Query(
        default=None,
        description="Restrict returned metric series to these keys. Omit for all keys.",
    ),
) -> Metrics:
    """Return the current metrics state for all nodes over the given window (seconds)."""
    if window not in (30, 60):
        raise HTTPException(status_code=422, detail="window must be 30 or 60")
    if keys:
        unknown = [k for k in keys if k not in _VALID_METRIC_KEY_STR]
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown metric keys: {unknown}")
    return await _bunny_talk.rpc(
        api_metrics_module.api_metrics.leader_queue,
        payload=MetricsPayload(
            probe=False,
            query_view=True,
            query_window=window,
            query_keys=keys,
            hostname=HOSTNAME,
        ),
        return_type=Metrics,
        timeout=2.0,
    )


@shared_router.get(
    "/metrics/prometheus",
    operation_id="metrics_prometheus",
    response_class=PlainTextResponse,
)
async def metrics_prometheus() -> PlainTextResponse:
    """Expose current cluster metrics in Prometheus text exposition (60s rolling window)."""
    metrics = await _bunny_talk.rpc(
        api_metrics_module.api_metrics.leader_queue,
        payload=MetricsPayload(
            probe=False,
            query_view=True,
            query_window=60,
            query_keys=None,
            hostname=HOSTNAME,
        ),
        return_type=Metrics,
        timeout=2.0,
    )
    body = metrics_to_prometheus_text(metrics, window_seconds=60)
    return PlainTextResponse(
        content=body,
        media_type="text/plain; charset=utf-8",
    )


@shared_router.get("/metrics/trends", operation_id="metrics_trends")
async def metrics_trends(
    since: datetime,
    until: datetime,
    resolution: int = Query(
        default=60,
        description="Bucket size in seconds — 60 for 1-minute, 300 for 5-minute.",
    ),
    keys: Optional[List[str]] = Query(default=None),
) -> List[MetricTrend]:
    """Return historical metric buckets for the given time range and resolution.

    Args:
        since: Inclusive start of the time range (ISO 8601, UTC assumed if no timezone given).
        until: Exclusive end of the time range (ISO 8601, UTC assumed if no timezone given).
        resolution: Bucket size in seconds — 60 for 1-minute, 300 for 5-minute.
        keys: One or more metric keys to return. Omit to return all keys.
    """
    if resolution not in (60, 300):
        raise HTTPException(status_code=422, detail="resolution must be 60 or 300")
    if since.tzinfo is None or until.tzinfo is None:
        raise HTTPException(status_code=422, detail="'since' and 'until' must include a timezone (e.g. 2024-01-01T00:00:00Z)")

    since_ts = since.timestamp()
    until_ts = until.timestamp()

    if since_ts >= until_ts:
        raise HTTPException(status_code=422, detail="'since' must be less than 'until'")
    valid_keys = {k.value for k in MetricKey}
    if keys:
        unknown = [k for k in keys if k not in valid_keys]
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown metric keys: {unknown}")

    buckets = await _database.query_metric_buckets(
        bucket_seconds=resolution,
        since=since_ts,
        until=until_ts,
        keys=keys or None,
    )

    grouped: Dict[str, List[MetricsBucket]] = {}
    for bucket in buckets:
        grouped.setdefault(bucket.key, []).append(bucket)

    return [
        MetricTrend(key=key, bucket_seconds=resolution, buckets=key_buckets)
        for key, key_buckets in grouped.items()
    ]


@shared_router.get("/metrics/definitions", operation_id="metrics_definitions")
async def metrics_definitions() -> List[MetricDefinitionItem]:
    """Return catalog entries for all known metric keys, their units, and descriptions."""
    return [
        MetricDefinitionItem(key=key.value, unit=defn.unit, description=defn.description)
        for key, defn in METRIC_DEFINITIONS.items()
    ]


# Collections
@shared_router.post("/collections/{collection_name}", operation_id="create_collection")
async def create_collection(collection_name: CollectionName, config: CollectionConfig = ...) -> OkResponse:
    """Create a new collection.

    This endpoint creates a new collection with the specified name and vector configurations.
    It validates the provided model configurations and ensures all required features are supported by the database.

    Args:
        collection_name: The unique name for the new collection (alphanumeric, underscores, hyphens only).
        config: Configuration details for the collection, including vector types and storage options.

    Returns:
        An `OkResponse` object indicating the success of the operation.

    Raises:
        HTTPException:
            - 400 if model validation fails or required features are not supported.
            - 409 if a collection with the same name already exists.
            - 500 if the collection creation fails in the database.
    """
    # Check if database supports all required features
    await _database.validate_features(config)
    
    # Convert user collection name to real collection name
    real_collection_name = get_real_collection_name(collection_name)
    
    # Check if collection already exists
    try:
        await _database.get_collection_info_internal(real_collection_name)
        raise HTTPException(status_code=409, detail=f"Collection '{collection_name}' already exists")
    except AmgixNotFound:
        # Collection doesn't exist, proceed with creation
        pass
    
    # Validate models and get dimensions for dense vectors
    validation_response = await _bunny_talk.rpc(
        "validate-models",
        vector_configs=config.vectors,
        start_trace=True,
        trace_meta={"collection": real_collection_name}
    )
    
    # Check for validation errors
    if validation_response.error:
        raise HTTPException(status_code=400, detail=f"Model validation failed: {validation_response.error}")
    
    # Convert request models to full models with discovered dimensions
    full_vector_configs = []
    for vector_request in config.vectors:
        if vector_request.type == VectorType.DENSE_MODEL:
            if not validation_response.results or vector_request.name not in validation_response.results:
                raise HTTPException(status_code=400, detail=f"Model validation failed for {vector_request.name}")
            
            validation_result = validation_response.results[vector_request.name]
            if not validation_result.valid:
                error_msg = validation_result.error or "Model validation failed"
                raise HTTPException(status_code=400, detail=f"Model {vector_request.name} is not valid: {error_msg}")
            
            # Create full VectorConfigInternal with discovered dimensions
            vector_data = vector_request.model_dump(exclude={'dimensions'})
            full_config = VectorConfigInternal(
                **vector_data,
                dimensions=validation_result.dimension
            )
        else:
            # For non-dense models, just convert directly
            full_config = VectorConfigInternal(**vector_request.model_dump())
        
        full_vector_configs.append(full_config)
    
    # Create full collection config for database storage
    full_collection_config = CollectionConfigInternal(
        collection_id=str(uuid.uuid4()),
        vectors=full_vector_configs,
        store_content=config.store_content,
        metadata_indexes=config.metadata_indexes
    )
    
    # Create collection with complete config (including dimensions)
    ok = await _database.create_collection(real_collection_name, full_collection_config)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to create collection '{collection_name}'")
    return OkResponse(ok=ok)


@shared_router.get("/collections", operation_id="list_collections")
async def list_collections() -> List[str]:
    """List all available collections.

    Retrieves a list of all collections managed by the application.

    Returns:
        A list of strings, where each string is the name of an available collection.
    """
    # Always get collections with our app prefix
    real_collections = await _database.list_collections()
    # Convert back to user-facing names
    return [get_user_collection_name(name) for name in real_collections]


@shared_router.delete("/collections/{collection_name}", operation_id="delete_collection")
async def delete_collection(collection_name: CollectionName) -> OkResponse:
    """Delete a collection.

    Deletes a collection and all its associated data. This operation is irreversible.

    Args:
        collection_name: The name of the collection to delete.

    Returns:
        An `OkResponse` object indicating the success of the operation.
    """
    real_collection_name = get_real_collection_name(collection_name)
    ok = await _database.delete_collection(real_collection_name)
    return OkResponse(ok=ok)


@shared_router.get("/collections/{collection_name}", operation_id="get_collection_config")
async def get_collection_config(collection_name: CollectionName) -> CollectionConfig:
    """Get collection configuration.

    Retrieves the configuration details for a specific collection.

    Args:
        collection_name: The name of the collection.

    Returns:
        The configuration of the specified collection.
    """
    real_collection_name = get_real_collection_name(collection_name)
    collection_config = await _database.get_collection_info(real_collection_name)
    return collection_config


@shared_router.get("/collections/{collection_name}/stats", operation_id="get_collection_stats")
async def get_collection_stats(collection_name: CollectionName) -> CollectionStatsResponse:
    """Get persisted collection statistics and queue counts.

    Returns document counts maintained by the indexing pipeline (not a live physical count),
    plus queue entry counts by state (same data as ``GET .../queue/info``).

    Args:
        collection_name: The name of the collection.

    Returns:
        A `CollectionStatsResponse` with `doc_count` and `queue`.

    Raises:
        HTTPException: 404 if the collection does not exist.
    """
    real_collection_name = get_real_collection_name(collection_name)

    # Calling this just to throw 404 if the collection doesn't exist
    await _database.get_collection_info_internal(real_collection_name)
    
    stats, queue_info = await asyncio.gather(
        _database.get_collection_stats(real_collection_name),
        _database.get_queue_info(real_collection_name),
    )
    doc_count = stats.get("doc_count", 0)
    if not isinstance(doc_count, int):
        doc_count = int(doc_count)
    return CollectionStatsResponse(doc_count=doc_count, queue=queue_info)


@shared_router.get("/collections/{collection_name}/exists", operation_id="collection_exists")
async def collection_exists(collection_name: CollectionName) -> CollectionExistsResponse:
    """Check if a collection exists. Always returns 200 with exists true or false."""
    real_collection_name = get_real_collection_name(collection_name)
    try:
        await _database.get_collection_info_internal(real_collection_name)
        return CollectionExistsResponse(exists=True)
    except AmgixNotFound:
        return CollectionExistsResponse(exists=False)


@shared_router.post("/collections/{collection_name}/empty", operation_id="empty_collection")
async def empty_collection(collection_name: CollectionName) -> OkResponse:
    """Empty a collection.

    Removes all documents and their associated vector data from a specified collection,
    but keeps the collection's configuration.

    Args:
        collection_name: The name of the collection to empty.

    Returns:
        An `OkResponse` object indicating the success of the operation.
    """
    real_collection_name = get_real_collection_name(collection_name)
    ok = await _database.empty_collection(real_collection_name)
    return OkResponse(ok=ok)


async def upload_documents_to_queue(collection_name: str, collection_id: str, documents: List[Document]):
    """Upload documents to the processing queue and publish events for encoder."""
    # Add documents to the processing queue
    queue_ids = await _database.add_to_queue(collection_name, collection_id, documents)
    
    try:
        if len(documents) > 1:
            # Publish one message with list of queue_ids for bulk processing
            await _bunny_talk.talk(
                "documents-bulk",
                queue_ids=queue_ids,
                start_trace=True,
                trace_meta={
                    "collection": collection_name,
                    "collection_id": collection_id,
                    "queue_ids": queue_ids,
                },
            )
        else:
            logger.info(f"Uploading single document {documents[0].id} to queue {queue_ids[0]}")
            # Single document - pass single queue_id
            await _bunny_talk.talk(
                "documents",
                queue_id=queue_ids[0],
                start_trace=True,
                trace_meta={
                    "collection": collection_name,
                    "collection_id": collection_id,
                    "document_id": documents[0].id,
                    "queue_id": queue_ids[0],
                },
            )
    except Exception as e:
        # If publishing fails, delete all queue entries and raise exception
        try:
            await _database.delete_from_queue(queue_ids)
        except Exception as e2:
            logger.error(f"Failed to delete records from queue: {str(e2)}")
        raise HTTPException(status_code=500, detail=f"Failed to publish event to internal queue: {str(e)}")



# Documents
@shared_router.post("/collections/{collection_name}/documents", operation_id="upsert_document")
async def upsert_document(collection_name: CollectionName, document: Document = ...) -> OkResponse:
    """Upsert a single document asynchronously.

    Adds or updates a single document in the specified collection by placing it into a processing queue.
    The document will be vectorized and indexed asynchronously.

    Args:
        collection_name: The name of the collection to upsert the document into.
        document: The document object to be upserted.

    Returns:
        An `OkResponse` object indicating that the document has been accepted for processing.

    Raises:
        HTTPException: 500 if publishing the event to the internal queue fails.
    """
    real_collection_name = get_real_collection_name(collection_name)
    collection_config = await _database.get_collection_info_internal(real_collection_name)
    validate_metadata_types(collection_config, document)
    await upload_documents_to_queue(real_collection_name, collection_config.collection_id, [document])
    return OkResponse(ok=True)

@shared_router.post("/collections/{collection_name}/documents/sync", operation_id="upsert_document_sync")
async def upsert_document_sync(collection_name: CollectionName, document: Document = ...) -> OkResponse:
    """Upsert a single document synchronously.

    Adds or updates a single document in the specified collection and waits for the operation
    to complete, including vectorization and indexing.

    Args:
        collection_name: The name of the collection to upsert the document into.
        document: The document object to be upserted.

    Returns:
        An `OkResponse` object indicating the success of the operation.

    Raises:
        HTTPException:
            - 409 if a document with the same ID and newer timestamp already exists (conflict).
            - 500 for other internal server errors during processing.
    """
    real_collection_name = get_real_collection_name(collection_name)
    collection_config = await _database.get_collection_info_internal(real_collection_name)
    validate_metadata_types(collection_config, document)
    try:
        await _bunny_talk.rpc(
            "document-sync",
            collection_name=real_collection_name,
            start_trace=True,
            document=document,
            trace_meta={"collection": real_collection_name, "document_id": document.id}
        )
    except Exception as e:
        msg = str(e)
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise
    return OkResponse(ok=True)

@shared_router.post("/collections/{collection_name}/documents/bulk", operation_id="upsert_documents_bulk")
async def upsert_documents_bulk(
    collection_name: CollectionName, 
    request: BulkUploadRequest
) -> OkResponse:
    """Upsert multiple documents in bulk asynchronously.

    Adds or updates multiple documents in the specified collection by placing them into a processing queue.
    Documents will be vectorized and indexed asynchronously. This method is optimized for bulk operations.

    Args:
        collection_name: The name of the collection to upsert the documents into.
        request: A `BulkUploadRequest` object containing a list of `Document` objects to be upserted.

    Returns:
        An `OkResponse` object indicating that the documents have been accepted for processing.

    Raises:
        HTTPException: 500 if publishing events to the internal queue fails for any document.
    """
    real_collection_name = get_real_collection_name(collection_name)
    collection_config = await _database.get_collection_info_internal(real_collection_name)
    for document in request.documents:
        validate_metadata_types(collection_config, document)
    await upload_documents_to_queue(real_collection_name, collection_config.collection_id, request.documents)
    return OkResponse(ok=True)


@shared_router.get("/collections/{collection_name}/documents/{document_id}", operation_id="get_document")
async def get_document(collection_name: CollectionName, document_id: str = Path(...)) -> Document:
    """Retrieve a single document.

    Retrieves a specific document by its ID from the specified collection.

    Args:
        collection_name: The name of the collection.
        document_id: The unique identifier of the document to retrieve.

    Returns:
        The retrieved `Document` object.

    Raises:
        HTTPException: 404 if the document is not found in the collection.
    """
    real_collection_name = get_real_collection_name(collection_name)
    doc_with_vectors = (await _database.get_documents(real_collection_name, [document_id]))[0]
    return Document(**doc_with_vectors.model_dump(exclude={'vectors', 'token_lengths'}))


@shared_router.delete("/collections/{collection_name}/documents/{document_id}", operation_id="delete_document")
async def delete_document(collection_name: CollectionName, document_id: str = Path(...)) -> OkResponse:
    """Delete a document.

    Deletes a specific document by its ID from the specified collection.

    Args:
        collection_name: The name of the collection.
        document_id: The unique identifier of the document to delete.

    Returns:
        An `OkResponse` object indicating the success of the operation.
    """
    real_collection_name = get_real_collection_name(collection_name)
    
    # Get document before deleting to retrieve token_lengths for stats update
    doc_with_vectors = (await _database.get_documents(real_collection_name, [document_id], suppress_not_found=True))[0]
    
    ok = await _database.delete_document(real_collection_name, document_id)
    
    # Send stats update with negative values
    updates = {}
    for field_vector_name, token_length in doc_with_vectors.token_lengths.items():
        updates[field_vector_name] = {
            "new_doc_count": -1,
            "new_sum_token_lengths": -token_length,
            "update_doc_count": 0,
            "update_sum_token_lengths": 0,
            "old_sum_token_lengths": 0
        }
    
    if updates:
        await _bunny_talk.talk("collection-stats", collection_name=real_collection_name, updates=updates)
    
    return OkResponse(ok=ok)


@shared_router.get("/collections/{collection_name}/documents/{document_id}/status", operation_id="get_document_status")
async def get_document_status(collection_name: CollectionName, document_id: str = Path(...)) -> DocumentStatusResponse:
    """Get document processing status.

    Retrieves the processing status of a document, including its current state in the queue
    and any associated messages.

    Args:
        collection_name: The name of the collection.
        document_id: The unique identifier of the document.

    Returns:
        A `DocumentStatusResponse` object containing the processing status of the document.

    Raises:
        HTTPException: 404 if the document is not found in the collection's queue.
    """
    real_collection_name = get_real_collection_name(collection_name)
    status_response = await _database.get_queue_statuses(real_collection_name, document_id)
    
    # Check if we have any statuses - if not, return 404
    if not status_response.statuses:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found in collection {collection_name}")
    
    return status_response


@shared_router.get("/collections/{collection_name}/queue/info", operation_id="get_collection_queue_info")
async def get_collection_queue_info(collection_name: CollectionName) -> QueueInfo:
    """Get queue statistics for a collection.

    Retrieves counts of documents in different queue states (queued, requeued, failed).

    Args:
        collection_name: The name of the collection.

    Returns:
        A `QueueInfo` object with counts for each queue state.
    """
    real_collection_name = get_real_collection_name(collection_name)
    queue_info = await _database.get_queue_info(real_collection_name)
    return queue_info


@shared_router.delete("/collections/{collection_name}/queue", operation_id="delete_collection_queue")
async def delete_collection_queue(collection_name: CollectionName) -> OkResponse:
    """Delete all queue entries for a collection.

    Removes all documents from the processing queue for a specified collection.
    This does not affect documents already indexed in the collection.

    Args:
        collection_name: The name of the collection for which to delete queue entries.

    Returns:
        An `OkResponse` object indicating the success of the operation.
    """
    real_collection_name = get_real_collection_name(collection_name)
    await _database.delete_from_queue_by_collection(real_collection_name)
    return OkResponse(ok=True)


# Search
@shared_router.post("/collections/{collection_name}/search", operation_id="search")
async def search(collection_name: CollectionName, query: SearchQuery = ...) -> List[SearchResult]:
    """Perform a search query on a collection.

    Executes a search query against the specified collection.

    Args:
        collection_name: The name of the collection to search.
        query: The `SearchQuery` object containing the search text, filters, and other parameters.

    Returns:
        A list of `SearchResult` objects, where each object represents a search result.
    """
    real_collection_name = get_real_collection_name(collection_name)
    # Delegate to encoder via RPC
    results = await _bunny_talk.rpc(
        "search",
        collection_name=real_collection_name,
        start_trace=True,
        query=query,
        trace_meta={"collection": real_collection_name},
        return_type=List[SearchResult]
    )
    return results


# -------------------------
# Version 1 API (explicit)
# -------------------------
v1_api = APIRouter(prefix="/v1")
v1_api.include_router(shared_router)

# -------------------------
# Default API (root, no prefix)
# -------------------------
# root_api = APIRouter(prefix="")
# root_api.include_router(shared_router)

# -------------------------
# Mount routers
# -------------------------
# app.include_router(root_api, tags=["Amgix"])  # /collections, /collections/{name}/documents, etc.
app.include_router(v1_api, tags=["Amgix"])    # /v1/collections, /v1/collections/{name}/documents, etc.

app.add_middleware(ApiMetricsMiddleware)

app.mount(
    "/dashboard",
    StaticFiles(directory=str(DASHBOARD_STATIC_DIR), html=True),
    name="dashboard",
)

