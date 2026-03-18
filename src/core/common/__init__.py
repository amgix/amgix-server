"""
Common utilities and constants for the core module.
"""

from .constants import *
from .enums import *
from .functions import *
from .cache import AMGIXCache
from .embed_router import EmbedRouter

__all__ = [
    # Constants
    "APP_NAME",
    "APP_PREFIX",
    "SEARCH_PREFETCH_MULTIPLIER",
    "IDF_THRESHOLD_MULTIPLIER",
    "MAX_BULK_UPLOAD",
    "MAX_COLLECTION_NAME_LENGTH",
    "MAX_INTERNAL_COLLECTION_NAME_LENGTH",
    "MAX_VECTOR_NAME_LENGTH",
    "MAX_FIELD_VECTOR_NAME_LENGTH",
    "MAX_MODEL_NAME_LENGTH",
    "MAX_DOCUMENT_NAME_LENGTH",
    "MAX_DOCUMENT_DESCRIPTION_LENGTH",
    "MAX_DOCUMENT_CONTENT_LENGTH",
    "MAX_DOCUMENT_ID_LENGTH",
    "MAX_METADATA_KEY_LENGTH",
    "MAX_METADATA_VALUE_LENGTH",
    "MAX_DOCUMENT_TAGS_COUNT",
    "MAX_DOCUMENT_TAG_LENGTH",
    "MAX_SEARCH_QUERY_LENGTH",
    "MAX_SEARCH_LIMIT",
    "MAX_STATUS_LENGTH",
    "UUID_LENGTH",
    "MODEL_CACHE_SIZE",
    "DEFAULT_DB_POOL_SIZE",
    "RPC_TIMEOUT_SECONDS",
    "COLLECTION_INGEST_LOCK_TIMEOUT",
    "MAX_DATABASE_WAIT_SECONDS",
    "MAX_QUEUE_DELIVERY_ATTEMPTS",
    "MAX_DB_RETRIES",
    "MAX_QUEUE_MESSAGES",
    "MAX_QUEUE_SIZE_BYTES",
    "LANGUAGE_DETECTION_CONFIDENCE",
    "DEFAULT_SEARCH_LIMIT",
    "WMTR_WORD_WEIGHT_PERCENTAGE",
    "MAX_TOP_K_VALUE",
    "MAX_VECTOR_DIMENSIONS",
    "DEFAULT_TOP_K",
    "TOKEN_HASH_RANGE",
    "DEFAULT_SQL_BATCH_SIZE",
    "MIN_DB_POOL_SIZE",
    "ENCODER_SERVICE_NAME",
    "CACHE_BASE_DIR",
    "HF_CACHE_DIR",
    "CUDA_CACHE_DIR",
    "DOC_NAMESPACE",
    
    # Enums and Types
    "VectorType",
    "VectorTypeLiteral", 
    "DocumentField",
    "DocumentFieldLiteral",
    "QueuedDocumentStatus",
    "QueuedDocumentStatusLiteral",
    "DenseDistance",
    "DenseDistanceLiteral",
    "MetadataValueType",
    "MetadataValueTypeLiteral",
    "MetadataFilterOpLiteral",
    "DatabaseInfo",
    "DatabaseFeatures",
    
    # Functions
    "get_real_collection_name",
    "get_user_collection_name",
    
    # Classes
    "AMGIXCache",
    "BunnyLock",
    "EmbedRouter",
]
