"""
Simple constants used throughout the application.
"""
import os
import uuid

# Application name
APP_NAME = "Amalgam Index"
APP_PREFIX = "amgix"

# Search configuration
SEARCH_PREFETCH_MULTIPLIER = 1.5  # Multiply search limit by this factor for prefetching
IDF_THRESHOLD_MULTIPLIER = 3  # Multiply search limit by this factor for IDF filtering threshold

# Bulk upload limits
MAX_BULK_UPLOAD = 100  # Maximum number of documents that can be uploaded in a single bulk request

# Length limits
MAX_COLLECTION_NAME_LENGTH = 100  # Maximum length for collection names
MAX_INTERNAL_COLLECTION_NAME_LENGTH = len(APP_PREFIX) + 1 + 36 + 1 + MAX_COLLECTION_NAME_LENGTH  # APP_PREFIX + "_" + extra_prefix + "_" + user_collection_name
MAX_VECTOR_NAME_LENGTH = 100  # Maximum length for vector names
MAX_FIELD_VECTOR_NAME_LENGTH = MAX_VECTOR_NAME_LENGTH + 1 + 11  # vector_name + "_" + "description" (longest field name)
MAX_MODEL_NAME_LENGTH = 210  # Maximum length for model names and revisions
MAX_DOCUMENT_NAME_LENGTH = 1500  # Maximum length for document names
MAX_DOCUMENT_DESCRIPTION_LENGTH = 3000  # Maximum length for document descriptions
MAX_DOCUMENT_CONTENT_LENGTH = 1000000  # Maximum length for document content
MAX_DOCUMENT_ID_LENGTH = 100  # Maximum length for document IDs
MAX_METADATA_KEY_LENGTH = 100  # Maximum length for metadata keys
MAX_METADATA_VALUE_LENGTH = 1024  # Maximum length for metadata values
MAX_DOCUMENT_TAGS_COUNT = 50  # Maximum number of document tags
MAX_DOCUMENT_TAG_LENGTH = 100  # Maximum length for individual document tags
MAX_SEARCH_QUERY_LENGTH = 10000  # Maximum length for search queries
MAX_SEARCH_LIMIT = 100  # Maximum limit for search results returned
MAX_STATUS_LENGTH = 8  # Length of longest status value ("requeued")
UUID_LENGTH = 36  # Length for UUID strings

# Cache sizes
MODEL_CACHE_SIZE = int(os.getenv("AMGIX_MODEL_CACHE_SIZE", "100"))  # Maximum number of models to cache
MODEL_CACHE_TTL = 1 * 60 * 60 # 1 hour
DEFAULT_DB_POOL_SIZE = 10  # Default database connection pool size

# Timeouts and delays
RPC_TIMEOUT_SECONDS = 60  # Timeout for RPC calls
COLLECTION_INGEST_LOCK_TIMEOUT = 10  # Max wait to acquire collection-level ingest lock (SQL backends)
MAX_DATABASE_WAIT_SECONDS = 30  # Maximum wait time for database connections
MAX_QUEUE_DELIVERY_ATTEMPTS = 4  # Maximum delivery attempts for queue messages (vectorization errors)
MAX_DB_RETRIES = 200  # Maximum retries for database errors before giving up
MAX_QUEUE_MESSAGES = 500000  # Maximum messages in queue
MAX_QUEUE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # Maximum queue size (1GB)

# Vectorization batching defaults (tune based on hardware)
DENSE_MODEL_BATCH_SIZE = 32
SPARSE_MODEL_BATCH_SIZE = 8

# Validation thresholds
LANGUAGE_DETECTION_CONFIDENCE = 0.9  # Minimum confidence for language detection
DEFAULT_SEARCH_LIMIT = 10  # Default limit for search results
WMTR_WORD_WEIGHT_PERCENTAGE = 80  # Percentage of top_k allocated to word weights

# Numeric limits
MAX_TOP_K_VALUE = 10000  # Maximum value for top_k parameter
MAX_VECTOR_DIMENSIONS = 8192  # Maximum vector dimensions
DEFAULT_TOP_K = 128  # Default top_k for sparse vectors
TOKEN_HASH_RANGE = 4294967291  # Large prime number close to 2^32 to reduce hash collisions

# Batch processing
DEFAULT_SQL_BATCH_SIZE = 100  # Default batch size for SQL operations
MIN_DB_POOL_SIZE = 1  # Minimum database connection pool size

# Vectorization concurrency
# Maximum threads to use for parallel sparse vector generation
MAX_SPARSE_VECTOR_THREADS = 4

# Service names
ENCODER_SERVICE_NAME = f"{APP_PREFIX}-encoder-service"
RPC_SERVICE_NAME = f"{APP_PREFIX}-rpc-service"

# Cache directory configuration
CACHE_BASE_DIR = "/data/amgix/cache"
HF_CACHE_DIR = f"{CACHE_BASE_DIR}/huggingface"
CUDA_CACHE_DIR = f"{CACHE_BASE_DIR}/cuda"

# Document ID namespace for UUID5 generation
DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, APP_NAME)
