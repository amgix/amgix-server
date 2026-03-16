"""
Enum-like classes and type literals used throughout the application.
"""
from typing import Literal

# Vector types
class VectorType:
    # Dense vector type (embedding-based)
    DENSE_MODEL = "dense_model"
    
    # Sparse vector types
    SPARSE_MODEL = "sparse_model"      # Transformer-based sparse vectors (e.g., SPLADE)
    FULL_TEXT = "full_text"            # Generic full-text sparse vectors
    TRIGRAMS = "trigrams"              # Sparse vectors using trigrams tokenization
    WHITESPACE = "whitespace"          # Sparse vectors using whitespace tokenization
    WMTR = "wmtr"                      # Weighted Multilevel Token Representation
    KEYWORD = "keyword"                # KEYWORD (alias for wmtr tokenizer)
    
    # Custom vector types (user-provided)
    DENSE_CUSTOM = "dense_custom"      # User-provided dense vectors
    SPARSE_CUSTOM = "sparse_custom"    # User-provided sparse vectors
    
    @classmethod
    def all(cls) -> list[str]:
        """Return all available vector types."""
        return [cls.DENSE_MODEL, cls.SPARSE_MODEL, cls.FULL_TEXT, cls.TRIGRAMS, cls.WHITESPACE, cls.WMTR, cls.KEYWORD, cls.DENSE_CUSTOM, cls.SPARSE_CUSTOM]
    
    @classmethod
    def sparse_types(cls) -> list[str]:
        """Return all sparse vector types."""
        return [cls.SPARSE_MODEL, cls.FULL_TEXT, cls.TRIGRAMS, cls.WHITESPACE, cls.WMTR, cls.KEYWORD, cls.SPARSE_CUSTOM]
    
    @classmethod
    def dense_types(cls) -> list[str]:
        """Return all dense vector types."""
        return [cls.DENSE_MODEL, cls.DENSE_CUSTOM]
    
    @classmethod
    def transformer_based(cls) -> list[str]:
        """Return vector types that use transformer models."""
        return [cls.DENSE_MODEL, cls.SPARSE_MODEL]
    
    @classmethod
    def custom_tokenization(cls) -> list[str]:
        """Return vector types that use custom tokenization logic."""
        return [cls.FULL_TEXT, cls.TRIGRAMS, cls.WHITESPACE, cls.WMTR]
    
    @classmethod
    def custom_vectors(cls) -> list[str]:
        """Return vector types that use user-provided vectors."""
        return [cls.DENSE_CUSTOM, cls.SPARSE_CUSTOM]
    
    @classmethod
    def is_dense(cls, vector_type: str) -> bool:
        """Check if a vector type is dense (embedding-based)."""
        return vector_type in cls.dense_types()


# Type-safe literal for vector types
VectorTypeLiteral = Literal[
    VectorType.DENSE_MODEL,
    VectorType.SPARSE_MODEL, 
    VectorType.FULL_TEXT,
    VectorType.TRIGRAMS,
    VectorType.WHITESPACE,
    VectorType.WMTR,
    VectorType.KEYWORD,
    VectorType.DENSE_CUSTOM,
    VectorType.SPARSE_CUSTOM,
]

# Dense vector distance metrics
class DenseDistance:
    COSINE = "cosine"
    DOT = "dot"
    EUCLID = "euclid"
    
    @classmethod
    def all(cls) -> list[str]:
        """Return all available dense distance metrics."""
        return [cls.COSINE, cls.DOT, cls.EUCLID]
    
    @classmethod
    def is_valid(cls, distance: str) -> bool:
        """Check if a distance metric is valid."""
        return distance in cls.all()


# Type-safe literal for dense distance metrics
DenseDistanceLiteral = Literal[
    DenseDistance.COSINE,
    DenseDistance.DOT,
    DenseDistance.EUCLID
]

# Document field types
class DocumentField:
    NAME = "name"
    DESCRIPTION = "description" 
    CONTENT = "content"
    
    @classmethod
    def all(cls) -> list[str]:
        """Return all available document fields."""
        return [cls.NAME, cls.DESCRIPTION, cls.CONTENT]

# Type-safe literal for document fields
DocumentFieldLiteral = Literal[
    DocumentField.NAME,
    DocumentField.DESCRIPTION,
    DocumentField.CONTENT
]

# Document processing status values
class QueuedDocumentStatus:
    QUEUED = "queued"
    REQUEUED = "requeued"
    INDEXED = "indexed"
    FAILED = "failed"
    
    @classmethod
    def all(cls) -> list[str]:
        """Return all available document status values."""
        return [cls.QUEUED, cls.REQUEUED, cls.INDEXED, cls.FAILED]
    
    @classmethod
    def is_valid(cls, status: str) -> bool:
        """Check if a status value is valid."""
        return status in cls.all()
    
    @classmethod
    def is_failed(cls, status: str) -> bool:
        """Check if a status indicates failure."""
        return status.startswith(cls.FAILED)

# Type-safe literal for document status values
QueuedDocumentStatusLiteral = Literal[
    QueuedDocumentStatus.QUEUED,
    QueuedDocumentStatus.REQUEUED,
    QueuedDocumentStatus.INDEXED,
    QueuedDocumentStatus.FAILED
]

# Metadata value types
class MetadataValueType:
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    
    @classmethod
    def all(cls) -> list[str]:
        """Return all available metadata value types."""
        return [cls.STRING, cls.INTEGER, cls.FLOAT, cls.BOOLEAN, cls.DATETIME]

# Type-safe literal for metadata value types
MetadataValueTypeLiteral = Literal[
    MetadataValueType.STRING,
    MetadataValueType.INTEGER,
    MetadataValueType.FLOAT,
    MetadataValueType.BOOLEAN,
    MetadataValueType.DATETIME
]

# Metadata filter operators
MetadataFilterOpLiteral = Literal["eq", "lt", "gt", "lte", "gte"]

# Database features that can be detected during probing
class DatabaseFeatures:
    DENSE_VECTORS = "dense_vectors"


class DatabaseInfo:
    """Information about database capabilities and status."""
    
    def __init__(
        self,
        version: str,
        features: dict[str, bool],
        connection_healthy: bool = True,
        error_message: str | None = None
    ):
        self.version = version
        self.features = features
        self.connection_healthy = connection_healthy
        self.error_message = error_message
    
    def has_feature(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        return self.features.get(feature, False)
    
    def __str__(self) -> str:
        status = "healthy" if self.connection_healthy else "unhealthy"
        return f"DatabaseInfo(version={self.version}, features={list(self.features.keys())}, status={status})"
