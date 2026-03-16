from typing import List, Optional, Any, Union, Tuple, Dict
import re
from pydantic import BaseModel, Field, field_validator, model_validator

from ..common import (
    VectorType, VectorTypeLiteral, DocumentField, DocumentFieldLiteral, 
    DenseDistance, DenseDistanceLiteral, MAX_VECTOR_NAME_LENGTH, 
    MAX_MODEL_NAME_LENGTH, MAX_DOCUMENT_TAGS_COUNT, MAX_DOCUMENT_TAG_LENGTH,
    MAX_SEARCH_QUERY_LENGTH, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT, MAX_TOP_K_VALUE,
    MAX_VECTOR_DIMENSIONS, DEFAULT_TOP_K, LANGUAGE_DETECTION_CONFIDENCE,
    WMTR_WORD_WEIGHT_PERCENTAGE,
    MetadataValueTypeLiteral, MetadataFilterOpLiteral, MAX_METADATA_KEY_LENGTH
)


class CustomVector(BaseModel):
    """Base custom vector model for search queries"""
    vector_name: str = Field(..., max_length=MAX_VECTOR_NAME_LENGTH, description="Name of the vector (must match collection config)")
    vector: Union[List[float], List[Tuple[int, float]]] = Field(
        ..., 
        description="Vector data: dense (list of floats) or sparse (list of (index, value) tuples)"
    )
    
    @field_validator('vector_name')
    @classmethod
    def validate_vector_name_format(cls, v):
        if not v or not v.strip():
            raise ValueError("Vector name cannot be empty or whitespace")
        
        if not re.match(r"^[a-zA-Z0-9_-]+$", v.strip()):
            raise ValueError("Vector name can only contain letters, numbers, underscores, and hyphens")
        
        if len(v.strip()) > MAX_VECTOR_NAME_LENGTH:
            raise ValueError(f"Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters")
        
        return v.strip()
    
    @field_validator('vector')
    @classmethod
    def validate_vector_data(cls, v):
        if not v:
            raise ValueError("Vector data cannot be empty")
        
        if isinstance(v[0], (int, float)):
            # Dense vector - validate all elements are floats
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Dense vector must contain only numbers")
        elif isinstance(v[0], tuple):
            # Sparse vector - validate format and values
            if not all(isinstance(x, tuple) and len(x) == 2 for x in v):
                raise ValueError("Sparse vector must contain (index, value) tuples")
            if not all(isinstance(x[0], int) and isinstance(x[1], (int, float)) for x in v):
                raise ValueError("Sparse vector tuples must be (int, float) pairs")
        else:
            raise ValueError("Vector must be either list of numbers (dense) or list of (index, value) tuples (sparse)")
        
        return v


class CustomDocumentVector(CustomVector):
    """Custom vector for documents - adds field specification"""
    field: DocumentFieldLiteral = Field(..., description="Document field this vector is for (name, description, or content)")


class VectorConfig(BaseModel):
    """
    Configuration for a vector generation method.
    
    This model defines how vectors should be generated for a specific field or set of fields.
    Different vector types have different requirements and behaviors.
    """
    model_config = {"extra": "forbid"}
    name: str = Field(..., description="Unique name for this vector configuration")
    type: VectorTypeLiteral = Field(..., description="Type of vector (dense_model, sparse_model, full_text, trigrams, whitespace, wmtr, dense_custom, sparse_custom)")
    model: Optional[str] = Field(
        None, max_length=MAX_MODEL_NAME_LENGTH, description=f"Model name for transformer-based vectors (max {MAX_MODEL_NAME_LENGTH} characters, e.g., 'sentence-transformers/all-MiniLM-L6-v2'). Used for document indexing."
    )
    revision: Optional[str] = Field(
        None, max_length=MAX_MODEL_NAME_LENGTH, description=f"Optional model revision (max {MAX_MODEL_NAME_LENGTH} characters, branch/tag/commit) for specific model version. Used for document indexing."
    )
    query_model: Optional[str] = Field(
        None, max_length=MAX_MODEL_NAME_LENGTH, description=f"Optional model name for query vectorization (max {MAX_MODEL_NAME_LENGTH} characters). If not specified, uses 'model' for both documents and queries."
    )
    query_revision: Optional[str] = Field(
        None, max_length=MAX_MODEL_NAME_LENGTH, description=f"Optional model revision for query vectorization (max {MAX_MODEL_NAME_LENGTH} characters). If not specified, uses 'revision' for both documents and queries."
    )
    dimensions: Optional[int] = Field(
        None, description="Dimensions for the vector. Required for dense_custom vectors. For dense_model vectors, auto-detected if not specified."
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K, description="Number of top-scoring terms to keep for sparse vectors. Used by sparse_model, full_text, trigrams, whitespace, wmtr, and sparse_custom vectors. Ignored by dense vectors."
    )
    wmtr_word_weight: int = Field(
        default=WMTR_WORD_WEIGHT_PERCENTAGE,
        ge=0,
        le=100,
        description="Percentage of WMTR top_k allocated to word weights."
    )
    index_fields: List[DocumentFieldLiteral] = Field(
        default_factory=lambda: [DocumentField.CONTENT], 
        description="List of fields to index with this vector (name, description, content). Defaults to ['content'] if not specified."
    )
    language_default_code: Optional[str] = Field(
        'en', description="Two-letter ISO 639-1 language code for language-based vector types (e.g., 'en', 'es', 'fr')"
    )
    language_detect: bool = Field(
        False, description="Whether to automatically detect language for language-based vector types"
    )
    language_confidence: float = Field(
        LANGUAGE_DETECTION_CONFIDENCE, description="Minimum confidence threshold for language detection. If detection confidence is below this value, language_default_code will be used instead."
    )
    normalization: Optional[bool] = Field(
        None, description="Whether to normalize vectors. Only supported for dense vectors. Sparse vectors do not support normalization."
    )
    dense_distance: "DenseDistanceLiteral" = Field(
        default=DenseDistance.COSINE, description="Distance metric for dense vectors (cosine, dot, euclid). Defaults to cosine."
    )
    keep_case: Optional[bool] = Field(
        default=False, description="Whether to keep original case for text preprocessing. Only applies to model-based vectors (dense_model, sparse_model). Defaults to False (lowercase)."
    )
    
    @model_validator(mode='before')
    @classmethod
    def normalize_type_case(cls, data: Any) -> Any:
        """Normalize vector type to lowercase for case-insensitive input."""
        if isinstance(data, dict) and 'type' in data and isinstance(data.get('type'), str):
            data = data.copy()
            data['type'] = data['type'].lower()
        return data
    
    @model_validator(mode='before')
    @classmethod
    def transform_keyword_alias(cls, data: Any) -> Any:
        """Transform keyword type to wmtr tokenizer."""
        if isinstance(data, dict) and data.get('type') == VectorType.KEYWORD:
            # Transform to wmtr
            data = data.copy()
            data['type'] = VectorType.WMTR
        
        return data
    
    @field_validator('index_fields')
    @classmethod
    def validate_index_fields(cls, v):
        valid_fields = DocumentField.all()
        for field in v:
            if field not in valid_fields:
                raise ValueError(f"Index field must be one of {valid_fields}")
        return v
    
    @field_validator('language_confidence')
    @classmethod
    def validate_language_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("language_confidence must be between 0.0 and 1.0")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name_format(cls, v):
        if not v or not v.strip():
            raise ValueError("Vector configuration name cannot be empty or whitespace")
        
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Vector name can only contain letters, numbers, underscores, and hyphens")
        
        if len(v) > MAX_VECTOR_NAME_LENGTH:
            raise ValueError(f"Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters")
        
        return v
    
    @field_validator('language_default_code')
    @classmethod
    def validate_language_code_format(cls, v):
        if v is not None:
            if not (len(v) == 2 and v.isalpha()):
                raise ValueError("Language code must be a valid ISO 639-1 code (2 letters)")
        return v
    
    @field_validator('dense_distance')
    @classmethod
    def validate_dense_distance(cls, v):
        from ..common import DenseDistance
        if v not in DenseDistance.all():
            raise ValueError(f"dense_distance must be one of {DenseDistance.all()}")
        return v
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k_positive(cls, v):
        if v <= 0:
            raise ValueError("top_k must be positive (greater than 0)")
        if v > MAX_TOP_K_VALUE:
            raise ValueError(f"top_k cannot exceed {MAX_TOP_K_VALUE}")
        return v
    
    @model_validator(mode='after')
    def validate_language_config(self):
        if self.type in [VectorType.FULL_TEXT, VectorType.WHITESPACE, VectorType.WMTR] and self.language_default_code is None:
            raise ValueError(f"language_default_code is required for {self.type} vector type")
        return self
    
    @model_validator(mode='after')
    def validate_normalization_for_sparse_vectors(self):
        """Validate that normalization is only used with dense vectors."""
        if self.normalization and not VectorType.is_dense(self.type):
            raise ValueError(f"Normalization is not supported for sparse vector type '{self.type}'. Only dense vectors support normalization.")
        return self
    
    @model_validator(mode='after')
    def set_normalization_default(self):
        """Set normalization defaults: True for dense vectors, False for sparse vectors if not specified."""
        if self.normalization is None:
            if VectorType.is_dense(self.type):
                self.normalization = True
            else:
                self.normalization = False
        return self
    
    @model_validator(mode='after')
    def validate_dense_distance_for_dense_vectors(self):
        """Validate that dense_distance is only specified for dense vectors."""
        if self.dense_distance != "cosine" and not VectorType.is_dense(self.type):
            raise ValueError(f"dense_distance can only be specified for dense vectors. Current type: {self.type}")
        return self
    
    @model_validator(mode='after')
    def validate_model_requirements(self):
        """Validate model field requirements for different vector types."""
        if self.type in VectorType.transformer_based() and self.model is None:
            raise ValueError(f"Model is required for {self.type} vector type")
        
        if self.model is not None and self.type in VectorType.custom_tokenization():
            raise ValueError(f"Model should not be specified for {self.type} vector type")
        
        return self
    
    @model_validator(mode='after')
    def validate_custom_vector_config(self):
        """Validate that custom vector types don't have unnecessary fields and have required fields."""
        if self.type in VectorType.custom_vectors():
            if self.model is not None:
                raise ValueError(f"Model should not be specified for {self.type} vector type")
            if self.type == VectorType.SPARSE_CUSTOM and self.top_k is None:
                raise ValueError(f"top_k is required for {self.type} vector type")
            if self.type == VectorType.DENSE_CUSTOM and self.dimensions is None:
                raise ValueError(f"Dimensions are required for {self.type} vector type")
        
        # Validate dimensions range if provided
        if self.dimensions is not None:
            if self.dimensions <= 0:
                raise ValueError("Dimensions must be positive (greater than 0)")
            if self.dimensions > MAX_VECTOR_DIMENSIONS:
                raise ValueError(f"Dimensions cannot exceed {MAX_VECTOR_DIMENSIONS}")
        
        return self


class VectorConfigInternal(VectorConfig):
    """
    Internal vector configuration including dimensions field.
    Inherits from VectorConfig and adds dimensions for internal use.
    """
    version: int = Field(default=1, description="Vectorization version for compatibility")
    
    # No validation needed - all validation is handled by the parent VectorConfig class
    # This class is for internal use only, so we prioritize performance over validation


class MetadataIndex(BaseModel):
    """Metadata field index configuration"""
    key: str = Field(..., description="Metadata field key to index")
    type: MetadataValueTypeLiteral = Field(..., description="Type of the metadata value")
    
    @field_validator('key')
    @classmethod
    def validate_key_format(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Metadata key can only contain letters, numbers, underscores, and hyphens")
        
        if len(v) > MAX_METADATA_KEY_LENGTH:
            raise ValueError(f"Metadata key cannot exceed {MAX_METADATA_KEY_LENGTH} characters")
        
        return v


class CollectionConfig(BaseModel):
    """
    API model for collection configuration - uses VectorConfig.
    """
    model_config = {"extra": "forbid"}
    vectors: List[VectorConfig] = Field(
        ..., description="List of vector configurations for this collection"
    )
    store_content: bool = Field(
        default=False, description="Whether to store document content in the database. Defaults to False for performance and cost reasons."
    )
    metadata_indexes: Optional[List[MetadataIndex]] = Field(
        default=None,
        description="List of metadata fields to index for filtering and sorting"
    )
    
    @field_validator('vectors')
    @classmethod
    def validate_vectors_not_empty(cls, v):
        if not v:
            raise ValueError("Collection must have at least one vector configuration")
        return v
    
    @field_validator('vectors')
    @classmethod
    def validate_unique_vector_names(cls, v):
        """Validate that all vector names are unique within the collection."""
        names = [vector.name for vector in v]
        seen = set()
        duplicates = []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate vector names found: {', '.join(duplicates)}. Each vector name must be unique within a collection.")
        return v


class CollectionConfigInternal(BaseModel):
    """
    Internal configuration for creating a new collection.
    Defines the vectors to use for the collection with full vector configurations.
    Each vector configuration specifies which fields it should be applied to.
    """
    version: int = Field(default=1, description="Collection configuration version for compatibility")

    collection_id: str = Field(..., description="UUID identifier for the collection")
    vectors: List[VectorConfigInternal] = Field(
        ..., description="List of vector configurations for this collection"
    )
    store_content: bool = Field(
        default=False, description="Whether to store document content in the database. Defaults to False for performance and cost reasons."
    )
    metadata_indexes: Optional[List[MetadataIndex]] = Field(
        default=None,
        description="List of metadata fields to index for filtering and sorting"
    )


def internal_to_user_config(internal_config: CollectionConfigInternal) -> CollectionConfig:
    """
    Convert internal collection configuration to user-facing configuration.
    Removes internal fields like collection_id and version, and converts VectorConfigInternal to VectorConfig.
    
    Args:
        internal_config: Internal collection configuration with all fields
        
    Returns:
        CollectionConfig: User-facing configuration without internal fields
    """
    # Convert internal vectors to user-facing vectors
    user_vectors = []
    for vector in internal_config.vectors:
        # Create VectorConfig - include dimensions for dense_custom vectors
        user_vector = VectorConfig(
            name=vector.name,
            type=vector.type,
            model=vector.model,
            revision=vector.revision,
            dimensions=vector.dimensions,  # Include dimensions for all vectors
            top_k=vector.top_k,
            wmtr_word_weight=vector.wmtr_word_weight,
            index_fields=vector.index_fields,
            language_default_code=vector.language_default_code,
            language_detect=vector.language_detect,
            language_confidence=vector.language_confidence,
            normalization=vector.normalization,
            dense_distance=vector.dense_distance
        )
        user_vectors.append(user_vector)
    
    # Create user-facing config without internal fields
    user_config = CollectionConfig(
        vectors=user_vectors,
        store_content=internal_config.store_content,
        metadata_indexes=internal_config.metadata_indexes
    )
    
    return user_config


class VectorSearchWeight(BaseModel):
    """
    Configuration for a vector search weight.
    Used in search queries to specify which vectors to search with and their weights.
    """
    model_config = {"extra": "forbid"}
    vector_name: str = Field(..., description="Name of the vector to search with")
    weight: float = Field(default=1.0, description="Weight to apply to this vector's search results")
    field: DocumentFieldLiteral = Field(
        ..., description="Field to search with this vector (name, description, content)"
    )
    
    @field_validator('vector_name')
    @classmethod
    def validate_vector_name_format(cls, v):
        if not v or not v.strip():
            raise ValueError("Vector name cannot be empty or whitespace")
        
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Vector name can only contain letters, numbers, underscores, and hyphens")
        
        if len(v) > MAX_VECTOR_NAME_LENGTH:
            raise ValueError(f"Vector name cannot exceed {MAX_VECTOR_NAME_LENGTH} characters")
        
        return v
    
    # @field_validator('weight')
    # @classmethod
    # def validate_weight_range(cls, v):
    #     if v < 0.0 or v > 1.0:
    #         raise ValueError("Weight must be between 0.0 and 1.0")
    #     return v


class MetadataFilter(BaseModel):
    """Recursive metadata filter structure (modeled on Qdrant filter with and/or/not)."""

    and_: Optional[List["MetadataFilter"]] = Field(
        None, alias="and", description="All conditions in this list must match (AND)"
    )
    or_: Optional[List["MetadataFilter"]] = Field(
        None, alias="or", description="Any condition in this list must match (OR)"
    )
    not_: Optional["MetadataFilter"] = Field(
        None, alias="not", description="This condition must NOT match (NOT)"
    )

    key: Optional[str] = Field(None, description="Metadata field key (for field condition)")
    op: Optional[MetadataFilterOpLiteral] = Field(None, description="Comparison operator")
    value: Optional[Any] = Field(None, description="Value to compare against (for field condition)")

    @model_validator(mode='after')
    def validate_structure(self):
        has_boolean = any([self.and_ is not None, self.or_ is not None, self.not_ is not None])
        has_field = any([self.key is not None, self.op is not None, self.value is not None])

        if not has_boolean and not has_field:
            raise ValueError("Filter must have either boolean logic (and/or/not) or field condition")
        if has_boolean and has_field:
            raise ValueError("Filter cannot have both boolean logic and field condition at the same level")

        if has_field:
            if not self.key:
                raise ValueError("Field condition must specify 'key'")
            if not self.op:
                raise ValueError("Field condition must specify 'op' (operator)")
            if self.value is None:
                raise ValueError("Field condition must specify 'value'")

        return self

    model_config = {"populate_by_name": True}


MetadataFilter.model_rebuild()


class SearchQuery(BaseModel):
    """
    Configuration for a search query.
    Defines the query string and vector weights.
    This is the model that will be sent by end users to the search API endpoint.
    """
    model_config = {"extra": "forbid"}
    query: str = Field(..., max_length=MAX_SEARCH_QUERY_LENGTH, description=f"The search query string (max {MAX_SEARCH_QUERY_LENGTH} characters)")
    vector_weights: List[VectorSearchWeight] = Field(
        default=[], description="List of vectors, fields, and weights to use for searching. If empty, equal weights will be auto-generated for all available vectors."
    )
    custom_vectors: Optional[List[CustomVector]] = Field(
        default=None,
        description="Pre-generated custom vectors for this search query (optional)"
    )
    limit: int = Field(
        DEFAULT_SEARCH_LIMIT,
        ge=1,
        le=MAX_SEARCH_LIMIT,
        description=f"Maximum number of results to return (1 to {MAX_SEARCH_LIMIT})"
    )
    score_threshold: Optional[float] = Field(
        None, description="Optional minimum score threshold. Results below this score will be excluded"
    )
    document_tags: Optional[List[str]] = Field(
        None, max_length=MAX_DOCUMENT_TAGS_COUNT, description=f"Optional filter to include only documents with specific tags (max {MAX_DOCUMENT_TAGS_COUNT} tags, each max {MAX_DOCUMENT_TAG_LENGTH} characters; cannot contain pipe characters)"
    )
    document_tags_match_all: bool = Field(
        default=False, description="If True, documents must have ALL specified tags (AND). If False, documents must have ANY of the specified tags (OR)."
    )
    metadata_filter: Optional[MetadataFilter] = Field(
        None,
        description="Optional recursive metadata filter. Only fields declared in collection metadata_indexes can be filtered."
    )
    raw_scores: bool = Field(
        default=False, description="Whether to include individual vector scores in results"
    )
    
    
    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query string cannot be empty or whitespace")
        return v
    

    
    @field_validator('score_threshold')
    @classmethod
    def validate_score_threshold_number(cls, v):
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("Score threshold must be a number")
        return v
    
    @field_validator('document_tags')
    @classmethod
    def validate_document_tag_lengths(cls, v):
        f"""Validate that individual tags don't exceed {MAX_DOCUMENT_TAG_LENGTH} characters and use proper format."""
        if v is not None:
            for tag in v:
                if not tag or not tag.strip():
                    raise ValueError("Document tags cannot be empty or whitespace")
                if '|' in tag.strip():
                    raise ValueError(f"Document tag '{tag}' cannot contain pipe characters (|)")
                if len(tag.strip()) > MAX_DOCUMENT_TAG_LENGTH:
                    raise ValueError(f"Document tag '{tag}' exceeds {MAX_DOCUMENT_TAG_LENGTH} character limit")
        return v


class VectorData(BaseModel):
    """
    Vector data for a specific vector.
    Contains the precalculated vector values, which can be either dense or sparse.
    """
    vector_name: str = Field(..., description="Name of the vector")
    field: DocumentFieldLiteral = Field(
        ..., description="Field this vector is for (name, description, content)"
    )
    vector_type: VectorTypeLiteral = Field(..., description="Type of vector (dense_model, sparse_model, full_text, trigrams, whitespace, wmtr, dense_custom, sparse_custom)")
    dense_vector: Optional[List[float]] = Field(None, description="Dense vector values as a list of floats")
    sparse_indices: Optional[List[int]] = Field(None, description="Sparse vector indices (token positions)")
    sparse_values: Optional[List[float]] = Field(None, description="Sparse vector values (token weights)")
    
    @field_validator('vector_name')
    @classmethod
    def validate_vector_name_format(cls, v):
        if not v or not v.strip():
            raise ValueError("Vector name cannot be empty or whitespace")
        
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Vector name can only contain letters, numbers, underscores, and hyphens")
        
        if len(v) > 100:
            raise ValueError("Vector name cannot exceed 100 characters")
        
        return v
    
    @field_validator('sparse_values')
    @classmethod
    def validate_sparse_values(cls, v, info):
        indices = info.data.get('sparse_indices')
        if v is not None and indices is not None and len(v) != len(indices):
            raise ValueError("The length of sparse_values must match the length of sparse_indices")
        return v


class SearchQueryWithVectors(SearchQuery):
    """
    Search query with precalculated vectors.
    This is an internal model used between the API layer and database layer.
    API layer calculates vectors and passes this enriched query to the database layer.
    """
    vectors: List[VectorData] = Field(
        ..., description="Precalculated vectors for the search query"
    )


class ModelValidationResult(BaseModel):
    """Result of validating a single vector model."""
    valid: bool = Field(..., description="Whether the model is valid")
    dimension: Optional[int] = Field(None, description="Vector dimension (for dense models)")
    error: Optional[str] = Field(None, description="Error message if validation failed")


class ModelValidationResponse(BaseModel):
    """Response from model validation RPC call."""
    results: Optional[Dict[str, ModelValidationResult]] = Field(None, description="Validation results by model name")
    error: Optional[str] = Field(None, description="Global error message if validation failed")


    