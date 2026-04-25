from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator
import re
import json

from .vector import VectorData, CustomDocumentVector
from src.core.common import (
    MAX_METADATA_KEY_LENGTH, MAX_METADATA_VALUE_LENGTH, MAX_DOCUMENT_TAGS_COUNT,
    MAX_DOCUMENT_ID_LENGTH, MAX_DOCUMENT_NAME_LENGTH, MAX_DOCUMENT_DESCRIPTION_LENGTH,
    MAX_DOCUMENT_CONTENT_LENGTH, MAX_DOCUMENT_TAG_LENGTH, QueuedDocumentStatusLiteral,
    QueueOperationTypeLiteral,
    MetadataValueType, MetadataValueTypeLiteral
)


class MetaValue(BaseModel):
    """Metadata value with explicit type"""
    value: Any = Field(..., description="The metadata value")
    type: MetadataValueTypeLiteral = Field(
        default=MetadataValueType.STRING,
        description="Type of the metadata value"
    )


class VectorScore(BaseModel):
    """Individual vector score for a search result"""
    field: str = Field(..., description="Document field that was searched (name, description, content)")
    vector: str = Field(..., description="Vector name that was used (wmtr, splade, dense, etc.)")
    score: float = Field(..., description="Raw score from this vector")
    rank: int = Field(..., description="Rank of this result within this vector's results (1-based)")


class Document(BaseModel):
    """
    Document model representing the structure of documents in the system.
    
    This model is used for document ingestion, retrieval, and search results.
    Content is excluded from database storage for performance and cost reasons.
    """
    id: str = Field(..., max_length=MAX_DOCUMENT_ID_LENGTH, description=f"Unique identifier for the document (max {MAX_DOCUMENT_ID_LENGTH} characters)")
    timestamp: datetime = Field(..., description="UTC timestamp when the document was created/updated")
    tags: Optional[List[str]] = Field(
        default=None,
        max_length=MAX_DOCUMENT_TAGS_COUNT,
        description=f"List of document tags (max {MAX_DOCUMENT_TAGS_COUNT} items; each max {MAX_DOCUMENT_TAG_LENGTH} characters; cannot contain pipe characters)"
    )
    name: Optional[str] = Field(None, max_length=MAX_DOCUMENT_NAME_LENGTH, description=f"Document name (max {MAX_DOCUMENT_NAME_LENGTH} characters)")
    description: Optional[str] = Field(None, max_length=MAX_DOCUMENT_DESCRIPTION_LENGTH, description=f"Brief description of the document (max {MAX_DOCUMENT_DESCRIPTION_LENGTH} characters)")
    content: Optional[str] = Field(None, max_length=MAX_DOCUMENT_CONTENT_LENGTH, description=f"Main content of the document (max {MAX_DOCUMENT_CONTENT_LENGTH} characters)")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of metadata key-value pairs. Values can be simple types (string, int, float, bool) or MetaValue objects (required for datetime)"
    )
    custom_vectors: Optional[List[CustomDocumentVector]] = Field(
        default=None,
        description="Pre-generated custom vectors for this document (optional)"
    )
    
    model_config = {
        "populate_by_name": True,
        "extra": "forbid"
    }

    
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v):
        if not v or not v.strip():
            raise ValueError("Document ID cannot be empty or whitespace")
        
        if not re.match(r"^[a-zA-Z0-9_-]+$", v.strip()):
            raise ValueError("Document ID can only contain letters, numbers, underscores, and hyphens")
        
        return v.strip()  # Return stripped value to remove leading/trailing whitespace
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp_utc(cls, v):
        if not isinstance(v, datetime):
            raise ValueError("Timestamp must be a datetime object")
        if v.tzinfo is None:
            raise ValueError("Timestamp must include timezone information")
        if v.tzinfo != timezone.utc:
            raise ValueError("Timestamp must be in UTC timezone")
        return v

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and clean tags: trim whitespace, filter empty tags, check format, check individual lengths, check for duplicates."""
        if v is None:
            return v
        
        # Trim whitespace and filter out empty tags
        cleaned_tags = [tag.strip() for tag in v if tag.strip()]
        
        # Check format and individual tag length limits
        for tag in cleaned_tags:
            if '|' in tag:
                raise ValueError(f"Tag '{tag}' cannot contain pipe characters (|)")
            if len(tag) > MAX_DOCUMENT_TAG_LENGTH:
                raise ValueError(f"Tag '{tag}' exceeds {MAX_DOCUMENT_TAG_LENGTH} character limit")
        
        # Check for duplicates (after trimming)
        if len(set(cleaned_tags)) != len(cleaned_tags):
            raise ValueError("Tags must not contain duplicates")
        
        return cleaned_tags

    @field_validator('metadata', mode='before')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, MetaValue]]:
        """Validate metadata keys and convert values to MetaValue instances.
        
        Accepts raw values (string, int, float, bool) and converts them to MetaValue.
        Datetime values must be provided as MetaValue objects (or dicts) with type='datetime'.
        """
        if v is None:
            return None
        
        if not isinstance(v, dict):
            raise ValueError("Metadata must be a dictionary")
        
        validated_metadata = {}
        for key, value in v.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key).__name__}")
            if not re.match(r"^[a-zA-Z0-9_-]+$", key):
                raise ValueError(f"Metadata key '{key}' can only contain letters, numbers, underscores, and hyphens")
            if len(key) > MAX_METADATA_KEY_LENGTH:
                raise ValueError(f"Metadata key '{key}' exceeds {MAX_METADATA_KEY_LENGTH} character limit")
            
            # Convert value to MetaValue
            meta_value = None
            
            # If already a MetaValue instance, use it
            if isinstance(value, MetaValue):
                meta_value = value
            # If it's a dict (could be MetaValue dict representation), try to parse it
            elif isinstance(value, dict):
                if 'value' in value and 'type' in value:
                    meta_value = MetaValue(**value)
                else:
                    raise ValueError(f"Metadata value for key '{key}' is a dict but missing 'value' or 'type' fields. For datetime, use {{'value': '...', 'type': 'datetime'}}")
            # If it's a raw value, infer type and convert
            # Note: bool must be checked before int because bool is a subclass of int in Python
            elif isinstance(value, str):
                meta_value = MetaValue(value=value, type=MetadataValueType.STRING)
            elif isinstance(value, bool):
                meta_value = MetaValue(value=value, type=MetadataValueType.BOOLEAN)
            elif isinstance(value, int):
                meta_value = MetaValue(value=value, type=MetadataValueType.INTEGER)
            elif isinstance(value, float):
                meta_value = MetaValue(value=value, type=MetadataValueType.FLOAT)
            else:
                raise ValueError(f"Metadata value for key '{key}' must be string, int, float, bool, or MetaValue (required for datetime), got {type(value).__name__}")
            
            # Validate type is in allowed values
            if meta_value.type not in MetadataValueType.all():
                raise ValueError(f"Invalid metadata type '{meta_value.type}' for key '{key}'. Allowed types: {MetadataValueType.all()}")
            
            # Validate value type matches declared type
            if meta_value.type == MetadataValueType.STRING:
                if not isinstance(meta_value.value, str):
                    raise ValueError(f"Metadata value for key '{key}' must be string for type='string', got {type(meta_value.value).__name__}")
                if len(meta_value.value) > MAX_METADATA_VALUE_LENGTH:
                    raise ValueError(f"String metadata value for key '{key}' exceeds {MAX_METADATA_VALUE_LENGTH} character limit")
            elif meta_value.type == MetadataValueType.INTEGER:
                if not isinstance(meta_value.value, int):
                    raise ValueError(f"Metadata value for key '{key}' must be integer for type='integer', got {type(meta_value.value).__name__}")
            elif meta_value.type == MetadataValueType.FLOAT:
                if not isinstance(meta_value.value, (int, float)):
                    raise ValueError(f"Metadata value for key '{key}' must be number for type='float', got {type(meta_value.value).__name__}")
            elif meta_value.type == MetadataValueType.BOOLEAN:
                if not isinstance(meta_value.value, bool):
                    raise ValueError(f"Metadata value for key '{key}' must be boolean for type='boolean', got {type(meta_value.value).__name__}")
            elif meta_value.type == MetadataValueType.DATETIME:
                if not isinstance(meta_value.value, str):
                    raise ValueError(f"Metadata value for key '{key}' must be string (ISO 8601) for type='datetime', got {type(meta_value.value).__name__}")
                # Validate ISO 8601 format
                try:
                    datetime.fromisoformat(meta_value.value.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    raise ValueError(f"Metadata value for key '{key}' must be a valid ISO 8601 datetime string, got '{meta_value.value}'")
            
            validated_metadata[key] = meta_value
        
        return validated_metadata

   
    @model_validator(mode='after')
    def validate_at_least_one_field_has_content(self):
        """Ensure at least one of name, description, or content has non-empty content."""
        # Skip validation for SearchResult - it's constructed internally and doesn't need validation
        # Check by class name to avoid circular import
        if self.__class__.__name__ == 'SearchResult':
            return self
        
        name_content = self.name.strip() if self.name else None
        description_content = self.description.strip() if self.description else None
        content_content = self.content.strip() if self.content else None
        
        if not (name_content or description_content or content_content):
            raise ValueError("Document must have at least one non-empty field (name, description, or content)")
        
        return self

    
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], store_content: bool = False, skip_validation: bool = False) -> "Document":
        """Create a Document instance from a dictionary"""
        # Convert metadata to dict format if present
        if "metadata" in data:
            meta_raw = data["metadata"]
            if isinstance(meta_raw, str):
                try:
                    meta_parsed = json.loads(meta_raw) if meta_raw else {}
                except Exception:
                    meta_parsed = {}
            elif isinstance(meta_raw, dict):
                meta_parsed = meta_raw
            else:
                meta_parsed = {}
            # Pydantic will automatically convert dict values to MetaValue instances
            data["metadata"] = meta_parsed
        
        # Timestamp: ensure proper datetime with UTC tzinfo
        if "timestamp" in data:
            if isinstance(data["timestamp"], str):
                # Convert string timestamp to datetime
                try:
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
                except ValueError:
                    # Keep as string if we can't parse it - validation will catch this if not skipping
                    pass
            
            if isinstance(data["timestamp"], datetime) and getattr(data["timestamp"], 'tzinfo', None) is None:
                data["timestamp"] = data["timestamp"].replace(tzinfo=timezone.utc)
        
        # Set content to None if not present or if content storage is disabled
        if not store_content or "content" not in data:
            data["content"] = None
        
        if skip_validation:
            # Create document without running validators (useful for DB retrieval)
            return cls.model_construct(**data)
        else:
            return cls(**data)


class DocumentWithVectors(Document):
    """
    Document with precalculated vectors.
    This is an internal model used between the API layer and database layer.
    API layer calculates vectors and passes this enriched document to the database layer.
    """
    vectors: List[VectorData] = Field(
        ..., description="Precalculated vectors for the document"
    )
    token_lengths: Dict[str, int] = Field(
        default_factory=dict, description="Token lengths per field_vector_name for sparse vectors"
    )


class QueueDocument(BaseModel):
    """
    Queue document model representing documents in the processing queue.
    
    This model is used for tracking document processing status and metadata
    in the system collection/table.
    """
    queue_id: str = Field(..., description="Unique identifier for this queue entry")
    collection_name: str = Field(..., description="Name of the collection this document belongs to")
    collection_id: str = Field(..., description="UUID identifier for the collection")
    doc_id: str = Field(..., description="Unique identifier for the document")
    op_type: QueueOperationTypeLiteral = Field(..., description="Operation type for this queue entry (upsert or delete)")
    doc_timestamp: datetime = Field(..., description="Caller-supplied operation timestamp used for ordering")
    status: QueuedDocumentStatusLiteral = Field(..., description="Current processing status of the document")
    info: Optional[str] = Field(None, description="Status information")
    document: Optional[Document] = Field(None, description="The document being processed (optional for status queries)")
    created_at: datetime = Field(..., description="UTC timestamp when the document was first added to the queue")
    timestamp: datetime = Field(..., description="UTC timestamp when the status was last updated")
    try_count: int = Field(default=0, description="Number of processing attempts")
    
    model_config = {
        "populate_by_name": True,
        "extra": "forbid"
    }


class QueueInfo(BaseModel):
    """
    Queue statistics for a collection.
    Provides counts of documents in different queue states.
    """
    queued: int = Field(..., description="Number of documents queued for processing")
    requeued: int = Field(..., description="Number of documents requeued after failure")
    failed: int = Field(..., description="Number of failed documents")
    total: int = Field(..., description="Total number of queue entries")


class CollectionStatsResponse(BaseModel):
    """
    Persisted index statistics for a collection (encoder-maintained counts) and queue counts.
    """
    doc_count: int = Field(..., description="Number of documents reflected in collection stats")
    queue: QueueInfo = Field(..., description="Counts of documents in each queue state")

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",
    }


class SearchResult(Document):
    """
    A search result containing document information and its relevance score.
    Inherits from Document but excludes content field and adds score.
    
    Validation is disabled for SearchResult as it is constructed internally from validated data.
    """
    model_config = {
        "validate_assignment": False,
        "revalidate_instances": "never",
        "populate_by_name": True,
        "extra": "forbid"
    }
    
    score: float = Field(..., description="The relevance score for this document")
    vector_scores: List[VectorScore] = Field(default_factory=list, description="Raw per-vector scores with field, vector, score, and rank information")
    
    # Override content field to exclude it from this model
    content: None = Field(None, exclude=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], skip_validation: bool = False) -> "SearchResult":
        """
        Create a SearchResult instance from a dictionary, with proper type conversion.
        
        Args:
            data: Dictionary containing search result data
            skip_validation: Whether to skip validation
            
        Returns:
            SearchResult: A properly constructed search result
        """
        # Reuse Document's from_dict to handle timestamp conversion and other common fields
        # We pass store_content=False since SearchResult doesn't need content
        doc_data = data.copy()
        if 'score' in doc_data:
            score = doc_data.pop('score')
            vector_scores_raw = doc_data.pop('vector_scores', {})
            
            # Convert vector_scores to List[VectorScore] format
            vector_scores = []
            if isinstance(vector_scores_raw, dict):
                # Old format: Dict[str, float] -> convert to List[VectorScore]
                for field_vector_name, score_val in vector_scores_raw.items():
                    # Parse field_vector_name (e.g., "name_wmtr" -> field="name", vector="wmtr")
                    field, vector = field_vector_name.rsplit('_', 1)
                    vector_scores.append(VectorScore(field=field, vector=vector, score=score_val, rank=0))
            elif isinstance(vector_scores_raw, list):
                # New format: already List[VectorScore] or List[dict]
                for item in vector_scores_raw:
                    if isinstance(item, dict):
                        vector_scores.append(VectorScore(**item))
                    else:
                        vector_scores.append(item)
            
            doc = Document.from_dict(doc_data, store_content=False, skip_validation=skip_validation)
            # Create a new dict with Document fields + score
            result_data = doc.model_dump()
            result_data['score'] = score
            result_data['vector_scores'] = vector_scores
            # Now construct the SearchResult
            if skip_validation:
                return cls.model_construct(**result_data)
            else:
                return cls(**result_data)
        else:
            # If no score, just convert to Document and add default score
            doc = Document.from_dict(doc_data, store_content=False, skip_validation=skip_validation)
            result_data = doc.model_dump()
            result_data['score'] = 0.0
            result_data['vector_scores'] = data.get('vector_scores', {})
            if skip_validation:
                return cls.model_construct(**result_data)
            else:
                return cls(**result_data)


class DocumentStatus(BaseModel):
    """
    Individual status entry for a document.
    """
    status: QueuedDocumentStatusLiteral = Field(..., description="Status of the document (queued, requeued, indexed, failed)")
    info: Optional[str] = Field(None, description="Status information")
    timestamp: datetime = Field(..., description="When this status occurred")
    queue_id: Optional[str] = Field(None, description="Queue entry ID (only for queue-related statuses)")
    try_count: Optional[int] = Field(None, description="Number of processing attempts (only for failed statuses)")


class DocumentStatusResponse(BaseModel):
    """
    Complete status response for a document including queue states.
    """
    statuses: List[DocumentStatus] = Field(..., description="List of statuses sorted by timestamp (newest first)")
