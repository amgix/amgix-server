"""
Common database utilities for getting connected database instances.
"""

from typing import Any, Optional, Dict, Tuple
from urllib.parse import urlparse, urlunparse
from datetime import datetime
from src.core.database.base_factory import DatabaseFactory
from src.core.models.document import Document
from src.core.models.vector import CollectionConfigInternal, SearchQuery, VectorData, VectorConfigInternal, MetadataFilter
from src.core.common import MetadataValueType, MAX_INDEXED_METADATA_VALUE_LENGTH, VectorType
from src.core.common.enums import DocumentField


class AmgixValidationError(Exception):
    """Exception raised when user input fails business validation."""
    pass


def _get_safe_url(url: str) -> str:
    """
    Convert URL to safe version with password hidden as ***.
    
    Args:
        url: Database connection URL
        
    Returns:
        str: Safe URL with password replaced by ***
    """
    parsed = urlparse(url)
    if parsed.password:
        # Replace password with ***
        safe_netloc = f"{parsed.username}:***@{parsed.hostname}"
        if parsed.port:
            safe_netloc += f":{parsed.port}"
        safe_parsed = parsed._replace(netloc=safe_netloc)
        return urlunparse(safe_parsed)
    return url


async def get_connected_database(connection_string: str, logger):
    """
    Common function to get database and ensure it's connected.
    
    Args:
        connection_string: Database connection string
        logger: Logger instance to use
        
    Returns:
        DatabaseBase: Connected and probed database instance
    """
    safe_url = _get_safe_url(connection_string)
    logger.info(f"Connecting to {safe_url} ...")
    
    database = DatabaseFactory.create(connection_string, logger=logger)
    await database.wait_connected()
    
    logger.info(f"Connected to {safe_url}")
    
    await database.probe()
    return database


def _metadata_value_matches_index_type(value: Any, expected_type: str) -> bool:
    if expected_type == MetadataValueType.STRING:
        return isinstance(value, str)
    if expected_type == MetadataValueType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == MetadataValueType.FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == MetadataValueType.BOOLEAN:
        return isinstance(value, bool)
    if expected_type == MetadataValueType.DATETIME:
        return isinstance(value, str) and _is_iso_datetime_string(value)
    return False


def validate_metadata_types(collection_config: CollectionConfigInternal, document: Document) -> None:
    """
    Validate that document metadata values match the types declared in collection_config.metadata_indexes.
    
    Args:
        collection_config: Collection configuration with metadata_indexes
        document: Document with metadata to validate
        
    Raises:
        AmgixValidationError: If a metadata key's value doesn't match the declared index type
    """
    if not collection_config.metadata_indexes:
        return
    
    if not document.metadata:
        return
    
    expected_types = {idx.key: idx.type for idx in collection_config.metadata_indexes}
    
    for key, expected_type in expected_types.items():
        if key in document.metadata:
            value = document.metadata[key]
            if value is None:
                continue
            if not _metadata_value_matches_index_type(value, expected_type):
                raise AmgixValidationError(
                    f"Metadata key '{key}' value does not match collection config expected type '{expected_type}'"
                )
            if (
                expected_type == MetadataValueType.STRING
                and isinstance(value, str)
                and len(value) > MAX_INDEXED_METADATA_VALUE_LENGTH
            ):
                raise AmgixValidationError(
                    f"String metadata value for key '{key}' exceeds {MAX_INDEXED_METADATA_VALUE_LENGTH} character limit"
                )


def _expected_non_custom_vector_slots(
    vector_configs: list[VectorConfigInternal],
) -> Dict[Tuple[str, str], VectorConfigInternal]:
    expected: Dict[Tuple[str, str], VectorConfigInternal] = {}
    for config in vector_configs:
        if config.type in VectorType.custom_vectors():
            continue
        for field in config.index_fields:
            expected[(config.name, field)] = config
    return expected


def _validate_provided_vector_shape(vd: VectorData, config: VectorConfigInternal) -> None:
    if VectorType.is_dense(config.type):
        if not vd.dense_vector:
            raise AmgixValidationError(
                f"Vector '{vd.vector_name}' field '{vd.field}' requires dense_vector"
            )
        if config.dimensions is not None and len(vd.dense_vector) != config.dimensions:
            raise AmgixValidationError(
                f"Vector '{vd.vector_name}' field '{vd.field}' has {len(vd.dense_vector)} "
                f"dimensions, expected {config.dimensions}"
            )
        return
    if vd.sparse_indices is None or vd.sparse_values is None:
        raise AmgixValidationError(
            f"Vector '{vd.vector_name}' field '{vd.field}' requires sparse_indices and sparse_values"
        )
    if len(vd.sparse_indices) != len(vd.sparse_values):
        raise AmgixValidationError(
            f"Vector '{vd.vector_name}' field '{vd.field}': sparse_indices and sparse_values length mismatch"
        )
    if len(vd.sparse_indices) > config.top_k:
        raise AmgixValidationError(
            f"Vector '{vd.vector_name}' field '{vd.field}' has {len(vd.sparse_indices)} entries, "
            f"max allowed: {config.top_k}"
        )


def validate_document_vectors(
    collection_config: CollectionConfigInternal,
    document: Document,
) -> None:
    """
    Validate precomputed document vectors when provided on upsert.

    When ``vectors`` is omitted, validation is skipped. When present, every
    non-custom collection vector slot must be included exactly once with matching
    type and shape.
    """
    if document.vectors is None:
        return
    if not document.vectors:
        raise AmgixValidationError(
            "vectors must be omitted or contain the complete non-custom vector set"
        )

    expected = _expected_non_custom_vector_slots(collection_config.vectors)
    provided: Dict[Tuple[str, str], VectorData] = {}
    for vd in document.vectors:
        if vd.vector_type in VectorType.custom_vectors():
            raise AmgixValidationError(
                f"Vector '{vd.vector_name}' field '{vd.field}' has type '{vd.vector_type}'; "
                "custom vector types must use custom_vectors"
            )
        key = (vd.vector_name, vd.field)
        if key in provided:
            raise AmgixValidationError(
                f"Duplicate vector entry for '{vd.vector_name}' field '{vd.field}'"
            )
        config = expected.get(key)
        if config is None:
            raise AmgixValidationError(
                f"Unexpected vector '{vd.vector_name}' field '{vd.field}' "
                "(not a non-custom collection vector slot)"
            )
        if vd.vector_type != config.type:
            raise AmgixValidationError(
                f"Vector '{vd.vector_name}' field '{vd.field}' has type '{vd.vector_type}', "
                f"expected '{config.type}'"
            )
        _validate_provided_vector_shape(vd, config)
        provided[key] = vd

    missing = set(expected.keys()) - set(provided.keys())
    if missing:
        missing_labels = ", ".join(f"{name}/{field}" for name, field in sorted(missing))
        raise AmgixValidationError(
            f"Incomplete vectors: missing non-custom slots: {missing_labels}"
        )

    if document.custom_vectors:
        custom_keys = {(cv.vector_name, cv.field) for cv in document.custom_vectors}
        overlap = custom_keys & set(provided.keys())
        if overlap:
            overlap_labels = ", ".join(f"{name}/{field}" for name, field in sorted(overlap))
            raise AmgixValidationError(
                f"Duplicate vector slots in vectors and custom_vectors: {overlap_labels}"
            )


def _is_iso_datetime_string(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate_metadata_filter(collection_config: CollectionConfigInternal, metadata_filter: MetadataFilter) -> None:
    """
    Validate metadata filter keys and value types against collection metadata_indexes.

    Args:
        collection_config: Collection configuration with metadata indexes
        metadata_filter: Recursive metadata filter to validate

    Raises:
        AmgixValidationError: If key is not indexed, operator/type is invalid, or value type mismatches index type
    """
    if not collection_config.metadata_indexes:
        raise AmgixValidationError("Collection has no metadata_indexes defined. Cannot filter on metadata.")

    indexed_types = {idx.key: idx.type for idx in collection_config.metadata_indexes}

    def validate_field_condition(filter_node: MetadataFilter) -> None:
        key = filter_node.key
        op = filter_node.op
        value = filter_node.value

        if key is None:
            return

        if key not in indexed_types:
            raise AmgixValidationError(f"Metadata filter key '{key}' is not indexed in collection metadata_indexes")

        expected_type = indexed_types[key]

        if expected_type == MetadataValueType.STRING:
            if op not in ("eq", "neq"):
                raise AmgixValidationError(f"Metadata filter operator '{op}' is not supported for string key '{key}'. Use 'eq' or '!='.")
            if not isinstance(value, str):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be a string")

        elif expected_type == MetadataValueType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be an integer")

        elif expected_type == MetadataValueType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be a number")

        elif expected_type == MetadataValueType.BOOLEAN:
            if op not in ("eq", "neq"):
                raise AmgixValidationError(f"Metadata filter operator '{op}' is not supported for boolean key '{key}'. Use 'eq' or '!='.")
            if not isinstance(value, bool):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be a boolean")

        elif expected_type == MetadataValueType.DATETIME:
            if not isinstance(value, str) or not _is_iso_datetime_string(value):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be an ISO datetime string")

    def validate_filter_recursive(filter_node: MetadataFilter) -> None:
        validate_field_condition(filter_node)

        if filter_node.and_:
            for child in filter_node.and_:
                validate_filter_recursive(child)
        if filter_node.or_:
            for child in filter_node.or_:
                validate_filter_recursive(child)
        if filter_node.not_:
            validate_filter_recursive(filter_node.not_)

    validate_filter_recursive(metadata_filter)


def resolve_skippable_fields(query: SearchQuery, required_fields: "set | frozenset" = frozenset()) -> set:
    """
    Fields from query.exclude that are safe to skip fetching at the DB layer for this
    search, i.e. excluded by the caller AND not present in required_fields.

    required_fields is computed once by the caller (see
    search_join.required_fields_for_joins) from the already-parsed join specs, so
    the join expression doesn't need to be parsed again here.
    """
    if not query.exclude:
        return set()
    return set(query.exclude) - set(required_fields)


def needs_revectorization(
    incoming: Document,
    existing: Optional[Document],
    collection_config: CollectionConfigInternal,
    store_content: bool,
) -> bool:
    if existing is None:
        return True
    if incoming.vectors is not None:
        return True
    if incoming.custom_vectors:
        return True
    indexed_fields: set[str] = set()
    for vector_config in collection_config.vectors:
        for field in vector_config.index_fields:
            indexed_fields.add(field)
    if DocumentField.CONTENT in indexed_fields and not store_content:
        return True
    for field in indexed_fields:
        if getattr(incoming, field) != getattr(existing, field):
            return True
    return False
