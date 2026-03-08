"""
Common database utilities for getting connected database instances.
"""

from urllib.parse import urlparse, urlunparse
from datetime import datetime
from src.core.database.base_factory import DatabaseFactory
from src.core.models.vector import CollectionConfigInternal
from src.core.models.document import Document
from src.core.models.vector import MetadataFilter
from src.core.common import MetadataValueType


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


def validate_metadata_types(collection_config: CollectionConfigInternal, document: Document) -> None:
    """
    Validate that document metadata types match the types declared in collection_config.metadata_indexes.
    
    For each key in metadata_indexes, if that key exists in document.metadata,
    validates that the MetaValue.type matches the declared type in metadata_indexes.
    
    Args:
        collection_config: Collection configuration with metadata_indexes
        document: Document with metadata to validate
        
    Raises:
        AmgixValidationError: If a metadata key's type doesn't match the declared type
    """
    if not collection_config.metadata_indexes:
        return
    
    if not document.metadata:
        return
    
    # Build mapping of key -> expected type from metadata_indexes
    expected_types = {idx.key: idx.type for idx in collection_config.metadata_indexes}
    
    # Validate each indexed metadata field
    for key, expected_type in expected_types.items():
        if key in document.metadata:
            meta_value = document.metadata[key]
            actual_type = meta_value.type
            
            if actual_type != expected_type:
                raise AmgixValidationError(
                    f"Metadata key '{key}' has type '{actual_type}' but collection config expects type '{expected_type}'"
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
            if op != "eq":
                raise AmgixValidationError(f"Metadata filter operator '{op}' is not supported for string key '{key}'. Use 'eq'.")
            if not isinstance(value, str):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be a string")

        elif expected_type == MetadataValueType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be an integer")

        elif expected_type == MetadataValueType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AmgixValidationError(f"Metadata filter value for key '{key}' must be a number")

        elif expected_type == MetadataValueType.BOOLEAN:
            if op != "eq":
                raise AmgixValidationError(f"Metadata filter operator '{op}' is not supported for boolean key '{key}'. Use 'eq'.")
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
