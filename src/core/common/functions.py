"""
Utility functions used throughout the application.
"""
import zlib
from .constants import APP_PREFIX


def get_real_collection_name(user_collection_name: str) -> str:
    """
    Convert user-provided collection name to the actual collection name used internally.
    
    Args:
        user_collection_name: Collection name provided by the user
        
    Returns:
        str: Real collection name with app prefix
    """
    return f"{APP_PREFIX}_{user_collection_name}"


def get_user_collection_name(real_collection_name: str) -> str:
    """
    Convert internal collection name back to user-facing collection name.
    
    Args:
        real_collection_name: Internal collection name with app prefix
        
    Returns:
        str: User-facing collection name without prefix
    """
    if real_collection_name.startswith(f"{APP_PREFIX}_"):
        return real_collection_name[len(f"{APP_PREFIX}_"):]
    return real_collection_name

def get_doc_queue_number(collection_name: str, max_queues: int) -> int:
    """
    Get the document queue number for a given collection name.
    
    Args:
        collection_name: Collection name
        max_queues: Maximum number of queues

    Returns:
        int: 0-based queue number in range [0, max_queues-1]
    """
    # Deterministic 32-bit unsigned hash using stdlib
    hash_value = zlib.crc32(collection_name.encode("utf-8"))
    return hash_value % max_queues

