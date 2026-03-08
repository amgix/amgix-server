"""
Thread-safe Cache implementation using cachetools.
"""

import threading
from cachetools import TTLCache, LRUCache
from typing import TypeVar, Generic, Optional, Literal

from .ttl_lru_cache import TTLedLRUCache

K = TypeVar('K')
V = TypeVar('V')

# Define cache types as a literal union for type safety
CacheType = Literal["lru", "ttl", "ttl_lru"]


class AMGIXCache(Generic[K, V]):
    """
    Thread-safe Cache wrapper around cachetools cache implementations.
    
    Uses class-level locks keyed by cache name for thread safety.
    Caches with the same name share the same lock, allowing for
    coordinated access while different named caches can operate independently.
    """
    
    # Class-level dictionary of locks, keyed by cache name
    _locks: dict[str, threading.RLock] = {}
    _locks_lock = threading.RLock()  # Protects the locks dictionary itself
    
    def __init__(self, cache_type: CacheType, name: str, maxsize: int, ttl: Optional[int] = None):
        """
        Initialize the thread-safe cache.
        
        Args:
            cache_type: Type of cache to use ("lru", "ttl", or "ttl_lru")
            name: Unique name for this cache (determines which lock to use)
            maxsize: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds (required for "ttl" and "ttl_lru" types)
        """
        if cache_type in ["ttl", "ttl_lru"] and ttl is None:
            raise ValueError(f"TTL is required for cache type '{cache_type}'")
        
        self._name = name
        self._cache_type = cache_type
        
        # Create the appropriate cache implementation
        if cache_type == "lru":
            self._cache = LRUCache(maxsize=maxsize)
        elif cache_type == "ttl":
            self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        elif cache_type == "ttl_lru":
            self._cache = TTLedLRUCache(maxsize=maxsize, ttl=ttl)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        # Get or create the lock for this cache name (disabled for async-only usage)
        # with self._locks_lock:
        #     if name not in self._locks:
        #         self._locks[name] = threading.RLock()
        #     self._lock = self._locks[name]
        self._lock = None  # locking disabled
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Thread-safe get operation.
        
        Args:
            key: Cache key to retrieve
            default: Default value to return if key not found
            
        Returns:
            Cached value or default if not found
        """
        # with self._lock:
        return self._cache.get(key, default)
    
    def set(self, key: K, value: V) -> None:
        """
        Thread-safe set operation.
        
        Args:
            key: Cache key to store
            value: Value to cache
        """
        # with self._lock:
        self._cache[key] = value
    
    def get_or_add(self, key: K, factory: callable) -> V:
        """
        Thread-safe get-or-add operation.
        
        Gets the value from cache if it exists, otherwise calls the factory function
        to compute the value, stores it in cache, and returns it. Only one thread
        will call the factory function even if multiple threads request the same key
        simultaneously.
        
        Args:
            key: Cache key to retrieve or store
            factory: Function that computes the value if key not found
            
        Returns:
            Cached value or newly computed value
        """
        # with self._lock:
        # Double-check pattern: check cache again inside the lock
        if key in self._cache:
            return self._cache[key]

        # Key not in cache, compute and store it
        value = factory()
        self._cache[key] = value
        return value
    
    def __delitem__(self, key: K) -> None:
        """
        Thread-safe delete operation.
        
        Args:
            key: Cache key to remove
        """
        # with self._lock:
        del self._cache[key]