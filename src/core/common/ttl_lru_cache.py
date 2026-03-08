from cachetools import LRUCache
import time

class TTLedLRUCache(LRUCache):
    def __init__(self, maxsize, ttl, *args, **kwargs):
        super().__init__(maxsize, *args, **kwargs)
        self.ttl = ttl
        self.__timestamps = {}

    def __setitem__(self, key, value, cache_setitem=LRUCache.__setitem__):
        super().__setitem__(key, value)
        self.__timestamps[key] = time.monotonic()

    def __getitem__(self, key, cache_getitem=LRUCache.__getitem__):
        # No TTL expiry on access — behave like plain LRU
        return super().__getitem__(key)

    def popitem(self):
        # On eviction, try expired items first
        now = time.monotonic()
        expired_keys = [k for k, ts in self.__timestamps.items() if now - ts >= self.ttl]
        if expired_keys:
            # Pick the oldest expired by comparing stored timestamps
            k = min(expired_keys, key=lambda key: self.__timestamps[key])
            self.__timestamps.pop(k, None)
            return k, super().pop(k)

        # Otherwise fall back to normal LRU eviction
        k, v = super().popitem()
        self.__timestamps.pop(k, None)
        return k, v

    def __delitem__(self, key):
        self.__timestamps.pop(key, None)
        super().__delitem__(key)
