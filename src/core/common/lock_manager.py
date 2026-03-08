import asyncio
import os
import time
import uuid
from typing import List, Union
from logging import Logger

from src.core.common.bunny_talk import BunnyTalk
from src.core.common.cache import AMGIXCache
from src.core.common.constants import RPC_TIMEOUT_SECONDS


class LockAcquisitionError(Exception):
    """Failed to acquire lock"""
    pass


class LockNotFoundError(Exception):
    """Lock not found - likely manager failover occurred"""
    pass


class LockOwnershipError(Exception):
    """Attempted to release lock not owned by caller"""
    pass


class LockService:
    """
    Lock service that runs as single-active-consumer.
    Maintains in-memory lock state and handles acquire/release requests via RPC.
    AMGIXCache TTL provides automatic cleanup after FAILSAFE_TTL seconds.
    """
    
    FAILSAFE_TTL = RPC_TIMEOUT_SECONDS + 5  # Auto-expire locks after this time
    
    def __init__(self, logger: Logger, bunny_talk: BunnyTalk):
        self.logger = logger
        self.bunny_talk = bunny_talk
        # Store lock data as {"owner_id": str}
        self.locks: AMGIXCache[str, dict] = AMGIXCache(
            cache_type="ttl", name="distributed_locks", maxsize=1000000, ttl=self.FAILSAFE_TTL
        )
    
    async def startup(self):
        """Register as lock service handler"""
        await self.bunny_talk.listen(
            routing_key="lock-service",
            handler=self.handle_request,
            prefetch_count=1000,
            single_active_consumer=True
        )
        self.logger.info("LockService started")
    
    async def handle_request(self, action: str, lock_names: List[str], owner_id: str) -> bool:
        """Handle lock acquire/release RPC requests"""
        
        if action == "acquire":
            # All-or-nothing: check if ALL locks are available or already owned by us (idempotent)
            for name in lock_names:
                lock_data = self.locks.get(name)
                if lock_data is not None:
                    # Lock is held - check if it's by us
                    if lock_data["owner_id"] != owner_id:
                        # Held by someone else
                        return False
                    # else: we already own it, continue (idempotent)
            
            # Acquire (or refresh) all atomically
            for name in lock_names:
                self.locks.set(name, {"owner_id": owner_id})
            return True
        
        elif action == "refresh":
            # Verify ownership and existence of all locks first
            for name in lock_names:
                lock_data = self.locks.get(name)
                if lock_data is None:
                    raise LockNotFoundError(f"Lock '{name}' not found - likely manager failover")
                
                if lock_data["owner_id"] != owner_id:
                    raise LockOwnershipError(f"Lock '{name}' owned by {lock_data['owner_id']}, not {owner_id}")
            
            # Refresh all locks atomically (resets TTL in cache)
            for name in lock_names:
                lock_data = self.locks.get(name)
                self.locks.set(name, lock_data)
            return True
        
        elif action == "release":
            # Verify ownership for locks that still exist
            for name in lock_names:
                lock_data = self.locks.get(name)
                
                # If lock doesn't exist, it's already released (idempotent)
                if lock_data is None:
                    continue
                
                # Lock exists - verify ownership
                if lock_data["owner_id"] != owner_id:
                    raise LockOwnershipError(f"Lock '{name}' owned by {lock_data['owner_id']}, not {owner_id}")
            
            # Release all locks that still exist
            for name in lock_names:
                if self.locks.get(name) is not None:
                    del self.locks[name]
            return True
        
        else:
            raise ValueError(f"Unknown action: {action}")


class LockClient:
    """
    Client for acquiring/releasing distributed locks.
    Supports both single and multiple lock acquisition.
    Usage: async with lock_client.acquire("lock-name") as lock: ...
    """
    
    def __init__(self, logger: Logger, bunny_talk: BunnyTalk):
        self.logger = logger
        self.bunny_talk = bunny_talk
        self.owner_id = f"{os.getenv('HOSTNAME', 'unknown')}-{os.getpid()}-{uuid.uuid4()}"
    
    def acquire(self, lock_names: Union[str, List[str]], timeout: float = 5.0):
        """Return context manager for acquiring lock(s)"""
        return LockContext(self.logger, self.bunny_talk, self.owner_id, lock_names, timeout)
    
    async def try_acquire(self, lock_names: Union[str, List[str]], timeout: float = 2.0) -> bool:
        """
        Try to acquire lock(s) once without retry.
        Returns True if acquired, False if held by someone else.
        Useful for leader election patterns.
        """
        if isinstance(lock_names, str):
            lock_names = [lock_names]
        
        try:
            success = await self.bunny_talk.rpc(
                "lock-service",
                action="acquire",
                lock_names=lock_names,
                owner_id=self.owner_id,
                timeout=timeout
            )
            return success
        except Exception as e:
            self.logger.debug(f"Failed to acquire locks {lock_names}: {e}")
            return False
    
    async def release(self, lock_names: Union[str, List[str]], timeout: float = 2.0) -> bool:
        """
        Release lock(s) explicitly (without context manager).
        Returns True if successfully released.
        """
        if isinstance(lock_names, str):
            lock_names = [lock_names]
        
        try:
            await self.bunny_talk.rpc(
                "lock-service",
                action="release",
                lock_names=lock_names,
                owner_id=self.owner_id,
                timeout=timeout
            )
            return True
        except Exception as e:
            self.logger.debug(f"Failed to release locks {lock_names}: {e}")
            return False
    
    async def refresh(self, lock_names: Union[str, List[str]], timeout: float = 2.0) -> bool:
        """
        Refresh lock(s) to extend their TTL by RPC_TIMEOUT_SECONDS.
        Returns True if successfully refreshed, False if lock not held or error occurred.
        """
        if isinstance(lock_names, str):
            lock_names = [lock_names]
        
        try:
            await self.bunny_talk.rpc(
                "lock-service",
                action="refresh",
                lock_names=lock_names,
                owner_id=self.owner_id,
                timeout=timeout
            )
            return True
        except Exception as e:
            self.logger.debug(f"Failed to refresh locks {lock_names}: {e}")
            return False


class LockContext:
    """Async context manager for lock acquisition/release"""
    
    def __init__(self, logger: Logger, bunny_talk: BunnyTalk, owner_id: str, 
                 lock_names: Union[str, List[str]], timeout: float):
        self.logger = logger
        self.bunny_talk = bunny_talk
        self.owner_id = owner_id
        self.lock_names = [lock_names] if isinstance(lock_names, str) else lock_names
        self.timeout = timeout
    
    async def __aenter__(self):
        """Acquire all locks with retry until timeout"""
        
        deadline = time.time() + self.timeout
        retry_delay = 0.1
        attempt = 0
        last_error = None
        
        while time.time() < deadline:
            try:
                # Send single RPC for all locks
                success = await self.bunny_talk.rpc(
                    "lock-service",
                    action="acquire",
                    lock_names=self.lock_names,
                    owner_id=self.owner_id,
                    timeout=2.0  # RPC timeout per attempt
                )
                
                # Clear error on successful RPC (even if lock not acquired)
                last_error = None
                
                if success:
                    # All acquired successfully
                    return self
                
                # Failed to acquire (lock held by someone else), wait and retry
                
            except Exception as e:
                # RPC error (network, timeout, etc), store and retry
                last_error = e
            
            # Wait before retry
            attempt += 1
            backoff = min(attempt * retry_delay, 1.0)
            await asyncio.sleep(backoff)
        
        # Timeout expired
        if last_error:
            raise LockAcquisitionError(f"Failed to acquire locks within {self.timeout}s (last error: {last_error}): {self.lock_names}")
        else:
            raise LockAcquisitionError(f"Failed to acquire locks within {self.timeout}s: {self.lock_names}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release all acquired locks"""
        await self._release_acquired()
        return False  # Don't suppress exceptions
    
    async def _release_acquired(self):
        """Release all locks we acquired"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                await self.bunny_talk.rpc(
                    "lock-service",
                    action="release",
                    lock_names=self.lock_names,
                    owner_id=self.owner_id,
                    timeout=2.0
                )
                return  # Success
                
            except LockOwnershipError:
                # Ownership issue - this is a bug, raise immediately
                raise
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Final attempt failed - log and exit (lock will auto-expire)
                    self.logger.error(f"Failed to release locks after {max_attempts} attempts: {e}")
                    return
                # Network/RPC error, retry
                await asyncio.sleep(0.2)

