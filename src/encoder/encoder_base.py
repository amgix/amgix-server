from __future__ import annotations

from abc import abstractmethod
import asyncio
import logging
import os
from logging import Logger
import signal
from typing import List, Type, Optional

from src.core.common.embed_router import EmbedRouter
from src.core.common.bunny_talk import BunnyTalk
from src.core.common.lock_manager import LockService, LockClient
from src.core.common.logging_config import configure_logging
from src.core.database.base import DatabaseBase
from src.core.database.common import get_connected_database
from src.core.common.cache import AMGIXCache
from src.core.models.vector import VectorConfigInternal


AMGIX_DATABASE_URL = os.getenv("AMGIX_DATABASE_URL")

# Configure logging
configure_logging()


class EncoderBase:
    """Base class for encoder services with common functionality."""

    def __init__(self, logger: Logger, database: DatabaseBase, bunny_talk: BunnyTalk, router: EmbedRouter = None, lock_client: Optional["LockClient"] = None) -> None:
        self.logger = logger
        self.bunny_talk = bunny_talk
        self.database = database
        self.router = router
        self.lock_client = lock_client
    
    # Shared LRU cache for collection configs across services
    _collection_info_cache: AMGIXCache[str, VectorConfigInternal] = AMGIXCache(
        cache_type="ttl", name="collection_info", maxsize=1000, ttl=60
    )

    @staticmethod
    async def get_collection_info_cached(db: DatabaseBase, collection_name: str) -> tuple[Optional[VectorConfigInternal], bool]:
        cached_cfg = EncoderBase._collection_info_cache.get(collection_name)
        if cached_cfg is not None:
            return cached_cfg, True
        cfg = await db.get_collection_info_internal(collection_name)
        if cfg is not None:
            EncoderBase._collection_info_cache.set(collection_name, cfg)
        return cfg, False

    @staticmethod
    def invalidate_collection_cache(collection_name: str) -> None:
        """Invalidate cached collection config for a given collection name."""
        try:
            del EncoderBase._collection_info_cache[collection_name]
        except KeyError:
            pass
      
    @abstractmethod
    async def startup(self):
        pass
    
class EncoderServiceRunner:
    """Service runner that handles the encoder service lifecycle with signal handling."""
    
    def __init__(self, amqp_url: str):
        self.amqp_url = amqp_url
        self.bunny_talk = None
        self.shutdown_event = asyncio.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command
    
    async def start(self, service_classes: List[Type[EncoderBase]]):
        """Start the encoder service."""
        
        # Create BunnyTalk connection
        self.logger.info("Starting BunnyTalk...")
        self.bunny_talk = await BunnyTalk.create(self.logger, self.amqp_url)
        self.logger.info("Starting BunnyTalk... done")

        # Create and start lock service before database (database init may need locks)
        self.logger.info("Starting lock service...")
        lock_service = LockService(self.logger, self.bunny_talk)
        await lock_service.startup()
        self.logger.info("Starting lock service... done")
        
        # Create lock client for all services
        lock_client = LockClient(self.logger, self.bunny_talk)

        self.logger.info("Starting database...")
        database = await self._start_database(lock_client)
        self.logger.info("Starting database... done")

        router = None

        # Create encoder service instances
        for service_class in service_classes:
            self.logger.info(f"Starting {service_class.__name__}...")
            
            if router is None:
                # First service class is the embed router
                svc = service_class(
                    logger=self.logger, 
                    database=database, 
                    bunny_talk=self.bunny_talk,
                    router=None,
                    lock_client=lock_client
                )

                if not hasattr(svc, 'route'):
                    raise ValueError(f"First Service class must have a router. {service_class.__name__} does not have a route method")

                router = getattr(svc, 'route')
            else:
                svc = service_class(
                    logger=self.logger, 
                    database=database, 
                    bunny_talk=self.bunny_talk,
                    router=router,
                    lock_client=lock_client
                )

            await svc.startup()
            self.logger.info(f"Started {service_class.__name__}.")

    
    async def stop(self):
        """Stop the encoder service gracefully."""
        self.logger.info("Stopping encoder service...")
        
        if self.bunny_talk:
            await self.bunny_talk.close()
            self.bunny_talk = None
        
        self.logger.info("Encoder service stopped")
    
    async def run_forever(self, service_classes: List[Type[EncoderBase]]):
        """Run the service until shutdown signal is received."""
        try:
            self.logger.info("Starting server...")
            await self.start(service_classes)
            self.logger.info("Server started")
            await self.shutdown_event.wait()  # Wait for shutdown signal
            self.logger.info("Server stopped")
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            raise
        finally:
            await self.stop()


    async def _start_database(self, lock_client: LockClient) -> DatabaseBase:
        # Ensure database is available at startup and check features
        database = await get_connected_database(AMGIX_DATABASE_URL, self.logger)
        await database.check_features()

        async with lock_client.acquire("database-configure", timeout=30.0):
            await database.configure()
            return database
