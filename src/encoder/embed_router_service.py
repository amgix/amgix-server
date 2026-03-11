import ast
import asyncio
from enum import Enum
from logging import Logger
import math
import os
import psutil
import time
from collections import defaultdict, deque
from aiorwlock import RWLock

from aio_pika import RobustQueue
from src.core.common.cache import AMGIXCache
from src.core.common.constants import RPC_TIMEOUT_SECONDS
from src.core.common.embed_router import EmbedRouter
from src.core.models.vector import VectorConfigInternal
from src.core.vector import VectorBase, TrigramsVector, FullTextVector, WhiteSpaceVector, WMTRVector, DenseModelVector, SparseModelVector, CustomDenseVector, CustomSparseVector, DenseFastEmbedVector, SparseFastEmbedVector
from src.core.common.bunny_talk import BunnyTalk
from src.core.database.base import DatabaseBase
from src.core.common.enums import VectorType
from typing import Dict, List, Tuple, Type, Union, Optional
from .encoder_base import EncoderBase


class LazyVectorDict(dict):
    """Dictionary that lazily creates vector instances on first access."""
    def __init__(self, vector_classes: Dict[str, Type[VectorBase]], trusted_organizations, logger: Logger):
        super().__init__()
        self._vector_classes = vector_classes
        self._trusted_organizations = trusted_organizations
        self._logger = logger
    
    def __getitem__(self, key):
        if key not in self:
            self[key] = self._vector_classes[key](self._trusted_organizations, self._logger)
        return super().__getitem__(key)

# Try to import pynvml (from nvidia-ml-py package) for VRAM monitoring (may not be available)
try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    PYNVML_AVAILABLE = False
    # Note: Can't log here since logger doesn't exist yet at module level
    # Will be logged when first memory check happens if VRAM monitoring is requested


HOSTNAME = os.getenv('HOSTNAME', 'unknown')

NODE_QUEUE_PREFIX = "embed-node"
NODE_QUEUE_NAME = f"{NODE_QUEUE_PREFIX}-{HOSTNAME}"
OPEN_ST_QUEUE_NAME = "embed-st-open"
OPEN_FE_QUEUE_NAME = "embed-fe-open"
LEADER_QUEUE_NAME = "embed-leader"

# Metrics aggregation windows in seconds
METRIC_WINDOWS = [5, 30, 60, 300]
METRICS_LOOP_INTERVAL_SECONDS = 5
TARGET_AVAILABLITY_PCT = 5 # percentage of total capacity

class ModelFamily(Enum):
    ALL = "all"
    ST = "st"
    FE = "fe"
    NONE = "none"

AMGIX_MODEL_FAMILY = os.getenv("AMGIX_MODEL_FAMILY", "all")

# Memory reserve configuration - parse once at module load
def _parse_memory_limit(value: str) -> Optional[Tuple[float, bool]]:
    """Parse memory limit string. Returns (amount, is_percentage) or None if empty."""
    if not value:
        return None
    if value.endswith('%'):
        return (float(value[:-1]), True)
    return (float(value), False)

RAM_RESERVE_CONFIG = _parse_memory_limit(os.getenv("AMGIX_MEMORY_RESERVE_RAM", "10%"))
VRAM_RESERVE_CONFIG = _parse_memory_limit(os.getenv("AMGIX_MEMORY_RESERVE_VRAM", "15%"))

# Get total memory and GPU handle once at module load
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024 ** 3)
TOTAL_VRAM_GB = None
GPU_HANDLE = None
if PYNVML_AVAILABLE:
    try:
        GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        TOTAL_VRAM_GB = info.total / (1024 ** 3)
    except Exception:
        GPU_HANDLE = None


def _get_free_memory(logger: Logger) -> Tuple[float, Optional[float]]:
    """Get free RAM and VRAM in GB. Returns (ram_gb, vram_gb)."""
    ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    vram_gb = None
    if GPU_HANDLE is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
            vram_gb = info.free / (1024 ** 3)
        except Exception as e:
            logger.debug(f"Failed to get VRAM info: {e}")
    
    return ram_gb, vram_gb


def _check_memory_reserve(reserve_config: Optional[Tuple[float, bool]], free_gb: Optional[float], total_gb: float) -> bool:
    """Check if we're violating memory reserve. Returns True if violation."""
    if not reserve_config or free_gb is None:
        return False
    
    amount, is_percentage = reserve_config
    
    if is_percentage:
        required_free_gb = (amount / 100.0) * total_gb
    else:
        required_free_gb = amount
    
    return free_gb < required_free_gb

class EmbedRouterService(EncoderBase):
    """Router for embedding documents."""

    def __init__(self, logger: Logger, database: DatabaseBase, bunny_talk: BunnyTalk, router: EmbedRouter, lock_client=None):
        super().__init__(logger, database, bunny_talk, router, lock_client)
        
        self._vector_classes: Dict[str, Type[VectorBase]] = {
            VectorType.TRIGRAMS: TrigramsVector,
            VectorType.FULL_TEXT: FullTextVector,
            VectorType.WHITESPACE: WhiteSpaceVector,
            VectorType.WMTR: WMTRVector,
            VectorType.DENSE_MODEL: DenseModelVector,
            VectorType.SPARSE_MODEL: SparseModelVector,
            VectorType.DENSE_CUSTOM: CustomDenseVector,
            VectorType.SPARSE_CUSTOM: CustomSparseVector,
            VectorType.DENSE_FASTEMBED: DenseFastEmbedVector,
            VectorType.SPARSE_FASTEMBED: SparseFastEmbedVector,
        }

        self.local_models = {}
        self.at_capacity = False
        self.leader: Tuple[RobustQueue, str] = None
        self.node: Tuple[RobustQueue, str] = None
        self.st_open: Tuple[RobustQueue, str] = None
        self.fe_open: Tuple[RobustQueue, str] = None

        self.model_family = ModelFamily(AMGIX_MODEL_FAMILY.lower())

        self._at_model_capacity(force_check=True)

        # Per-model metrics storage
        # { model_key: deque[tuple[float, float]] }  # (timestamp, service_ms)
        self._metrics: Dict[str, deque] = {}
        self._cluster_metrics: Dict[str, Dict[Tuple[VectorType, str, str], Dict]] = {}

        self.model_locks: defaultdict[str, RWLock] = defaultdict(RWLock)
        self.model_load_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.metrics_lock = RWLock()
        
        # Cache for known model queues (tracks queues with active consumers)
        # TTL of 60s means we check at most once per minute if no requests come in
        self.known_queues: AMGIXCache[str, bool] = AMGIXCache("ttl", "known_queues", maxsize=1000, ttl=60)

    async def startup(self):
        """Startup the embed router."""

        # Load trusted organizations based on environment variable
        trusted_orgs_enabled = os.getenv('AMGIX_TRUSTED_ORGS', 'false').lower() == 'true'
        if trusted_orgs_enabled:
            self.logger.info("Loading trusted organizations...")
            self.trusted_organizations = await self._load_trusted_organizations()
            self.logger.info(f"Trusted organizations enabled, loaded {len(self.trusted_organizations)} orgs")
        else:
            self.trusted_organizations = None
            self.logger.info("Trusted organizations disabled")

        # Lazy-load vector instances on first access (stateless aside from trusted_organizations)
        self._instances: Dict[str, VectorBase] = LazyVectorDict(self._vector_classes, self.trusted_organizations, self.logger)

        self.node = await self.bunny_talk.listen(
            routing_key=NODE_QUEUE_NAME,
            handler=self._node_signal,
            auto_delete=True,
        )

        await self._listen_on_open_queues(start_listening=True)

        asyncio.create_task(self._metrics_loop())

    async def route(self, vector_config: VectorConfigInternal, docs: List[str], hops: int = 0, avgdls: Optional[List[float]] = None) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        """Embed a list of documents using the specified vector type."""

        result = None

        # measure service time
        start_ns = time.monotonic_ns()

        try:
            model_key = self._get_model_key(vector_config)

            if vector_config.type in [VectorType.DENSE_MODEL, VectorType.DENSE_FASTEMBED, VectorType.SPARSE_MODEL, VectorType.SPARSE_FASTEMBED]:

                self.bunny_talk.log_trace_context(f"Router: request for model {model_key} (hops: {hops})")
                
                async with self.metrics_lock.reader:
                    async with self.model_locks[model_key].reader:
                        # do we have this model already loaded?
                        if model_key in self.local_models:
                            # yes, just embed locally
                            self.logger.debug(f"Router: known model {model_key}. Embedding locally.")

                            result = self.embed(vector_config, docs)

                        else:
                            # get model specific queue name
                            queue_name = self._get_model_queue_name(vector_config)

                            try:
                                # Check if queue is known (has active consumers)
                                if not self.known_queues.get(queue_name):
                                    # Not in cache, check if queue has consumers before sending RPC
                                    consumer_count = await self.bunny_talk.get_queue_consumers(queue_name)
                                    self.logger.debug(f"Router: Queue {queue_name} has {consumer_count} consumers")
                                    if consumer_count == 0:
                                        # Queue exists but has no consumers (stale queue or doesn't exist)
                                        raise Exception("no_route")
                                
                                # Queue has consumers (or is in cache), send RPC
                                self.logger.debug(f"Router: Do not have {model_key}. Routing to {queue_name}.")
                                result = await self.bunny_talk.rpc(queue_name, vector_config=vector_config, docs=docs, hops=hops + 1)
                                
                                # RPC succeeded, cache the queue
                                self.known_queues.set(queue_name, True)
                            except Exception as e:
                                error_msg = str(e)
                                self.logger.debug(f"Router: Error routing to {queue_name}: {error_msg}")
                                if "no_route" in error_msg.casefold():
                                    # no route found, nobody is ready for this specific model
                                    # Invalidate cache entry so we recheck next time
                                    try:
                                        del self.known_queues[queue_name]
                                    except KeyError:
                                        pass

                                    # is it supported model and do we have space for another model?
                                    if self._is_supported_model_family(vector_config) and not self._at_model_capacity():
                                        # yes, try to load the model and embed locally
                                        result = await self._own_the_model(vector_config, model_key, queue_name, docs, hops)
                                    else:
                                        # no, either not supported model or at capacity, we need to route to other workers
                                        self.logger.debug(f"Router: Can't embed locally. Supported model family: {self.model_family}. Local models: {list(self.local_models.keys())}.")

                                        open_queue_name = OPEN_ST_QUEUE_NAME if vector_config.type in [VectorType.DENSE_MODEL, VectorType.SPARSE_MODEL] else OPEN_FE_QUEUE_NAME
                                        try:
                                            # try to send it to open queue
                                            self.logger.debug(f"Router: Routing to {open_queue_name}.")
                                            result = await self.bunny_talk.rpc(open_queue_name, vector_config=vector_config, docs=docs, hops=hops + 1)
                                        except Exception as e:
                                            error_msg = str(e)
                                            self.logger.debug(f"Router: Error routing to {open_queue_name}: {error_msg}")
                                            if "no_route" in error_msg.casefold():
                                                self.logger.debug(f"Router: No route found. Giving up.")
                                                raise Exception(f"No workers available to handle this model: {vector_config.type} {vector_config.model} {vector_config.revision}")
                                            raise e
                                            
                                if result is None:
                                    raise e
            else:
                result = self.embed(vector_config, docs, avgdls=avgdls)
                
        except Exception as e:
            self.logger.error(f"Router: Error embedding documents: {e}")
            raise e

        finally:
            end_ns = time.monotonic_ns()
            service_ms = (end_ns - start_ns) / 1_000_000.0

            # update metrics, for direct calls from vectorizer
            if hops == 0:
                self._update_metrics(model_key, service_ms)

        return result

    def _update_metrics(self, model_key: str, service_ms: float) -> None:
        """Update per-model metrics by adding new sample data."""
        try:
            now_s = time.time()
            if model_key not in self._metrics:
                self._metrics[model_key] = deque()
            self._metrics[model_key].append((now_s, service_ms))
        except Exception as e:
            self.logger.error(f"Router: Error updating metrics for {model_key}: {e}")

    def _get_metrics_snapshots(self) -> Dict[str, Dict[str, Dict[str, Union[float, int]]]]:
        """Refresh and return current metrics snapshots for all models."""
        now_s = time.time()
        processed_metrics = {}

        cutoff = now_s - max(METRIC_WINDOWS)
        keys_to_delete = []

        for model_key, samples in self._metrics.items():
            
            # Prune old data
            while samples and samples[0][0] < cutoff:
                samples.popleft()

            # Mark empty deques for deletion
            if not samples:
                keys_to_delete.append(model_key)
                continue

            # Recalculate snapshots for all windows
            snapshots = {}
            for win in METRIC_WINDOWS:
                win_cut = now_s - win
                count = 0
                total_ms = 0.0
                
                for ts, ms in reversed(samples):
                    if ts < win_cut:
                        break
                    count += 1
                    total_ms += ms
                
                rps = count / win if win > 0 else 0.0
                avg_ms = (total_ms / count) if count > 0 else 0.0
                snapshots[str(win)] = {'rps': rps, 'avg_ms': avg_ms, 'n': count}
            
            processed_metrics[model_key] = snapshots
        
        # Remove empty deques after iteration
        for key in keys_to_delete:
            del self._metrics[key]
        
        return processed_metrics

    def embed(self, vector_config: VectorConfigInternal, docs: List[str], avgdls: Optional[List[float]] = None) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        """Embed a list of documents using the specified vector type."""

        if VectorType.is_dense(vector_config.type):
            result = self._embed_dense_model(vector_config, docs)
        else:
            result = self._embed_sparse_model(vector_config, docs, avgdls=avgdls)

        return result

    def _embed_dense_model(self, vector_config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        """Embed a list of documents using the specified dense model."""

        return self._instances[vector_config.type].get_dense_vector(vector_config, docs)

    def _embed_sparse_model(self, vector_config: VectorConfigInternal, docs: List[str], avgdls: Optional[List[float]]) -> List[Tuple[List[int], List[float]]]:
        """Embed a list of documents using the specified sparse model."""

        if vector_config.type in VectorType.custom_tokenization():
            return self._instances[vector_config.type].get_sparse_vector(vector_config, docs, avgdls=avgdls)
        return self._instances[vector_config.type].get_sparse_vector(vector_config, docs)

    async def _load_trusted_organizations(self) -> set:
        """Load trusted organizations from file."""
        
        try:
            config_file = os.path.join(os.path.dirname(__file__), 'trusted_orgs.txt')
            with open(config_file, 'r') as f:
                orgs = set()
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        orgs.add(stripped)
            self.logger.info(f"Loaded {len(orgs)} trusted organizations from config")
            return orgs
        except Exception as e:
            self.logger.warning(f"Failed to load trusted organizations: {e}. Using empty set.")
            return set()

    def _get_model_key(self, vector_config: VectorConfigInternal) -> str:
        """Get the key for the model as a tuple (type, model, revision)."""

        return str((vector_config.type, vector_config.model, vector_config.revision))

    def _get_model_queue_name(self, vector_config: VectorConfigInternal) -> str:
        """Get the queue name for the model."""

        model_family = 'st' if vector_config.type in [VectorType.DENSE_MODEL, VectorType.SPARSE_MODEL] else 'fe'
        model_type = 'd' if vector_config.type in [VectorType.DENSE_MODEL, VectorType.DENSE_FASTEMBED] else 's'

        model_name = vector_config.model

        if vector_config.revision:
            model_name = f"{model_name}:{vector_config.revision[-20:]}"

        return f"embed-{model_family}-{model_type}-{model_name}"

    def _at_model_capacity(self, force_check: bool = False) -> bool:
        """Check if we can load another model based on memory reserves."""

        if force_check:
            if self.model_family == ModelFamily.NONE:
                self.at_capacity = True
            else:
                # Get current free memory (the only dynamic part)
                free_ram_gb, free_vram_gb = _get_free_memory(self.logger)
                
                # Check RAM reserve
                ram_violation = _check_memory_reserve(RAM_RESERVE_CONFIG, free_ram_gb, TOTAL_RAM_GB)
                
                # Check VRAM reserve
                vram_violation = _check_memory_reserve(VRAM_RESERVE_CONFIG, free_vram_gb, TOTAL_VRAM_GB if TOTAL_VRAM_GB else 0.0)
                
                self.at_capacity = ram_violation or vram_violation

        return self.at_capacity

    def _is_supported_model_family(self, vector_config: VectorConfigInternal) -> bool:
        """Check if the model family is supported."""

        return self.model_family == ModelFamily.ALL or \
            (self.model_family == ModelFamily.ST and vector_config.type in [VectorType.DENSE_MODEL, VectorType.SPARSE_MODEL]) or \
            (self.model_family == ModelFamily.FE and vector_config.type in [VectorType.DENSE_FASTEMBED, VectorType.SPARSE_FASTEMBED])

    async def _own_the_model(self, 
                            vector_config: VectorConfigInternal,
                            model_key: str,
                            queue_name: str,
                            docs: List[str],
                            hops: int,
                            force_load: bool = False
                            ) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        """Load and embed the model locally with distributed locking to prevent race conditions."""

        async with self.model_load_locks[model_key]:
            if model_key in self.local_models:
                self.logger.debug(f"Router: {model_key} was loaded locally while waiting. Embedding locally.")
                return self.embed(vector_config, docs)

            # Use distributed lock to prevent multiple workers from loading the same model
            async with self.lock_client.acquire(f"{queue_name}", timeout=RPC_TIMEOUT_SECONDS):
                if model_key in self.local_models:
                    self.logger.debug(f"Router: {model_key} was loaded locally while waiting for distributed lock. Embedding locally.")
                    return self.embed(vector_config, docs)

                # Double-check if someone else loaded the model while we were waiting for the lock
                try:
                    if force_load:
                        raise Exception("force_load")
                    
                    self.logger.debug(f"Router: Double-checking if {model_key} was loaded by another worker.")
                    
                    # Check if queue is known (has active consumers)
                    if not self.known_queues.get(queue_name):
                        # Not in cache, check if queue has consumers before sending RPC
                        consumer_count = await self.bunny_talk.get_queue_consumers(queue_name)
                        self.logger.debug(f"Router: Queue {queue_name} has {consumer_count} consumers")
                        if consumer_count == 0:
                            # Queue exists but has no consumers (stale queue or doesn't exist)
                            raise Exception("no_route")
                    
                    result = await self.bunny_talk.rpc(queue_name, vector_config=vector_config, docs=docs, hops=hops + 1)
                    
                    # RPC succeeded, cache the queue
                    self.known_queues.set(queue_name, True)
                    return result
                except Exception as e:
                    error_msg = str(e)
                    if error_msg == "force_load":
                        # Force load requested, skip double-check
                        self.logger.debug(f"Router: Force loading {model_key}.")
                    else:
                        self.logger.debug(f"Router: Error routing to {queue_name}: {error_msg}")
                        if "no_route" not in error_msg.casefold():
                            # Some other error, re-raise it
                            raise e
                        # Still no route, we can safely load the model
                        self.logger.debug(f"Router: Confirmed no one has {model_key}. Loading locally.")
                        
                    result = self.embed(vector_config, docs)

                    # successfully embedded, we can start listening for requests for this model
                    self.logger.debug(f"Router: Listening for requests for model {model_key}.")
                    queue, consumer_tag = await self.bunny_talk.listen(
                        routing_key=queue_name,
                        handler=self.route,
                        auto_delete=True
                    )

                    # store the queue info for this model
                    self.local_models[model_key] = (queue, consumer_tag, time.time())

                    # if we are at capacity, we need to stop listening for requests on open queue for this type
                    if self._at_model_capacity(force_check=True):
                        self.logger.debug(f"Router: At model capacity.")

                        await self._listen_on_open_queues(start_listening=False)

                    return result

    async def _listen_on_open_queues(self, start_listening: bool) -> None:
        """Listen for requests on open queues."""

        if start_listening:
            if self.model_family == ModelFamily.ST or self.model_family == ModelFamily.ALL:
                if not self.st_open:
                    self.logger.debug(f"Router: Listening for requests on {OPEN_ST_QUEUE_NAME}.")
                    try:
                        self.st_open = await self.bunny_talk.listen(
                            routing_key=OPEN_ST_QUEUE_NAME,
                                handler=self.route,
                                auto_delete=True
                        )
                    except Exception as e:
                        self.logger.error(f"Router: Error listening on {OPEN_ST_QUEUE_NAME}: {e}")
                else:
                    self.logger.debug(f"Router: Already listening for requests on {OPEN_ST_QUEUE_NAME}.")

            if self.model_family == ModelFamily.FE or self.model_family == ModelFamily.ALL:
                if not self.fe_open:
                    self.logger.debug(f"Router: Listening for requests on {OPEN_FE_QUEUE_NAME}.")
                    try:
                        self.fe_open = await self.bunny_talk.listen(
                            routing_key=OPEN_FE_QUEUE_NAME,
                            handler=self.route,
                            auto_delete=True
                        )
                    except Exception as e:
                        self.logger.error(f"Router: Error listening on {OPEN_FE_QUEUE_NAME}: {e}")
                else:
                    self.logger.debug(f"Router: Already listening for requests on {OPEN_FE_QUEUE_NAME}.")
        else:
            if self.st_open:
                # cancel the listener
                try:
                    queue, consumer_tag = self.st_open
                    self.logger.debug(f"Router: Cancelling listener on {queue.name}.")
                    await self.bunny_talk.cancel_listener(queue, consumer_tag)
                except Exception as e:
                    self.logger.error(f"Router: Error cancelling listener on {queue.name}: {e}")

                # clear the queue info
                self.st_open = None

            if self.fe_open:
                # cancel the listener
                try:
                    queue, consumer_tag = self.fe_open
                    self.logger.debug(f"Router: Cancelling listener on {queue.name}.")
                    await self.bunny_talk.cancel_listener(queue, consumer_tag)
                except Exception as e:
                    self.logger.error(f"Router: Error cancelling listener on {queue.name}: {e}")
                    
                # clear the queue info
                self.fe_open = None


    async def _node_signal(self, load: bool, model_key: str) -> None:
        """Listen for node signals."""

        try:
            self.logger.info(f"Router: Received node signal (load: {load}, model_key: {model_key})")

            type, model, revision = ast.literal_eval(model_key)

            if not model:
                self.logger.error(f"Router: Signal received but no model name in model_key: {model_key}")
                return
            
            if load:
                if self.at_capacity:
                    self.logger.info(f"Router: At capacity. Ignoring load signal for {model_key}.")
                    return
                
                config = VectorConfigInternal(name="dummy",type=type, model=model, revision=revision)
                queue_name = self._get_model_queue_name(config)

                async with self.metrics_lock.writer:
                    async with self.model_locks[model_key].writer:
                        await self._own_the_model(config, model_key, queue_name, ["x"], 0, force_load=True)

                self.logger.info(f"Router: Loaded model {model_key}.")
            else:
                # Get the queue info and cancel consumer
                if model_key not in self.local_models:
                    self.logger.warning(f"Router: Model {model_key} not found in local_models. Skipping.")
                    return

                async with self.metrics_lock.writer:
                    async with self.model_locks[model_key].writer:
                        queue, consumer_tag, _ = self.local_models[model_key]
                        
                        # Cancel the consumer to stop listening for requests
                        self.logger.debug(f"Router: Cancelling consumer for {model_key}")
                        await self.bunny_talk.cancel_listener(queue, consumer_tag)

                        # Invalidate known_queues cache for this queue
                        config = VectorConfigInternal(name="dummy", type=type, model=model, revision=revision)
                        queue_name = self._get_model_queue_name(config)
                        try:
                            del self.known_queues[queue_name]
                        except KeyError:
                            self.logger.debug(f"Router: known_queues had no entry for {queue_name} during unload.")

                        # Unload the model from vector class cache
                        self._instances[type].unload_model(model, revision)

                        # Remove from local_models dict
                        self.local_models.pop(model_key, None)

                        # if we are not at capacity, we need to start listening for requests on open queues
                        if not self._at_model_capacity(force_check=True):
                            await self._listen_on_open_queues(start_listening=True)

                self.logger.info(f"Router: unloaded model {model_key}.")
                
        except Exception as e:
            self.logger.error(f"Router: Error handling node signal (load: {load}, model_key: {model_key}): {e}")

    async def _metrics_signal(self, 
                probe: bool,
                hostname: str,  
                metrics: Dict[str, Dict[str, Dict[str, Union[float, int]]]],
                loaded_models: List[Tuple[str, float]],
                model_family: ModelFamily,
                at_capacity: bool
                ) -> None:
        """Accumulate metrics from workers."""

        if probe:
            return

        # Store metrics per hostname with timestamp
        # metrics now contains only the snapshots data
        self._cluster_metrics[hostname] = {
            'metrics': {ast.literal_eval(k): v for k, v in metrics.items()},
            'loaded_models': [
                (ast.literal_eval(model_key), load_timestamp)
                for model_key, load_timestamp in loaded_models
            ],
            'model_family': ModelFamily(model_family),
            'capacity': 0 if at_capacity else 1 if len(loaded_models) > 0 else 2,
            'last_seen': time.time()
        }
        
        # Log summary of what we received
        self.logger.debug(f"Router: Received metrics from {hostname} ({model_family}): metrics: {len(metrics)}, models: {len(loaded_models)}, capacity: {self._cluster_metrics[hostname]['capacity']}")
        self.logger.debug(f"Router:     models: {[model_key for model_key, _ in loaded_models]}")

    def _cleanup_stale_metrics(self) -> None:
        """Remove metrics from hosts that haven't reported in a while."""
        cutoff_time = time.time() - (METRICS_LOOP_INTERVAL_SECONDS * 2)
        
        stale_hosts = [
            hostname for hostname, data in self._cluster_metrics.items()
            if data['last_seen'] < cutoff_time
        ]
        
        for hostname in stale_hosts:
            del self._cluster_metrics[hostname]
            self.logger.info(f"Router: Expired metrics for {hostname} (stale)")

    async def _metrics_loop(self) -> None:
        """Send metrics to leader."""

        while True:        
            # 1. Do a play for leader
            try:

                try_leader = True

                try:
                    # try to send a probe to the leader (even if it's us)
                    await self.bunny_talk.talk(
                        routing_key=LEADER_QUEUE_NAME,
                        probe=True,
                        hostname=HOSTNAME,
                        metrics=dict(),
                        loaded_models=list(),
                        model_family=self.model_family.value,
                        at_capacity=False,
                        start_trace=True
                    )
                    try_leader = False
                except Exception as e:
                    # if it fails, we have no leader
                    pass

                if try_leader:
                    was_leader = self.leader is not None

                    self.leader = await self.bunny_talk.listen(
                        routing_key=LEADER_QUEUE_NAME,
                        handler=self._metrics_signal,
                        auto_delete=True,
                        exclusive=True,
                        robust=False
                    )

                    if not was_leader:
                        self.logger.info("Router: I'm the leader.")

            except Exception as e:
                self.leader = None
                self.logger.debug(f"Router: Not the leader: {e}")

            # 2. Send metrics to leader
            try:
                # Get fresh snapshots for all models
                processed_metrics = self._get_metrics_snapshots()

                async with self.metrics_lock.reader:
                    loaded_models = [
                        (model_key, load_timestamp)
                        for model_key, (queue, consumer_tag, load_timestamp) in self.local_models.items()
                    ]

                if self.leader:
                    # no need for round trip, just call our own handler
                    await self._metrics_signal(False,HOSTNAME, processed_metrics, 
                                    loaded_models, self.model_family.value, self._at_model_capacity())
                else:
                    # send it over the pipe
                    await self.bunny_talk.talk(
                        routing_key=LEADER_QUEUE_NAME,
                        probe=False,
                        hostname=HOSTNAME,
                        metrics=processed_metrics,
                        loaded_models=loaded_models,
                        model_family=self.model_family.value,
                        at_capacity=self._at_model_capacity(),
                        start_trace=True
                    )
            except Exception as e:
                self.logger.warning(f"Router: Error sending metrics to leader: {e}")

            # 3. rebalance models
            if self.leader:
                await self._rebalance_models()

            # 4. Sleep
            await asyncio.sleep(METRICS_LOOP_INTERVAL_SECONDS)

    async def _rebalance_models(self) -> None:
        """Rebalance models."""

        start_ns = time.monotonic_ns()
        self.logger.debug(f"Router: Rebalancing models --------------------------------")

        async def _send_signal(hostname: str, model_key: Tuple[VectorType, str, str], load: bool) -> None:
            node_queue_name = f"{NODE_QUEUE_PREFIX}-{hostname}"
            try:
                self.logger.debug(f"Router: Sending signal to {hostname} for {model_key}: load={load}")
                await self.bunny_talk.talk(node_queue_name, load=load, model_key=str(model_key), start_trace=True)
            except Exception as e:
                self.logger.warning(f"Router: Failed to send signal to {hostname} for {model_key} (load={load}): {e}")

        # Clean up stale metrics (older than 2x the metrics interval)
        self._cleanup_stale_metrics()

        # find hosts that per model family
        fe_host_count = 0
        st_host_count = 0
        fe_available_count = 0
        st_available_count = 0
        available_fe_hosts = dict()
        available_st_hosts = dict()

        for hostname, data in self._cluster_metrics.items():
            if data['model_family'] in [ModelFamily.FE, ModelFamily.ALL]:
                fe_host_count += 1
                if data['capacity'] > 0:
                    available_fe_hosts[hostname] = data['capacity']
                    fe_available_count += 1
            if data['model_family'] in [ModelFamily.ST, ModelFamily.ALL]:
                st_host_count += 1
                if data['capacity'] > 0:
                    available_st_hosts[hostname] = data['capacity']
                    st_available_count += 1

        fe_reservations = max(1, math.floor(TARGET_AVAILABLITY_PCT * fe_host_count / 100))
        fe_direction = fe_available_count - fe_reservations

        st_reservations = max(1, math.floor(TARGET_AVAILABLITY_PCT * st_host_count / 100))
        st_direction = st_available_count - st_reservations

        # Get all models that are loaded
        model_list = set(model_tuple[0] for host_data in self._cluster_metrics.values() 
                        for model_tuple in host_data['loaded_models'] if host_data['loaded_models'])

        max_window = max(METRIC_WINDOWS)
        total_rps = 0.0
        
        models = {}
        for model_key in model_list:
            models[model_key] = {
                'hosts': sorted([(k, v['capacity'], model_tuple[1]) for k, v in self._cluster_metrics.items() 
                               for model_tuple in v['loaded_models']
                               if model_tuple[0] == model_key], key=lambda x: x[1]),
                'host_count': 0,
                'weighted_rps': 0.0,
                'proportion': 0.0,
                'model_family': ModelFamily.FE if model_key[0] in [VectorType.DENSE_FASTEMBED, VectorType.SPARSE_FASTEMBED] else ModelFamily.ST
            }

            for host in self._cluster_metrics.keys():
                metrics = self._cluster_metrics[host]['metrics'].get(model_key, {})

                # Weighted RPS across all windows (shorter windows get higher weight)
                weighted_rps = 0.0
                for window in METRIC_WINDOWS:
                    window_rps = metrics.get(str(window), {}).get('rps', 0.0)
                    weight = max_window / window  # 600/5, 600/30, etc.
                    weighted_rps += window_rps * weight
                
                models[model_key]['weighted_rps'] += weighted_rps
                total_rps += weighted_rps

        # Track best candidates for each family
        best_add_fe = None
        best_score_fe = -1
        second_best_score_fe = -1
        best_remove_fe = None
        best_remove_score_fe = float('inf')
        second_best_remove_score_fe = float('inf')
        
        best_add_st = None
        best_score_st = -1
        second_best_score_st = -1
        best_remove_st = None
        best_remove_score_st = float('inf')
        second_best_remove_score_st = float('inf')

        for model_key, data in models.items():
            # host_count = fe_host_count if data['model_family'] == ModelFamily.FE else st_host_count
            data['host_count'] = len(data['hosts'])
            data['proportion'] = data['weighted_rps'] / total_rps if total_rps > 0 else 0.0
            data['score'] = data['proportion'] / (data['host_count'] + 1)

            # Compute how many additional hosts could actually load this model
            # Hosts that do NOT already have this model
            if data['model_family'] == ModelFamily.FE:
                data['target_hosts'] = [(host, capacity) for host, capacity in available_fe_hosts.items() if host not in (h for h, _, _ in data['hosts'])]
                
                # Track best add candidate for FE
                if (data['weighted_rps'] > 0 and data['target_hosts'] and data['score'] > best_score_fe):
                    second_best_score_fe = best_score_fe
                    best_score_fe = data['score']
                    best_add_fe = (model_key, data)
                elif (data['weighted_rps'] > 0 and data['target_hosts'] and data['score'] > second_best_score_fe):
                    second_best_score_fe = data['score']
                
                # Track best remove candidate for FE
                if (data['weighted_rps'] > 0 and data['host_count'] > 1):
                    remove_score = data['proportion'] / data['host_count']
                    if remove_score < best_remove_score_fe:
                        second_best_remove_score_fe = best_remove_score_fe
                        best_remove_score_fe = remove_score
                        best_remove_fe = (model_key, data)
                    elif remove_score < second_best_remove_score_fe:
                        second_best_remove_score_fe = remove_score
            else:
                data['target_hosts'] = [(host, capacity) for host, capacity in available_st_hosts.items() if host not in (h for h, _, _ in data['hosts'])]
                
                # Track best add candidate for ST
                if (data['weighted_rps'] > 0 and data['target_hosts'] and data['score'] > best_score_st):
                    second_best_score_st = best_score_st
                    best_score_st = data['score']
                    best_add_st = (model_key, data)
                elif (data['weighted_rps'] > 0 and data['target_hosts'] and data['score'] > second_best_score_st):
                    second_best_score_st = data['score']
                
                # Track best remove candidate for ST
                if (data['weighted_rps'] > 0 and data['host_count'] > 1):
                    remove_score = data['proportion'] / data['host_count']
                    if remove_score < best_remove_score_st:
                        second_best_remove_score_st = best_remove_score_st
                        best_remove_score_st = remove_score
                        best_remove_st = (model_key, data)
                    elif remove_score < second_best_remove_score_st:
                        second_best_remove_score_st = remove_score

        # unload models that have no rps
        for model_key, data in [(k, v) for k, v in models.items() if v['weighted_rps'] == 0]:
            if data['host_count'] > 1:
                hostname = data['hosts'][0][0]
                self.logger.debug(f"Router: Unloading model {model_key} from {hostname} because it has no rps.")
                await _send_signal(hostname, model_key, False)
            elif data['host_count'] == 1:
                current_time = time.time()
                hostname, _, load_timestamp = data['hosts'][0]
                if current_time - load_timestamp > max_window:
                    self.logger.debug(f"Router: Unloading model {model_key} from {hostname} because it was loaded {int(current_time - load_timestamp)} seconds ago and has no rps.")
                    await _send_signal(hostname, model_key, False)

        # load/uload model that have rps per family per direction.
        
        # Short-circuit demand-based rebalancing on very small clusters (<=2 capable hosts)
        capable_hosts_count = sum(1 for _, d in self._cluster_metrics.items() if d['model_family'] != ModelFamily.NONE)
        if capable_hosts_count < 3:
            self.logger.debug(f"Router: Skipping demand-based rebalancing: only {capable_hosts_count} capable host(s).")
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
            self.logger.debug(f"Router: Done rebalancing models in {elapsed_ms:.3f}ms ---------- ")
            return

        # Swap step when family is full (direction == 0): move one slot from lowest p/h to highest p/(h+1)
        if fe_direction == 0:
            swap_add_fe = None
            swap_add_score_fe = -1
            swap_remove_fe = None
            swap_remove_score_fe = float('inf')

            for m_key, m_data in models.items():
                if m_data['model_family'] != ModelFamily.FE or m_data['weighted_rps'] == 0:
                    continue
                if m_data['score'] > swap_add_score_fe:
                    swap_add_score_fe = m_data['score']
                    swap_add_fe = (m_key, m_data)
                if m_data['host_count'] > 1:
                    rm_score = (m_data['proportion'] / m_data['host_count'])
                    if rm_score < swap_remove_score_fe:
                        swap_remove_score_fe = rm_score
                        swap_remove_fe = (m_key, m_data)

            if swap_add_fe and swap_remove_fe and swap_add_fe[0] != swap_remove_fe[0]:
                add_key, add_data = swap_add_fe
                remove_key, remove_data = swap_remove_fe
                add_hosts = set(h for h, _, _ in add_data['hosts'])
                remove_host_tuple = next(((h, cap, ts) for h, cap, ts in remove_data['hosts'] if h not in add_hosts), None)
                if remove_host_tuple:
                    hostname, _, load_ts = remove_host_tuple
                    current_time = time.time()
                    if not load_ts or current_time - load_ts >= METRICS_LOOP_INTERVAL_SECONDS * 2:
                        self.logger.debug(f"Router: FE swap: unloading {remove_key} from {hostname} to make room for {add_key}")
                        await _send_signal(hostname, remove_key, False)

        if st_direction == 0:
            swap_add_st = None
            swap_add_score_st = -1
            swap_remove_st = None
            swap_remove_score_st = float('inf')

            for m_key, m_data in models.items():
                if m_data['model_family'] != ModelFamily.ST or m_data['weighted_rps'] == 0:
                    continue
                if m_data['score'] > swap_add_score_st:
                    swap_add_score_st = m_data['score']
                    swap_add_st = (m_key, m_data)
                if m_data['host_count'] > 1:
                    rm_score = (m_data['proportion'] / m_data['host_count'])
                    if rm_score < swap_remove_score_st:
                        swap_remove_score_st = rm_score
                        swap_remove_st = (m_key, m_data)

            if swap_add_st and swap_remove_st and swap_add_st[0] != swap_remove_st[0]:
                add_key, add_data = swap_add_st
                remove_key, remove_data = swap_remove_st
                add_hosts = set(h for h, _, _ in add_data['hosts'])
                remove_host_tuple = next(((h, cap, ts) for h, cap, ts in remove_data['hosts'] if h not in add_hosts), None)
                if remove_host_tuple:
                    hostname, _, load_ts = remove_host_tuple
                    current_time = time.time()
                    if not load_ts or current_time - load_ts >= METRICS_LOOP_INTERVAL_SECONDS * 2:
                        self.logger.debug(f"Router: ST swap: unloading {remove_key} from {hostname} to make room for {add_key}")
                        await _send_signal(hostname, remove_key, False)

        # Handle FE family
        if fe_direction > 0 and best_add_fe:
            model_key, data = best_add_fe
            # Pick best host (highest capacity)
            hostname = max(data['target_hosts'], key=lambda x: x[1])[0]
            self.logger.debug(f"Router: Adding {model_key} to {hostname} (score: {best_score_fe:.3f})")
            await _send_signal(hostname, model_key, True)
        elif fe_direction < 0 and best_remove_fe:
            model_key, data = best_remove_fe
            # Pick worst host (lowest capacity)
            hostname = min(data['hosts'], key=lambda x: x[1])[0]
            
            # Dwell-time hysteresis: don't unload if recently loaded
            current_time = time.time()
            _, _, load_timestamp = next((h for h in data['hosts'] if h[0] == hostname), (None, None, None))
            if load_timestamp and current_time - load_timestamp < METRICS_LOOP_INTERVAL_SECONDS * 2:
                self.logger.debug(f"Router: Skipping removal of {model_key} from {hostname} (too recent: {int(current_time - load_timestamp)}s)")
            else:
                self.logger.debug(f"Router: Removing {model_key} from {hostname} (score: {best_remove_score_fe:.3f})")
                await _send_signal(hostname, model_key, False)

        # Handle ST family
        if st_direction > 0 and best_add_st:
            model_key, data = best_add_st
            # Pick best host (highest capacity)
            hostname = max(data['target_hosts'], key=lambda x: x[1])[0]
            self.logger.debug(f"Router: Adding {model_key} to {hostname} (score: {best_score_st:.3f})")
            await _send_signal(hostname, model_key, True)
        elif st_direction < 0 and best_remove_st:
            model_key, data = best_remove_st
            # Pick worst host (lowest capacity)
            hostname = min(data['hosts'], key=lambda x: x[1])[0]
            
            # Dwell-time hysteresis: don't unload if recently loaded
            current_time = time.time()
            _, _, load_timestamp = next((h for h in data['hosts'] if h[0] == hostname), (None, None, None))
            if load_timestamp and current_time - load_timestamp < METRICS_LOOP_INTERVAL_SECONDS * 2:
                self.logger.debug(f"Router: Skipping removal of {model_key} from {hostname} (too recent: {int(current_time - load_timestamp)}s)")
            else:
                self.logger.debug(f"Router: Removing {model_key} from {hostname} (score: {best_remove_score_st:.3f})")
                await _send_signal(hostname, model_key, False)

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        self.logger.debug(f"Router: Done rebalancing models in {elapsed_ms:.3f}ms ---------- ")
