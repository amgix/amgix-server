import asyncio
from logging import Logger
import os
import psutil
import time
from collections import defaultdict
from aiorwlock import RWLock

from src.core.common.cache import AMGIXCache
from src.core.common.constants import RPC_TIMEOUT_SECONDS, WMTR_DEFAULT_TRIGRAM_WEIGHT
from src.core.common.embed_router import EmbedRouter
from src.core.common.metrics_definitions import MetricKey
from src.core.common.metrics_service import MetricsService
from src.core.models.vector import VectorConfigInternal
from src.core.vector import VectorBase, TrigramsVector, FullTextVector, WhiteSpaceVector, WMTRVector, DenseModelVector, SparseModelVector, CustomDenseVector, CustomSparseVector
from src.core.common.bunny_talk import BunnyTalk
from src.core.database.base import DatabaseBase
from src.core.common.enums import VectorType
from typing import Any, Dict, List, Tuple, Type, Union, Optional
from .encoder_base import EncoderBase
from .model_rebalancer import ModelRebalancer


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
AMGIX_AMQP_URL = os.getenv("AMGIX_AMQP_URL", "pyamqp://guest:guest@rabbitmq//")

NODE_QUEUE_PREFIX = "embed-node"
NODE_QUEUE_NAME = f"{NODE_QUEUE_PREFIX}-{HOSTNAME}"
OPEN_EMBED_QUEUE_NAME = "embed-open"
# Metrics aggregation windows in seconds
METRIC_WINDOWS = [10, 30, 60]
# Subset of METRIC_WINDOWS exposed in the cluster view API response
CLUSTER_VIEW_WINDOWS = {30, 60}
# How long a loaded model with zero traffic is kept before being unloaded
MODEL_IDLE_GRACE_SECONDS = 300
NODE_META_MODEL_LAST_USED = "model_last_used"
NODE_META_LAST_USED_AT = "last_used_at"
METRICS_LOOP_INTERVAL_SECONDS = 5
METRICS_NODE_EXPIRY_SECONDS = 30
TARGET_AVAILABLITY_PCT = 5 # percentage of total capacity

def _parse_load_models(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")

AMGIX_LOAD_MODELS = _parse_load_models(os.getenv("AMGIX_LOAD_MODELS", "true"))
AMGIX_ENCODER_ROLE = os.getenv("AMGIX_ENCODER_ROLE", "all")

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


def _loaded_model_label(mk_list: List[str]) -> Optional[str]:
    if len(mk_list) < 3 or not mk_list[1]:
        return None
    model_name, revision = mk_list[1], mk_list[2]
    return f"{model_name}:{revision}" if revision else model_name


def _serialize_loaded_models_meta(loaded_models: List[Tuple[List[str], float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mk_list, load_timestamp in loaded_models:
        entry: Dict[str, Any] = {
            "model_key": list(mk_list),
            "loaded_at": float(load_timestamp),
        }
        label = _loaded_model_label(mk_list)
        if label is not None:
            entry["label"] = label
        out.append(entry)
    out.sort(key=lambda entry: str(entry.get("label", "")))
    return out


def _serialize_model_last_used_meta(last_used: List[Tuple[Tuple[str, ...], float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model_key, last_used_at in last_used:
        out.append(
            {
                "model_key": list(model_key),
                NODE_META_LAST_USED_AT: float(last_used_at),
            }
        )
    out.sort(key=lambda entry: tuple(entry["model_key"]))
    return out

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
        }

        self.local_models = {}
        self.at_capacity = False
        self.node = None
        self.embed_open = None

        self.load_models = AMGIX_LOAD_MODELS

        self._free_ram_gb: float = 0.0
        self._free_vram_gb: Optional[float] = None

        self._at_model_capacity(force_check=True)

        self.metrics = MetricsService(
            amqp_url=AMGIX_AMQP_URL,
            logger=self.logger,
            hostname=HOSTNAME,
            source="router",
            role=AMGIX_ENCODER_ROLE,
            windows=METRIC_WINDOWS,
            database=database,
            report_interval_s=METRICS_LOOP_INTERVAL_SECONDS,
            leader_loop_interval_s=METRICS_LOOP_INTERVAL_SECONDS,
            cluster_view_windows=CLUSTER_VIEW_WINDOWS,
            metrics_node_expiry_seconds=METRICS_NODE_EXPIRY_SECONDS,
            last_used_ttl_seconds=MODEL_IDLE_GRACE_SECONDS,
        )
        self.model_rebalancer = ModelRebalancer(
            bunny_talk=self.bunny_talk,
            logger=self.logger,
            windows=METRIC_WINDOWS,
            leader_loop_interval_s=METRICS_LOOP_INTERVAL_SECONDS,
            model_idle_grace_seconds=MODEL_IDLE_GRACE_SECONDS,
            target_availability_pct=TARGET_AVAILABLITY_PCT,
            node_queue_prefix=NODE_QUEUE_PREFIX,
        )
        self._rebalance_task: Optional[asyncio.Task] = None
        self._metrics_meta_task: Optional[asyncio.Task] = None

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

        self.metrics.publish_meta(await self._metrics_meta())
        self.metrics.start_reporting()
        self.metrics.start_leader_loop()
        self._rebalance_task = asyncio.create_task(self._rebalance_loop())
        self._metrics_meta_task = asyncio.create_task(self._metrics_meta_loop())

    async def route(
        self,
        vector_config: VectorConfigInternal,
        docs: List[str],
        hops: int = 0,
        avgdls: Optional[List[float]] = None,
        trigram_weight: float = WMTR_DEFAULT_TRIGRAM_WEIGHT,
    ) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        """Embed a list of documents using the specified vector type."""

        result = None

        # measure service time
        start_ns = time.monotonic_ns()

        try:
            model_key = self._get_model_key(vector_config)

            if vector_config.type in [VectorType.DENSE_MODEL, VectorType.SPARSE_MODEL]:

                self.bunny_talk.log_trace_context(f"Router: request for model {model_key} (hops: {hops})")
                
                async with self.metrics_lock.reader:
                    async with self.model_locks[model_key].reader:
                        # do we have this model already loaded?
                        if model_key in self.local_models:
                            # yes, just embed locally
                            self.logger.debug(f"Router: known model {model_key}. Embedding locally.")

                            result = self.embed(model_key, vector_config, docs, hops)

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

                                    # Load on this node only when AMGIX_LOAD_MODELS is on and memory reserve is not exceeded.
                                    if self.load_models and not self._at_model_capacity():
                                        # Try to acquire the model locally and embed.
                                        result = await self._own_the_model(vector_config, model_key, queue_name, docs, hops)
                                    else:
                                        # If load_models is off, skip logging (by design). If on, we're only here due to capacity.
                                        if self.load_models:
                                            self.logger.debug(
                                                "Router: Not loading model locally (memory reserve / capacity). model_key=%s local_models=%s",
                                                model_key,
                                                list(self.local_models.keys()),
                                            )

                                        open_queue_name = OPEN_EMBED_QUEUE_NAME
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
                result = self.embed(
                    model_key,
                    vector_config,
                    docs,
                    hops,
                    avgdls=avgdls,
                    trigram_weight=trigram_weight,
                )
                
        except Exception as e:
            self.logger.error(f"Router: Error embedding documents: {e}")
            raise e

        finally:
            end_ns = time.monotonic_ns()
            e2e_ms = (end_ns - start_ns) / 1_000_000.0

            if hops == 0:
                n_docs = len(docs)
                self.metrics.record(MetricKey.EMBED_BATCHES_ORIGIN, dims=model_key)
                self.metrics.record(MetricKey.EMBED_PASSAGES_ORIGIN, n_docs, dims=model_key)
                self.metrics.record(MetricKey.EMBED_INFERENCE_ORIGIN_MS, e2e_ms, dims=model_key, n=1)
                if result is None:
                    self.metrics.record(MetricKey.EMBED_INFERENCE_ORIGIN_ERRORS, dims=model_key)

        return result

    def embed(
        self,
        model_key: tuple[str, str, str],
        vector_config: VectorConfigInternal,
        docs: List[str],
        hops: int,
        avgdls: Optional[List[float]] = None,
        trigram_weight: float = WMTR_DEFAULT_TRIGRAM_WEIGHT,
    ) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        """Embed a list of documents using the specified vector type."""

        start_ns = time.monotonic_ns()

        if VectorType.is_dense(vector_config.type):
            result = self._embed_dense_model(vector_config, docs)
        else:
            result = self._embed_sparse_model(
                vector_config,
                docs,
                avgdls=avgdls,
                trigram_weight=trigram_weight,
            )

        n_docs = len(docs)
        inference_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
        self.metrics.mark_last_used(model_key)
        self.metrics.record(MetricKey.EMBED_BATCHES, dims=model_key)
        self.metrics.record(MetricKey.EMBED_PASSAGES, n_docs, dims=model_key)
        self.metrics.record(MetricKey.EMBED_INFERENCE_MS, inference_ms, dims=model_key, n=1)
        self.metrics.record(MetricKey.EMBED_HOPS, hops, dims=model_key, n=1)

        return result

    def _embed_dense_model(self, vector_config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        """Embed a list of documents using the specified dense model."""

        return self._instances[vector_config.type].get_dense_vector(vector_config, docs)

    def _embed_sparse_model(
        self,
        vector_config: VectorConfigInternal,
        docs: List[str],
        avgdls: Optional[List[float]],
        trigram_weight: float,
    ) -> List[Tuple[List[int], List[float]]]:
        """Embed a list of documents using the specified sparse model."""

        if vector_config.type in VectorType.custom_tokenization():
            return self._instances[vector_config.type].get_sparse_vector(
                vector_config,
                docs,
                avgdls=avgdls,
                trigram_weight=trigram_weight,
            )
        return self._instances[vector_config.type].get_sparse_vector(
            vector_config,
            docs,
            trigram_weight=trigram_weight,
        )

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

    def _get_model_key(self, vector_config: VectorConfigInternal) -> tuple[str, str, str]:
        return (vector_config.type, vector_config.model or "", vector_config.revision or "")

    def _get_model_queue_name(self, vector_config: VectorConfigInternal) -> str:
        """Get the queue name for the model."""

        # RPC queue for this model: embed-<d|s>-<name> (d=dense, s=sparse).
        model_type = 'd' if vector_config.type == VectorType.DENSE_MODEL else 's'

        model_name = vector_config.model

        if vector_config.revision:
            model_name = f"{model_name}:{vector_config.revision[-20:]}"

        return f"embed-{model_type}-{model_name}"

    def _at_model_capacity(self, force_check: bool = False) -> bool:
        """Check if we can load another model based on memory reserves."""

        if force_check:
            if not self.load_models:
                self.at_capacity = True
            else:
                # Get current free memory (the only dynamic part)
                free_ram_gb, free_vram_gb = _get_free_memory(self.logger)
                self._free_ram_gb = free_ram_gb
                self._free_vram_gb = free_vram_gb

                # Check RAM reserve
                ram_violation = _check_memory_reserve(RAM_RESERVE_CONFIG, free_ram_gb, TOTAL_RAM_GB)
                
                # Check VRAM reserve
                vram_violation = _check_memory_reserve(VRAM_RESERVE_CONFIG, free_vram_gb, TOTAL_VRAM_GB if TOTAL_VRAM_GB else 0.0)
                
                self.at_capacity = ram_violation or vram_violation

        return self.at_capacity

    async def _own_the_model(self,
                            vector_config: VectorConfigInternal,
                            model_key: tuple[str, str, str],
                            queue_name: str,
                            docs: List[str],
                            hops: int,
                            force_load: bool = False
                            ) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        """Load and embed the model locally with distributed locking to prevent race conditions."""

        async with self.model_load_locks[model_key]:
            if model_key in self.local_models:
                self.logger.debug(f"Router: {model_key} was loaded locally while waiting. Embedding locally.")
                return self.embed(model_key, vector_config, docs, hops)

            # Use distributed lock to prevent multiple workers from loading the same model
            async with self.lock_client.acquire(f"{queue_name}", timeout=RPC_TIMEOUT_SECONDS):
                if model_key in self.local_models:
                    self.logger.debug(f"Router: {model_key} was loaded locally while waiting for distributed lock. Embedding locally.")
                    return self.embed(model_key, vector_config, docs, hops)

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
                        
                    result = self.embed(model_key, vector_config, docs, hops)

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
            if self.load_models:
                if not self.embed_open:
                    self.logger.debug(f"Router: Listening for requests on {OPEN_EMBED_QUEUE_NAME}.")
                    try:
                        self.embed_open = await self.bunny_talk.listen(
                            routing_key=OPEN_EMBED_QUEUE_NAME,
                            handler=self.route,
                            auto_delete=True,
                        )
                    except Exception as e:
                        self.logger.error(f"Router: Error listening on {OPEN_EMBED_QUEUE_NAME}: {e}")
                else:
                    self.logger.debug(f"Router: Already listening for requests on {OPEN_EMBED_QUEUE_NAME}.")
        else:
            if self.embed_open:
                # cancel the listener
                try:
                    queue, consumer_tag = self.embed_open
                    self.logger.debug(f"Router: Cancelling listener on {queue.name}.")
                    await self.bunny_talk.cancel_listener(queue, consumer_tag)
                except Exception as e:
                    self.logger.error(f"Router: Error cancelling listener on {queue.name}: {e}")

                # clear the queue info
                self.embed_open = None

    async def _node_signal(self, load: bool, model_key: List[str]) -> None:
        """Listen for node signals."""

        try:
            self.logger.info(f"Router: Received node signal (load: {load}, model_key: {model_key})")

            type_str, model, revision_str = model_key
            mk = (type_str, model, revision_str)
            revision = revision_str or None

            if not model:
                self.logger.error(f"Router: Signal received but no model name in model_key: {model_key}")
                return
            
            if load:
                if self.at_capacity:
                    self.logger.info(f"Router: At capacity. Ignoring load signal for {mk}.")
                    return
                
                config = VectorConfigInternal(name="dummy", type=type_str, model=model, revision=revision)
                queue_name = self._get_model_queue_name(config)

                async with self.metrics_lock.writer:
                    async with self.model_locks[mk].writer:
                        await self._own_the_model(config, mk, queue_name, ["x"], 0, force_load=True)

                self.logger.info(f"Router: Loaded model {mk}.")
            else:
                # Get the queue info and cancel consumer
                if mk not in self.local_models:
                    self.logger.warning(f"Router: Model {mk} not found in local_models. Skipping.")
                    return

                async with self.metrics_lock.writer:
                    async with self.model_locks[mk].writer:
                        queue, consumer_tag, _ = self.local_models[mk]
                        
                        # Cancel the consumer to stop listening for requests
                        self.logger.debug(f"Router: Cancelling consumer for {mk}")
                        await self.bunny_talk.cancel_listener(queue, consumer_tag)

                        # Invalidate known_queues cache for this queue
                        config = VectorConfigInternal(name="dummy", type=type_str, model=model, revision=revision)
                        queue_name = self._get_model_queue_name(config)
                        try:
                            del self.known_queues[queue_name]
                        except KeyError:
                            self.logger.debug(f"Router: known_queues had no entry for {queue_name} during unload.")

                        # Unload the model from vector class cache
                        self._instances[type_str].unload_model(model, revision)

                        # Remove from local_models dict
                        self.local_models.pop(mk, None)

                        # if we are not at capacity, we need to start listening for requests on open queues
                        if not self._at_model_capacity(force_check=True):
                            await self._listen_on_open_queues(start_listening=True)

                self.logger.info(f"Router: unloaded model {model_key}.")
                
        except Exception as e:
            self.logger.error(f"Router: Error handling node signal (load: {load}, model_key: {model_key}): {e}")

    async def _rebalance_loop(self) -> None:
        while True:
            if self.metrics.is_leader():
                await self.model_rebalancer.rebalance(self.metrics.cluster_snapshot())
            await asyncio.sleep(METRICS_LOOP_INTERVAL_SECONDS)

    async def _metrics_meta_loop(self) -> None:
        while True:
            self.metrics.publish_meta(await self._metrics_meta())
            await asyncio.sleep(METRICS_LOOP_INTERVAL_SECONDS)

    async def _metrics_meta(self) -> Dict[str, Any]:
        async with self.metrics_lock.reader:
            loaded_models = [
                (list(model_key), load_timestamp)
                for model_key, (queue, consumer_tag, load_timestamp) in self.local_models.items()
            ]
        model_last_used = _serialize_model_last_used_meta(self.metrics.last_used_snapshot())

        return {
            "load_models": self.load_models,
            "at_capacity": self._at_model_capacity(),
            "total_ram_gb": round(TOTAL_RAM_GB, 2),
            "free_ram_gb": round(self._free_ram_gb, 2),
            "total_vram_gb": round(TOTAL_VRAM_GB, 2) if TOTAL_VRAM_GB is not None else None,
            "free_vram_gb": round(self._free_vram_gb, 2) if self._free_vram_gb is not None else None,
            "gpu_support": PYNVML_AVAILABLE,
            "gpu_available": GPU_HANDLE is not None,
            "loaded_models": _serialize_loaded_models_meta(loaded_models),
            NODE_META_MODEL_LAST_USED: model_last_used,
        }
