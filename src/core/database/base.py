from abc import ABC, abstractmethod
import copy
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Hashable, Tuple
from datetime import datetime
from heapq import nlargest
import uuid
import asyncio

from ..models.cluster import MetricsBucket
from ..models.document import Document, DocumentWithVectors, SearchResult, QueueDocument, QueueInfo, DocumentStatusResponse
from ..models.vector import CollectionConfigInternal, SearchQueryWithVectors, CollectionConfig
from ..common import (
    APP_PREFIX, DOC_NAMESPACE, DatabaseInfo, DatabaseFeatures, QueueOperationTypeLiteral,
    MAX_DATABASE_WAIT_SECONDS
)
from ..common.lock_manager import LockClient


class AmgixNotFound(Exception):
    """Exception raised when a resource (collection, document, etc.) is not found."""
    pass


class DatabaseBase(ABC):
    """
    Abstract base class for database operations.
    
    This class defines the interface for all database implementations.
    Concrete implementations should inherit from this class and implement
    all abstract methods.
    """
    
    # Maximum wait time in seconds for connection retries
    MAX_WAIT_SECONDS = MAX_DATABASE_WAIT_SECONDS
    
    # Class-level cache for database info with lock for thread safety
    _db_info_locked: Optional[DatabaseInfo] = None
    _db_info: Optional[DatabaseInfo] = None
    _probe_lock = asyncio.Lock()

    class SysCollectionType(Enum):
        """Enumeration of system collection types."""
        META = "meta"
        QUEUE = "queue"
        METRICS = "metrics"

    
    def __init__(self, connection_string: str, logger, **kwargs):
        """
        Store connection parameters for later use.
        
        Args:
            connection_string: Connection string for the database
            logger: Logger instance to use for this database
            **kwargs: Additional connection parameters specific to the database implementation
        """
        self.connection_string = connection_string
        self.connection_params = kwargs
        
        # Use provided logger
        self.logger = logger
        
        self.meta_collection = self.get_sys_collection_name(self.SysCollectionType.META)
        self.queue_collection = self.get_sys_collection_name(self.SysCollectionType.QUEUE)
        self.metrics_collection = self.get_sys_collection_name(self.SysCollectionType.METRICS)
    
    @abstractmethod
    async def probe(self) -> None:
        """
        Probe database and store DatabaseInfo internally.
        
        This method should:
        1. Test available features (mark unavailable features as False)
        2. Store results in _db_info class variable
        
        Note: This method should use self._probe_lock when setting _db_info.
        """
        pass
    
    def _string_to_uuid(self, string_id: str) -> str:
        """
        Convert a string ID to a deterministic UUID5 for database storage.
        
        Args:
            string_id: The string ID to convert
            
        Returns:
            str: UUID5 string representation
        """
        return str(uuid.uuid5(DOC_NAMESPACE, string_id))

    @staticmethod
    def rrf_fuse(
        id_lists: List[List[Hashable]],
        weights: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        k: int = 2,
    ) -> List[Tuple[Hashable, float]]:
        """
        Reciprocal Rank Fusion (RRF) combining multiple ranked lists using only ranks.

        Args:
            id_lists: One ranked list of ids per vector/prefetch (highest-score first).
            weights: Weight per list; must be same length as id_lists.
            limit: Max number of fused items to return.
            score_threshold: Optional minimum fused score to keep.
            k: RRF constant. Default 2.

        Returns:
            List of (item_id, fused_rrf_score), sorted by fused score desc.
        """

        fused_scores: Dict[Hashable, float] = {}
        fused_scores_get = fused_scores.get

        for list_idx, ids in enumerate(id_lists):
            weight = weights[list_idx]
            # ids are assumed pre-ranked desc; ranks start at 1
            for rank_idx, item_id in enumerate(ids, start=1):
                fused_scores[item_id] = fused_scores_get(item_id, 0.0) + weight / (k + rank_idx)

        # Build iterator of fused items, optionally apply threshold
        items_iter = (
            (item_id, fused_scores[item_id])
            for item_id in fused_scores
        )
        if score_threshold is not None:
            items_iter = (t for t in items_iter if t[1] >= score_threshold)

        # Efficiently take top-k by fused score without sorting everything
        return nlargest(limit, items_iter, key=lambda t: t[1])

    @staticmethod
    def linear_weighted_score_fuse(
        scored_lists: List[List[Tuple[Hashable, float]]],
        weights: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Hashable, float]]:
        """
        Fuse retrievers by min-max normalizing raw scores within each list, then summing weight * norm_score.

        Missing retriever scores for an id count as 0. Ties (min == max) within a list assign normalized 1.0.
        """
        normalized_maps: List[Dict[Hashable, float]] = []
        for arm in scored_lists:
            if not arm:
                normalized_maps.append({})
                continue
            raw_scores = [s for _, s in arm]
            mn, mx = min(raw_scores), max(raw_scores)
            if mx == mn:
                normalized_maps.append({item_id: 1.0 for item_id, _ in arm})
            else:
                scale = mx - mn
                normalized_maps.append(
                    {item_id: (s - mn) / scale for item_id, s in arm}
                )

        candidates: Dict[Hashable, float] = {}
        for list_idx, w in enumerate(weights):
            for item_id, nscore in normalized_maps[list_idx].items():
                candidates[item_id] = candidates.get(item_id, 0.0) + w * nscore

        items_iter = ((i, candidates[i]) for i in candidates)
        if score_threshold is not None:
            items_iter = (t for t in items_iter if t[1] >= score_threshold)
        return nlargest(limit, items_iter, key=lambda t: t[1])

    @abstractmethod
    async def create_collection(self, collection_name: str, config: CollectionConfigInternal) -> bool:
        """
        Create a new collection with specified vector configurations.
        
        Args:
            collection_name: Name of the collection to create
            config: Collection configuration including vectors and indexed fields
            
        Returns:
            bool: True if collection was created successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def validate_features(self, config: CollectionConfig) -> None:
        """
        Check if the database supports all features required by the collection configuration.
        
        Args:
            config: Collection configuration to validate against database capabilities
            
        Raises:
            RuntimeError: If the database doesn't support required features
        """
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        List existing collections.
        
        Args:
        
        Returns:
            List[str]: List of collection names
        """
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and all its documents.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if collection was deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def empty_collection(self, collection_name: str) -> bool:
        """
        Remove all documents from a collection without deleting the collection itself.
        
        Args:
            collection_name: Name of the collection to empty
            
        Returns:
            bool: True if collection was emptied successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def add_documents(self, collection_name: str, documents_with_vectors: List[DocumentWithVectors], is_new: bool, store_content: bool, collection_config: CollectionConfigInternal, lock_client: LockClient) -> None:
        """
        Add or update documents in a collection.

        Args:
            collection_name: Name of the collection to add the documents to
            documents_with_vectors: List of documents with pre-calculated vectors
            is_new: Whether the documents are known to be new (insert) or existing (update)
            store_content: Whether to store document content in the database
            lock_client: Distributed lock client for serializing writes (SQL backends only)
        """
        pass
    
    @abstractmethod
    async def get_documents(self, collection_name: str, document_ids: List[str], suppress_not_found: bool = False) -> List[Optional[DocumentWithVectors]]:
        """
        Retrieve multiple documents by IDs.
        
        Args:
            collection_name: Name of the collection to retrieve from
            document_ids: List of document IDs to retrieve
            suppress_not_found: If True, don't raise AmgixNotFound when documents are missing (default: False)
            
        Returns:
            List[Optional[DocumentWithVectors]]: List of documents (with token_lengths from payload, vectors=[]) 
                                                  in the same order as document_ids, None for missing documents
            
        Raises:
            AmgixNotFound: If suppress_not_found is False and not all documents are found
        """
        pass
    
    @abstractmethod
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            collection_name: Name of the collection to delete from
            document_id: ID of the document to delete
            
        Returns:
            bool: True if document was deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        collection_name: str, 
        query: SearchQueryWithVectors,
        collection_config: CollectionConfigInternal
    ) -> List[SearchResult]:
        """
        Perform a hybrid search on the collection using precalculated vectors.
        
        Args:
            collection_name: Name of the collection to search
            query: Search query with precalculated vectors
            collection_config: Collection configuration for distance function selection
            
        Returns:
            List[SearchResult]: List of search results with document data and scores
        """
        pass
    

    
    @abstractmethod
    async def get_collection_info(self, collection_name: str) -> CollectionConfig:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionConfig: Collection configuration
        """
        pass

    @abstractmethod
    async def get_collection_info_internal(self, collection_name: str) -> CollectionConfigInternal:
        """
        Get internal information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionConfigInternal: Collection configuration
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if the database connection is active and healthy.
        
        Returns:
            bool: True if connected and healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def configure(self) -> None:
        """
        Configure the database with system objects and ensure proper setup.
        
        This method is responsible for:
        1. Creating system tables/collections if they don't exist
        2. Setting up required database objects
        3. Ensuring the database is properly configured for operation
        """
        pass
    
    async def wait_connected(self) -> None:
        """
        Wait for the database connection to become available.
        
        Uses exponential backoff starting at 2 seconds, increasing by 2 seconds
        on each iteration, up to a maximum of MAX_WAIT_SECONDS.
        """
        wait_time = 2
        
        while True:
            if await self.is_connected():
                return
            
            self.logger.warning(f"Database is not available, will retry in {wait_time} seconds")
            
            await asyncio.sleep(wait_time)
            
            wait_time = min(wait_time + 2, self.MAX_WAIT_SECONDS)
    
    async def check_features(self) -> None:
        """
        Check database features and validate vector support.
        
        This method:
        1. Calls probe() to discover database capabilities
        2. Logs database version and features
        3. Validates that DENSE_VECTORS support is available
        4. Raises exception if vector support is missing
        """
        await self.probe()
        
        async with self._probe_lock:
            if self._db_info_locked:
                self._db_info = copy.deepcopy(self._db_info_locked)
                self.logger.info(f"Database version: {self._db_info_locked.version}")
                self.logger.info(f"Database features: {self._db_info_locked.features}")
                
                # if not self._db_info_locked.features.get(DatabaseFeatures.DENSE_VECTORS, False):
                #     raise RuntimeError(f"Database {self._db_info_locked.version} does not support dense vectors, which is required")
            else:
                raise RuntimeError("Failed to probe database features")

    def get_cached_database_info(self) -> Optional[DatabaseInfo]:
        """Return version/features from the last successful ``probe()`` (no I/O)."""
        return self._db_info_locked

    def get_sys_collection_name(self, collection_type: 'DatabaseBase.SysCollectionType') -> str:
        """
        Generate a system collection name.
        
        Args:
            collection_type: Type of system collection (meta)
            
        Returns:
            Full collection name in the format
        """
        return f"{APP_PREFIX}_sys_{collection_type.value}"
    
    @abstractmethod
    async def add_to_queue(
        self,
        collection_name: str,
        collection_id: str,
        documents: List[Document],
        op_type: QueueOperationTypeLiteral,
        request_timestamp: Optional[datetime] = None,
    ) -> List[str]:
        """
        Add documents to the processing queue.
        
        Args:
            collection_name: Name of the collection these documents belong to
            collection_id: Internal collection identifier
            documents: List of documents to add to the queue
            op_type: Queue operation type (upsert or delete)
            request_timestamp: Caller-supplied timestamp for delete operations
            
        Returns:
            List[str]: The queue_ids for the queue entries
        """
        pass
    
    @abstractmethod
    async def get_from_queue(self, queue_ids: List[str], suppress_not_found: bool = False) -> List['QueueDocument']:
        """
        Retrieve documents from the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
            suppress_not_found: If True, return only found entries instead of raising AmgixNotFound
            
        Returns:
            List[QueueDocument]: List of queue documents with status and metadata
        """
        pass
    
    @abstractmethod
    async def delete_from_queue(self, queue_ids: List[str]) -> None:
        """
        Remove documents from the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
        """
        pass
    
    @abstractmethod
    async def delete_from_queue_by_collection(self, collection_name: str) -> None:
        """
        Remove all documents from the processing queue for a specific collection.
        
        Args:
            collection_name: Name of the collection to clear from queue
        """
        pass

    @abstractmethod
    async def delete_upserts_from_queue(self, collection_name: str, doc_id: str, before_timestamp: datetime) -> None:
        """
        Remove upsert queue entries for a document with doc_timestamp <= before_timestamp.

        Args:
            collection_name: Name of the collection
            doc_id: Document identifier
            before_timestamp: Delete upsert entries with doc_timestamp at or before this timestamp
        """
        pass
    
    @abstractmethod
    async def update_queue_status(self, queue_ids: List[str], status: str, try_count: int, info: str) -> None:
        """
        Update the status of documents in the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
            status: New status to set for the documents
            try_count: New try count to set for the documents
            info: Additional information about the status (e.g., error details)
        """
        pass
    
    @abstractmethod
    async def get_queue_entries(self, collection_name: str, doc_id: Optional[str] = None) -> List['QueueDocument']:
        """
        Get all queue entries for a specific document or all queue entries for a collection.
        
        Args:
            collection_name: Name of the collection this document belongs to
            doc_id: Unique identifier for the document, or None to get all queue entries for the collection
            
        Returns:
            List[QueueDocument]: List of queue entries for this document or collection
        """
        pass
    
    @abstractmethod
    async def get_queue_info(self, collection_name: str) -> 'QueueInfo':
        """
        Get queue statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            QueueInfo: Counts of documents in each queue state
        """
        pass
    
    @abstractmethod
    async def get_queue_statuses(self, collection_name: str, doc_id: str) -> 'DocumentStatusResponse':
        """
        Get comprehensive status of a document including collection and queue states.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document identifier
            
        Returns:
            DocumentStatusResponse: Complete status information with empty list if no statuses found
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Union[int, Dict[str, float]]]:
        """
        Get collection statistics: document count and average document length per field_vector_name.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with structure: {"doc_count": int, "avgdls": {"field_vector_name": float, ...}}
        """
        pass

    @abstractmethod
    async def set_collection_stats(self, collection_name: str, stats: Dict[str, Union[int, Dict[str, float]]]) -> None:
        """
        Set collection statistics: document count and average document length per field_vector_name.
        
        Args:
            collection_name: Name of the collection
            stats: Dict with structure: {"doc_count": int, "avgdls": {"field_vector_name": float, ...}}
        """
        pass

    @abstractmethod
    async def append_metric_buckets(self, hostname: str, source: str, buckets: List[MetricsBucket]) -> None:
        """
        Persist completed metric buckets for this node. Implementations should upsert on
        (hostname, source, key, dims, bucket_start, bucket_seconds).

        Args:
            hostname: Reporting node hostname
            source: Reporting node source identifier (e.g. 'router', 'api')
            buckets: Completed, immutable metric buckets to persist
        """
        pass

    @abstractmethod
    async def trim_metric_buckets(self, bucket_seconds: int, cutoff: float) -> None:
        """
        Delete metric buckets older than the given cutoff timestamp.

        Args:
            bucket_seconds: Resolution of buckets to trim (e.g. 60 for 1m, 300 for 5m)
            cutoff: Unix timestamp; buckets with bucket_start < cutoff are deleted
        """
        pass

    @abstractmethod
    async def query_metric_buckets(
        self,
        bucket_seconds: int,
        since: float,
        until: float,
        keys: Optional[List[str]] = None,
    ) -> List[MetricsBucket]:
        """
        Return stored buckets for the given resolution and time range, ordered by
        bucket_start ascending.

        Args:
            bucket_seconds: Resolution to query (e.g. 60 for 1m, 300 for 5m)
            since: Inclusive start unix timestamp
            until: Exclusive end unix timestamp
            keys: If provided, restrict results to these metric keys; None returns all keys
        """
        pass
