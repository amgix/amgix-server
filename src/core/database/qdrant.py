import copy
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
import uuid
from urllib.parse import urlparse
from datetime import datetime, timezone

from qdrant_client import AsyncQdrantClient

from qdrant_client.grpc import *
from qdrant_client.http import models as rest
from qdrant_client.models import FormulaQuery, SumExpression, MultExpression, Prefetch

from .base import DatabaseBase, AmgixNotFound
from ..models.cluster import MetricsBucket
from ..models.document import Document, DocumentWithVectors, SearchResult, QueueDocument, QueueInfo, DocumentStatus, DocumentStatusResponse, VectorScore
from ..models.vector import CollectionConfigInternal, SearchQueryWithVectors, CollectionConfig, VectorConfig, MetadataFilter, internal_to_user_config
from ..common import APP_PREFIX, VectorType, DatabaseInfo, DatabaseFeatures, SEARCH_PREFETCH_MULTIPLIER, DenseDistance, QueuedDocumentStatus, QueueOperationType, QueueOperationTypeLiteral, get_user_collection_name, MetadataValueType
from ..common.lock_manager import LockClient


class QdrantDatabase(DatabaseBase):
    """
    Qdrant implementation of the DatabaseBase interface.
    
    This class handles interactions with a Qdrant vector database.
    """
    
    def __init__(self, connection_string: str, logger, **kwargs):
        """
        Initialize with Qdrant connection parameters and establish a persistent client.
        
        Args:
            connection_string: URL to the Qdrant server (e.g., "http://localhost:6333" or "localhost:6334")
            logger: Logger instance to use for this database
            **kwargs: Additional parameters for the Qdrant client
        """
        super().__init__(connection_string, logger=logger, **kwargs)
        
        # Parse the connection string to extract host and port
        parsed_url = urlparse(connection_string)
        
        if parsed_url.scheme and parsed_url.hostname:
            # URL format (e.g., "qdrant://localhost:6334")
            host = parsed_url.hostname
            port = parsed_url.port or 6334  # Default to 6334 if no port specified
        else:
            # Simple host:port format (e.g., "localhost:6334" or "qdrant:6334")
            parts = connection_string.split(":")
            host = parts[0] or "localhost"
            port = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 6334
            
        # Create a persistent async client instance
        self.client = AsyncQdrantClient(
            host=host,
            port=port,        # Use the port from connection string
            prefer_grpc=True, # Use gRPC when possible
            **kwargs
        )
    
    async def probe(self) -> None:
        # Get actual version info from Qdrant
        info = await self.client.info()
        
        # Qdrant always supports dense vectors
        features = {
            DatabaseFeatures.DENSE_VECTORS: True
        }
        
        async with self._probe_lock:
            self._db_info_locked = DatabaseInfo(
                version=info.version,
                features=features
            )
            self._db_info = copy.deepcopy(self._db_info_locked)
    
    async def configure(self) -> None:
        """
        Configure the database with system objects and ensure proper setup.
        """

        # Create the system meta collection if it doesn't exist
        try:
            if not await self.client.collection_exists(self.meta_collection):
                self.logger.info(f"Creating system meta collection")

                # Create a minimal vector configuration for the metadata collection
                # We need at least one vector field even if we don't use it
                await self.client.create_collection(
                    collection_name=self.meta_collection,
                    vectors_config={
                        "dummy": rest.VectorParams(
                            size=1,
                            distance=rest.Distance.DOT
                        )
                    }
                )
            else:
                self.logger.debug(f"System meta collection already exists")
        except Exception as e:
            if "already exists" in str(e).lower() or "already exists" in str(e).lower():
                # Another instance beat us to it - that's fine
                self.logger.info(f"System meta collection already exists")
            else:
                self.logger.error(f"Failed to create system meta collection: {e}")
                raise
        
        # Create the system queue collection if it doesn't exist
        try:
            if not await self.client.collection_exists(self.queue_collection):
                self.logger.info(f"Creating system queue collection")

                # Create a minimal vector configuration for the queue collection
                # We need at least one vector field even if we don't use it
                await self.client.create_collection(
                    collection_name=self.queue_collection,
                    vectors_config={
                        "dummy": rest.VectorParams(
                            size=1,
                            distance=rest.Distance.DOT
                        )
                    }
                )
                
                # Create payload indexes for queue fields after collection creation
                await self.client.create_payload_index(
                    collection_name=self.queue_collection,
                    field_name="collection_name",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                await self.client.create_payload_index(
                    collection_name=self.queue_collection,
                    field_name="doc_id",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                await self.client.create_payload_index(
                    collection_name=self.queue_collection,
                    field_name="status",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                await self.client.create_payload_index(
                    collection_name=self.queue_collection,
                    field_name="timestamp",
                    field_schema=rest.PayloadSchemaType.DATETIME
                )
            else:
                self.logger.debug(f"System queue collection already exists")

            # Ensure new queue payload indexes exist for all deployments.
            queue_info = await self.client.get_collection(collection_name=self.queue_collection)
            payload_schema = getattr(queue_info, "payload_schema", None) or {}

            if "op_type" not in payload_schema:
                await self.client.create_payload_index(
                    collection_name=self.queue_collection,
                    field_name="op_type",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )

            if "doc_timestamp" not in payload_schema:
                await self.client.create_payload_index(
                    collection_name=self.queue_collection,
                    field_name="doc_timestamp",
                    field_schema=rest.PayloadSchemaType.DATETIME
                )
        except Exception as e:
            if "already exists" in str(e).lower():
                self.logger.info(f"System queue collection already exists")
            else:
                self.logger.error(f"Failed to create system queue collection: {e}")
                raise

        # Create the system metrics collection if it doesn't exist
        try:
            if not await self.client.collection_exists(self.metrics_collection):
                self.logger.info(f"Creating system metrics collection")

                await self.client.create_collection(
                    collection_name=self.metrics_collection,
                    vectors_config={
                        "dummy": rest.VectorParams(
                            size=1,
                            distance=rest.Distance.DOT
                        )
                    }
                )

                await self.client.create_payload_index(
                    collection_name=self.metrics_collection,
                    field_name="key",
                    field_schema=rest.PayloadSchemaType.KEYWORD
                )
                await self.client.create_payload_index(
                    collection_name=self.metrics_collection,
                    field_name="bucket_seconds",
                    field_schema=rest.PayloadSchemaType.INTEGER
                )
                await self.client.create_payload_index(
                    collection_name=self.metrics_collection,
                    field_name="bucket_start",
                    field_schema=rest.PayloadSchemaType.INTEGER
                )
            else:
                self.logger.debug(f"System metrics collection already exists")
        except Exception as e:
            if "already exists" in str(e).lower():
                self.logger.info(f"System metrics collection already exists")
            else:
                self.logger.error(f"Failed to create system metrics collection: {e}")
                raise

    async def create_collection(self, collection_name: str, config: CollectionConfigInternal) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection to create
            config: Collection configuration including vectors and indexed fields
            
        Returns:
            bool: True if collection was created successfully
        """
        # Define vector configurations for the collection
        vectors_config = {}
        sparse_vectors_config = {}
        
        # For each vector in the configuration, create field-specific vectors
        # based on the index_fields specified in the vector config
        for vector_config in config.vectors:
            vector_name = vector_config.name
            
            # Determine vector parameters based on type
            if VectorType.is_dense(vector_config.type):
                # For dense vectors, we need dimensions
                if not vector_config.dimensions:
                    raise ValueError(f"Dimensions are required for dense vector {vector_name}")
                
                # Create field-specific vectors for each field in index_fields
                for field in vector_config.index_fields:
                    field_vector_name = f"{field}_{vector_name}"
                    
                    # Map our distance metric to Qdrant's distance enum
                    if vector_config.dense_distance == DenseDistance.COSINE:
                        qdrant_distance = rest.Distance.COSINE
                    elif vector_config.dense_distance == DenseDistance.DOT:
                        qdrant_distance = rest.Distance.DOT
                    elif vector_config.dense_distance == DenseDistance.EUCLID:
                        qdrant_distance = rest.Distance.EUCLID
                    else:
                        # This should never happen due to Pydantic validation, but fail fast if it does
                        raise ValueError(f"Invalid dense_distance '{vector_config.dense_distance}' for vector '{vector_name}'. Must be one of {list(DenseDistance.all())}")
                    
                    vectors_config[field_vector_name] = rest.VectorParams(
                        size=vector_config.dimensions,
                        distance=qdrant_distance
                    )
                
            elif vector_config.type in VectorType.sparse_types():
                # For sparse vectors, we'll use Qdrant's sparse vector support
                # Create field-specific vectors for each field in index_fields
                idf_modifier = rest.Modifier.IDF if vector_config.type in VectorType.custom_tokenization() else None
                for field in vector_config.index_fields:
                    field_vector_name = f"{field}_{vector_name}"
                    sparse_vectors_config[field_vector_name] = rest.SparseVectorParams(
                        modifier=idf_modifier,
                        index=rest.SparseIndexParams(
                            on_disk=True
                        )
                    )
        
        # Create the collection
        create_kwargs = {
            "collection_name": collection_name,
            "vectors_config": vectors_config
        }
        
        # Add sparse vectors config if we have any sparse vectors
        if sparse_vectors_config:
            create_kwargs["sparse_vectors_config"] = sparse_vectors_config
        
        await self.client.create_collection(**create_kwargs)
        
        # Store the configuration as a point in the metadata collection
        await self.client.upsert(
            collection_name=self.meta_collection,
            points=[
                rest.PointStruct(
                    id=self._string_to_uuid(f"{collection_name}_config"),
                    payload={
                        "config": config.model_dump(mode="json")
                    },
                    vector={"dummy": [0.0]}  # Dummy vector
                )
            ]
        )
        
        # Create payload indexes for tags (keyword list)
        # Vector fields (name, description, content) are covered by vectors
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="tags",
            field_schema=rest.PayloadSchemaType.KEYWORD
        )
        
        # Create payload indexes for declared metadata fields
        # Index on metadata.<key>.value to access the actual value, not the MetaValue object
        if config.metadata_indexes:
            for metadata_index in config.metadata_indexes:
                field_path = f"metadata.{metadata_index.key}.value"
                if metadata_index.type == MetadataValueType.STRING:
                    schema_type = rest.PayloadSchemaType.KEYWORD
                elif metadata_index.type == MetadataValueType.INTEGER:
                    schema_type = rest.PayloadSchemaType.INTEGER
                elif metadata_index.type == MetadataValueType.FLOAT:
                    schema_type = rest.PayloadSchemaType.FLOAT
                elif metadata_index.type == MetadataValueType.BOOLEAN:
                    schema_type = rest.PayloadSchemaType.BOOL
                elif metadata_index.type == MetadataValueType.DATETIME:
                    schema_type = rest.PayloadSchemaType.DATETIME
                else:
                    continue
                
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_path,
                    field_schema=schema_type
                )
        
        return True
    
    async def list_collections(self) -> List[str]:
        """
        List existing collections.
        
        Args:
        
        Returns:
            List[str]: List of collection names
        """
        collections = await self.client.get_collections()
        collection_names = [
            collection.name for collection in collections.collections
            if collection.name.startswith(f"{APP_PREFIX}_") and not collection.name.startswith(f'{APP_PREFIX}_sys_')
        ]
        
        return collection_names
    
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and its metadata collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if collection was deleted successfully
        """
        # Delete the main collection
        await self.client.delete_collection(collection_name=collection_name)

        # Delete collection config and stats
        await self.client.delete(
            collection_name=self.meta_collection,
            points_selector=[
                self._string_to_uuid(f"{collection_name}_config"),
                self._string_to_uuid(f"{collection_name}_stats")
            ]
        )
        
        # Clean up queue entries for this collection
        await self.delete_from_queue_by_collection(collection_name)
        
        return True
    
    async def empty_collection(self, collection_name: str) -> bool:
        """
        Remove all documents from a collection without deleting the collection itself.
        
        Args:
            collection_name: Name of the collection to empty
            
        Returns:
            bool: True if collection was emptied successfully
        """
        # Delete all points using an empty filter (matches everything)
        await self.client.delete(
            collection_name=collection_name,
            points_selector=Filter()
        )
        
        # Delete collection stats (reset to empty since collection is now empty)
        await self.client.delete(
            collection_name=self.meta_collection,
            points_selector=[self._string_to_uuid(f"{collection_name}_stats")]
        )
        
        return True
    
    async def add_documents(self, collection_name: str, documents_with_vectors: List[DocumentWithVectors], is_new: bool, store_content: bool, collection_config: CollectionConfigInternal, lock_client: LockClient) -> None:
        """
        Add or update documents in a collection.

        Args:
            collection_name: Name of the collection to add the documents to
            documents_with_vectors: List of documents with pre-calculated vectors
            is_new: Whether the documents are known to be new (insert) or existing (update)
            store_content: Whether to store document content in the database
            lock_client: Unused (Qdrant does not require collection ingest lock)

        Returns:
            None
        """
        points = []
        
        for document_with_vectors in documents_with_vectors:
            vectors = document_with_vectors.vectors
            
            # Convert document to a format Qdrant can understand
            exclude_fields = {'vectors'}
            if not store_content:
                exclude_fields.add('content')
            doc_dict = document_with_vectors.model_dump(exclude=exclude_fields)
            
            # Convert VectorData objects to Qdrant vector format
            qdrant_vectors = {}
            
            for vector_data in vectors:
                field_vector_name = f"{vector_data.field}_{vector_data.vector_name}"
                
                if VectorType.is_dense(vector_data.vector_type):
                    qdrant_vectors[field_vector_name] = vector_data.dense_vector
                else:
                    qdrant_vectors[field_vector_name] = rest.SparseVector(
                        indices=vector_data.sparse_indices,
                        values=vector_data.sparse_values
                    )
            
            # Create PointStruct for this document
            points.append(
                rest.PointStruct(
                    id=self._string_to_uuid(document_with_vectors.id),
                    payload=doc_dict,
                    vector=qdrant_vectors
                )
            )
        
        # Add all documents with their vectors in a single upsert
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    async def get_documents(self, collection_name: str, document_ids: List[str], suppress_not_found: bool = False) -> List[Optional[DocumentWithVectors]]:
        """
        Retrieve multiple documents by IDs.
        
        Args:
            collection_name: Name of the collection to retrieve from
            document_ids: List of document IDs to retrieve
            suppress_not_found: If True, don't raise AmgixNotFound when documents are missing (default: False)
            
        Returns:
            List[Optional[DocumentWithVectors]]: List of documents in the same order as document_ids, None for missing documents
            
        Raises:
            AmgixNotFound: If suppress_not_found is False and not all documents are found
        """
        # Convert document IDs to UUIDs and create mapping
        uuid_to_doc_id = {}
        uuids = []
        for doc_id in document_ids:
            uuid = self._string_to_uuid(doc_id)
            uuid_to_doc_id[uuid] = doc_id
            uuids.append(uuid)
        
        # Get all documents in one call
        result = await self.client.retrieve(
            collection_name=collection_name,
            ids=uuids
        )
        
        # Check if we got the expected number of results
        if len(result) != len(document_ids) and not suppress_not_found:
            found_ids = {uuid_to_doc_id[point.id] for point in result}
            missing_ids = set(document_ids) - found_ids
            raise AmgixNotFound(f"Documents not found for document_ids: {', '.join(missing_ids)}")
        
        # Create a map of document_id to DocumentWithVectors
        doc_map = {}
        for point in result:
            doc_id = uuid_to_doc_id[point.id]
            doc_map[doc_id] = DocumentWithVectors.from_dict(
                {**point.payload, "vectors": []},
                store_content=True,
                skip_validation=True
            )
        
        # Return documents in the same order as document_ids, None for missing ones
        return [doc_map.get(doc_id) for doc_id in document_ids]
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            collection_name: Name of the collection to delete from
            document_id: ID of the document to delete
            
        Returns:
            bool: True if document was deleted successfully
            
        Raises:
            AmgixNotFound: If document doesn't exist
        """
        # Check if document exists first
        result = await self.client.retrieve(
            collection_name=collection_name,
            ids=[self._string_to_uuid(document_id)]
        )
        
        if not result:
            raise AmgixNotFound(f"Document with ID '{document_id}' not found in collection '{get_user_collection_name(collection_name)}'")
        
        # Delete the document by ID
        await self.client.delete(
            collection_name=collection_name,
            points_selector=[self._string_to_uuid(document_id)]
        )
        
        return True
    
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
        # Prepare search conditions
        search_conditions = []
        
        # Add tag filter if specified (mapped from document_tags at API model level)
        if query.document_tags:
            if query.document_tags_match_all:
                # AND behavior: documents must have ALL specified tags
                for tag in query.document_tags:
                    search_conditions.append(
                        rest.FieldCondition(
                            key="tags",
                            match=rest.MatchValue(value=tag)
                        )
                    )
            else:
                # OR behavior: documents must have ANY of the specified tags
                search_conditions.append(
                    rest.FieldCondition(
                        key="tags",
                        match=rest.MatchAny(any=query.document_tags)
                    )
                )
        
        metadata_filter = self._convert_metadata_filter_to_qdrant(query.metadata_filter, collection_config) if query.metadata_filter else None
        tags_filter = rest.Filter(must=search_conditions) if search_conditions else None

        if tags_filter and metadata_filter:
            final_filter = rest.Filter(must=[tags_filter, metadata_filter])
        elif tags_filter:
            final_filter = tags_filter
        else:
            final_filter = metadata_filter

        # Build weight map keyed by field_vector_name (consistent with SQL backends).
        # Default is 1.0, overridden by user-provided weights.
        weight_lookup = {(w.vector_name, w.field): w.weight for w in query.vector_weights}
        weight_map: Dict[str, float] = {}

        # formula_expressions are no longer used; keeping example commented out for reference
        # formula_expressions = []
        # for i, _vector_data in enumerate(query.vectors):
        #     weight = weight_map.get(i, 1.0)
        #     formula_expressions.append(
        #         MultExpression(mult=[weight, f"$score[{i}]"])
        #     )
        
        # Execute batch vector searches and combine with RRF fusion
        
        # Prepare batch search requests
        batch_requests = []
        batch_vector_names: List[str] = []
        for vector_data in query.vectors:
            vector_name = vector_data.vector_name
            field = vector_data.field
            
            # Create the field-specific vector name
            field_vector_name = f"{field}_{vector_name}"
            
            # Skip vectors with weight 0 - they contribute nothing
            weight = weight_lookup.get((vector_name, field), 1.0)
            if weight == 0:
                continue
            weight_map[field_vector_name] = weight
            
            # Create search request for this vector
            # Request only payload fields needed for SearchResult (exclude content to reduce transfer)
            payload_fields = ["id", "timestamp", "name", "description", "metadata", "tags"]
            if VectorType.is_dense(vector_data.vector_type):
                query_request = rest.QueryRequest(
                    using=field_vector_name,
                    query=vector_data.dense_vector,
                    limit=int(query.limit * SEARCH_PREFETCH_MULTIPLIER),
                    with_payload=payload_fields,
                    with_vector=False,
                    filter=final_filter
                )
            else:
                # Sparse vector
                sparse_vector = rest.SparseVector(
                    indices=vector_data.sparse_indices, 
                    values=vector_data.sparse_values
                )
                query_request = rest.QueryRequest(
                    using=field_vector_name,
                    query=sparse_vector,
                    limit=int(query.limit * SEARCH_PREFETCH_MULTIPLIER),
                    with_payload=payload_fields,
                    with_vector=False,
                    filter=final_filter
                )
            
            batch_requests.append(query_request)
            batch_vector_names.append(field_vector_name)

        if not batch_requests:
            return []
        
        # Execute batch search
        # batch_start = time.time()
        batch_response = await self.client.query_batch_points(
            collection_name=collection_name,
            requests=batch_requests
        )
        # batch_time = (time.time() - batch_start) * 1000
        # print(f"Qdrant batch search time: {batch_time:.2f}ms")

        # for res in batch_response:
        #     for point in res.points:
        #         print(f"{point.payload["id"]} {point.score}")
        
        # Extract ranked ids, optional raw scores for response, and linear-fusion inputs in one pass
        id_lists: List[List] = []
        scored_lists: List[List[tuple]] = []
        point_lookup = {}
        raw_scores_map = {}
        arm_weights = [weight_map[name] for name in batch_vector_names]

        for idx, response in enumerate(batch_response):
            field_vector_name = batch_vector_names[idx]

            ids = []
            scored_arm: List[tuple] = []
            for rank, point in enumerate(response.points, 1):  # 1-based ranking
                ids.append(point.id)
                scored_arm.append((point.id, float(point.score)))
                point_lookup[point.id] = point

                if query.raw_scores:
                    field, vector = field_vector_name.split('_', 1)
                    vector_score = VectorScore(
                        field=field,
                        vector=vector,
                        score=point.score,
                        rank=rank
                    )
                    if point.id not in raw_scores_map:
                        raw_scores_map[point.id] = []
                    raw_scores_map[point.id].append(vector_score)
            id_lists.append(ids)
            scored_lists.append(scored_arm)

        if query.fusion_mode == "linear":
            fused_results = self.linear_weighted_score_fuse(
                scored_lists=scored_lists,
                weights=arm_weights,
                limit=query.limit,
                score_threshold=query.score_threshold,
            )
        else:
            fused_results = self.rrf_fuse(
                id_lists=id_lists,
                weights=arm_weights,
                limit=query.limit,
                score_threshold=query.score_threshold,
                k=2
            )
        # fusion_time = (time.time() - fusion_start) * 1000
        # print(f"RRF fusion post-processing time: {fusion_time:.2f}ms")
        
        # Convert fused results back to SearchResult format
        # conversion_start = time.time()
        results = []
        for item_id, fused_score in fused_results:
            point_data = point_lookup[item_id]
            # Create SearchResult by adding score and vector_scores to payload
            search_result_data = point_data.payload
            search_result_data['score'] = fused_score
            search_result_data['vector_scores'] = raw_scores_map.get(item_id, [])
            # Use from_dict to handle proper type conversion
            results.append(SearchResult.from_dict(search_result_data, skip_validation=True))
        # conversion_time = (time.time() - conversion_start) * 1000
        # print(f"Result conversion time: {conversion_time:.2f}ms")
        
        return results

    def _convert_metadata_filter_to_qdrant(self, metadata_filter: MetadataFilter, collection_config: CollectionConfigInternal) -> Optional[rest.Filter]:
        """Convert recursive MetadataFilter to Qdrant Filter."""
        if not metadata_filter:
            return None

        metadata_indexes_map = {idx.key: idx.type for idx in (collection_config.metadata_indexes or [])}

        def convert_node(node: MetadataFilter) -> Any:
            if node.key:
                field_path = f"metadata.{node.key}.value"
                if node.op == "eq":
                    return rest.FieldCondition(
                        key=field_path,
                        match=rest.MatchValue(value=node.value)
                    )

                range_params: Dict[str, Any] = {}
                if node.op == "gt":
                    range_params["gt"] = node.value
                elif node.op == "gte":
                    range_params["gte"] = node.value
                elif node.op == "lt":
                    range_params["lt"] = node.value
                elif node.op == "lte":
                    range_params["lte"] = node.value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {node.op}")

                if metadata_indexes_map.get(node.key) == MetadataValueType.DATETIME:
                    parsed_params = {
                        k: datetime.fromisoformat(v.replace("Z", "+00:00"))
                        for k, v in range_params.items()
                    }
                    return rest.FieldCondition(
                        key=field_path,
                        range=rest.DatetimeRange(**parsed_params)
                    )

                return rest.FieldCondition(
                    key=field_path,
                    range=rest.Range(**range_params)
                )

            must_conditions = [convert_node(child) for child in (node.and_ or [])]
            should_conditions = [convert_node(child) for child in (node.or_ or [])]
            must_not_conditions = [convert_node(node.not_)] if node.not_ else []

            return rest.Filter(
                must=must_conditions or None,
                should=should_conditions or None,
                must_not=must_not_conditions or None
            )

        converted = convert_node(metadata_filter)
        if isinstance(converted, rest.Filter):
            return converted
        return rest.Filter(must=[converted])
    
    async def get_collection_info_internal(self, collection_name: str) -> CollectionConfigInternal:
        """
        Get internal information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionConfigInternal: Collection configuration
        """
        # Retrieve the configuration point
        result = await self.client.retrieve(
            collection_name=self.meta_collection,
            ids=[self._string_to_uuid(f"{collection_name}_config")]
        )
        
        if not result or len(result) == 0:
            self.logger.debug(f"Configuration not found for collection {get_user_collection_name(collection_name)}")
            raise AmgixNotFound("Configuration not found")
        
        # Extract the configuration from the payload
        config_data = result[0].payload.get("config", {})
        
        # Create a CollectionConfigInternal object from the data
        return CollectionConfigInternal.model_validate(config_data)
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Union[int, Dict[str, float]]]:
        result = await self.client.retrieve(
            collection_name=self.meta_collection,
            ids=[self._string_to_uuid(f"{collection_name}_stats")]
        )
        
        if not result or len(result) == 0:
            return {"doc_count": 0, "avgdls": {}}
        
        payload = result[0].payload
        return payload.get("stats", {"doc_count": 0, "avgdls": {}})
    
    async def set_collection_stats(self, collection_name: str, stats: Dict[str, Union[int, Dict[str, float]]]) -> None:
        await self.client.upsert(
            collection_name=self.meta_collection,
            points=[
                rest.PointStruct(
                    id=self._string_to_uuid(f"{collection_name}_stats"),
                    payload={
                        "stats": stats
                    },
                    vector={"dummy": [0.0]}
                )
            ]
        )
    
    async def get_collection_info(self, collection_name: str) -> CollectionConfig:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionInfo: Collection information
        """
        internal_config = await self.get_collection_info_internal(collection_name)
        
        # Convert to user-facing CollectionConfig (remove internal fields)
        return internal_to_user_config(internal_config)
    
    async def is_connected(self) -> bool:
        """
        Check if the Qdrant database connection is active and healthy.
        
        Returns:
            bool: True if connected and healthy, False otherwise
        """
        try:
            # Simple health check - just test if we can get basic instance info
            # This is lightweight and just returns version/commit info
            info = await self.client.info()
            return True
        except Exception as e:
            self.logger.debug(f"is_connected check failed: {str(e)}")
            return False
            
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
        current_time = datetime.now(timezone.utc)
        points = []
        queue_ids = []

        for document in documents:
            queue_id = str(uuid.uuid4())
            queue_ids.append(queue_id)
            
            queue_doc = QueueDocument(
                queue_id=queue_id,
                collection_name=collection_name,
                collection_id=collection_id,
                doc_id=document.id,
                op_type=op_type,
                doc_timestamp=request_timestamp if op_type == QueueOperationType.DELETE else document.timestamp,
                status=QueuedDocumentStatus.QUEUED,
                document=document,
                info=None,
                created_at=current_time,
                timestamp=current_time,
                try_count=0
            )
            
            payload = queue_doc.model_dump()
            
            points.append(
                rest.PointStruct(
                    id=queue_id,
                    payload=payload,
                    vector={"dummy": [0.0]}
                )
            )
        
        await self.client.upsert(
            collection_name=self.queue_collection,
            points=points
        )
        
        return queue_ids
    
    async def get_from_queue(self, queue_ids: List[str], suppress_not_found: bool = False) -> List['QueueDocument']:
        """
        Retrieve documents from the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
            suppress_not_found: If True, return only found entries instead of raising AmgixNotFound
            
        Returns:
            List[QueueDocument]: List of queue documents with status and metadata
        """
        # Retrieve from the system queue collection using queue_ids
        result = await self.client.retrieve(
            collection_name=self.queue_collection,
            ids=queue_ids
        )
        
        # Check if we got the expected number of results
        if len(result) != len(queue_ids) and not suppress_not_found:
            found_ids = {point.id for point in result}
            missing_ids = set(queue_ids) - found_ids
            raise AmgixNotFound(f"Queue documents not found for queue_ids: {', '.join(missing_ids)}")
        
        # Convert payloads back to QueueDocument list
        queue_docs = []
        for point in result:
            queue_docs.append(QueueDocument(**point.payload))
        
        return queue_docs
    
    async def delete_from_queue(self, queue_ids: List[str]) -> None:
        """
        Remove documents from the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
        """
        if not queue_ids:
            return
            
        await self.client.delete(
            collection_name=self.queue_collection,
            points_selector=rest.PointIdsList(
                points=queue_ids
            )
        )
    
    async def delete_from_queue_by_collection(self, collection_name: str) -> None:
        """
        Remove all documents from the processing queue for a specific collection.
        
        Args:
            collection_name: Name of the collection to clear from queue
        """
        # Delete from the system queue collection using collection_name filter
        await self.client.delete(
            collection_name=self.queue_collection,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="collection_name",
                        match=rest.MatchValue(value=collection_name)
                    )
                ]
            )
        )

    async def delete_upserts_from_queue(self, collection_name: str, doc_id: str, before_timestamp: datetime) -> None:
        await self.client.delete(
            collection_name=self.queue_collection,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="collection_name",
                        match=rest.MatchValue(value=collection_name)
                    ),
                    rest.FieldCondition(
                        key="doc_id",
                        match=rest.MatchValue(value=doc_id)
                    ),
                    rest.FieldCondition(
                        key="op_type",
                        match=rest.MatchValue(value=QueueOperationType.UPSERT)
                    ),
                    rest.FieldCondition(
                        key="doc_timestamp",
                        range=rest.DatetimeRange(lte=before_timestamp)
                    ),
                ]
            )
        )

    async def update_queue_status(self, queue_ids: List[str], status: str, try_count: int, info: str) -> None:
        """
        Update the status of documents in the processing queue.
        
        Args:
            queue_ids: List of unique identifiers for queue entries
            status: New status to set for the documents
            try_count: New try count to set for the documents
            info: Additional information about the status (e.g., error details)
        """
        # Update status, timestamp, try_count, and info in the system queue collection
        await self.client.set_payload(
            collection_name=self.queue_collection,
            payload={
                "status": status,
                "timestamp": datetime.now(timezone.utc),
                "try_count": try_count,
                "info": info
            },
            points=queue_ids
        )
        

    async def get_queue_entries(self, collection_name: str, doc_id: Optional[str] = None) -> List[QueueDocument]:
        """
        Get all queue entries for a specific document or all queue entries for a collection.
        
        Args:
            collection_name: Name of the collection this document belongs to
            doc_id: Unique identifier for the document, or None to get all queue entries for the collection
            
        Returns:
            List[QueueDocument]: List of queue entries for this document or collection
        """
        # Build filter conditions based on whether doc_id is provided
        must_conditions = [
            rest.FieldCondition(
                key="collection_name",
                match=rest.MatchValue(value=collection_name)
            )
        ]
        
        if doc_id is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="doc_id", 
                    match=rest.MatchValue(value=doc_id)
                )
            )
        
        # Query queue collection by payload fields using indexed search
        # Use projection to only fetch the fields we actually need, avoiding full document payload
        # Order by doc_id and timestamp for consistent results
        result = await self.client.scroll(
            collection_name=self.queue_collection,
            scroll_filter=rest.Filter(must=must_conditions),
            with_payload=True,
            with_vectors=False,  # We don't need vectors for queue entries
            order_by=rest.OrderBy(key="timestamp", direction="asc")
            # limit=100  # Reasonable limit for queue entries
        )
        
        # Convert results to QueueDocument objects with only the fields we use
        queue_entries = []
        for point in result[0]:
            payload = point.payload

            # Create QueueDocument with only the fields we actually use
            # Avoid loading the full document object into memory
            queue_doc = QueueDocument(
                queue_id=point.id,
                collection_name=collection_name,  # We know this from the method parameter
                collection_id=payload.get("collection_id"),
                doc_id=payload.get("doc_id"),
                op_type=payload.get("op_type"),
                doc_timestamp=payload.get("doc_timestamp"),
                status=payload.get("status"),
                info=payload.get("info"),
                document=None,                   # Not used upstream, set to None
                created_at=payload.get("created_at"), 
                timestamp=payload.get("timestamp"),
                try_count=payload.get("try_count", 0)
            )
            queue_entries.append(queue_doc)
        
        return queue_entries
    
    async def get_queue_info(self, collection_name: str) -> QueueInfo:
        """
        Get queue statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            QueueInfo: Counts of documents in each queue state
        """
        counts = {}
        
        # Only query for statuses that exist in the queue (INDEXED documents are removed from queue)
        queue_statuses = [s for s in QueuedDocumentStatus.all() if s != QueuedDocumentStatus.INDEXED]
        
        for status in queue_statuses:
            result = await self.client.count(
                collection_name=self.queue_collection,
                count_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="collection_name",
                            match=rest.MatchValue(value=collection_name)
                        ),
                        rest.FieldCondition(
                            key="status",
                            match=rest.MatchValue(value=status)
                        )
                    ]
                )
            )
            counts[status] = result.count
        
        total = sum(counts.values())
        
        return QueueInfo(
            queued=counts[QueuedDocumentStatus.QUEUED],
            requeued=counts[QueuedDocumentStatus.REQUEUED],
            failed=counts[QueuedDocumentStatus.FAILED],
            total=total
        )
    
    async def get_queue_statuses(self, collection_name: str, doc_id: str) -> DocumentStatusResponse:
        """
        Get comprehensive status of a document including collection and queue states.
        
        Args:
            collection_name: Name of the collection
            doc_id: Document identifier
            
        Returns:
            DocumentStatusResponse: Complete status information with empty list if no statuses found
        """
        statuses = []
        
        # 1. Check if document is indexed in collection
        docs = (await self.get_documents(collection_name, [doc_id], suppress_not_found=True))
        if docs and docs[0] is not None:
            statuses.append(DocumentStatus(
                status=QueuedDocumentStatus.INDEXED,
                timestamp=docs[0].timestamp
            ))
        
        # 2. Get all queue entries for this doc_id
        queue_entries = await self.get_queue_entries(collection_name, doc_id)
        for entry in queue_entries:
            statuses.append(DocumentStatus(
                status=entry.status,
                info=entry.info,
                timestamp=entry.timestamp,
                queue_id=entry.queue_id,
                try_count=entry.try_count
            ))
        
        # 3. Sort by timestamp (newest first)
        statuses.sort(key=lambda x: x.timestamp, reverse=True)
        
        return DocumentStatusResponse(
            statuses=statuses
        )

    async def append_metric_buckets(self, hostname: str, source: str, buckets: List[MetricsBucket]) -> None:
        if not buckets:
            return

        points = []
        for bucket in buckets:
            # Stable ID so re-inserting the same bucket is an upsert (idempotent on retry).
            identity = f"{hostname}:{source}:{bucket.key}:{':'.join(bucket.dims)}:{bucket.bucket_start}:{bucket.bucket_seconds}"
            point_id = self._string_to_uuid(identity)
            points.append(rest.PointStruct(
                id=point_id,
                vector={"dummy": [0.0]},
                payload={
                    "hostname": hostname,
                    "source": source,
                    "key": bucket.key,
                    "dims": bucket.dims,
                    "bucket_start": bucket.bucket_start,
                    "bucket_seconds": bucket.bucket_seconds,
                    "value": bucket.value,
                    "n": bucket.n,
                },
            ))

        await self.client.upsert(
            collection_name=self.metrics_collection,
            points=points,
        )

    async def trim_metric_buckets(self, bucket_seconds: int, cutoff: float) -> None:
        await self.client.delete(
            collection_name=self.metrics_collection,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="bucket_seconds",
                        match=rest.MatchValue(value=bucket_seconds),
                    ),
                    rest.FieldCondition(
                        key="bucket_start",
                        range=rest.Range(lt=cutoff),
                    ),
                ]
            ),
        )

    async def query_metric_buckets(
        self,
        bucket_seconds: int,
        since: float,
        until: float,
        keys: Optional[List[str]] = None,
    ) -> List[MetricsBucket]:
        must = [
            rest.FieldCondition(
                key="bucket_seconds",
                match=rest.MatchValue(value=bucket_seconds),
            ),
            rest.FieldCondition(
                key="bucket_start",
                range=rest.Range(gte=since, lt=until),
            ),
        ]
        if keys:
            must.append(rest.FieldCondition(
                key="key",
                match=rest.MatchAny(any=keys),
            ))

        scroll_filter = rest.Filter(must=must)
        order_by = rest.OrderBy(key="bucket_start", direction=rest.Direction.ASC)

        results: List[MetricsBucket] = []
        offset = None
        while True:
            points, next_offset = await self.client.scroll(
                collection_name=self.metrics_collection,
                scroll_filter=scroll_filter,
                order_by=order_by,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                p = point.payload
                results.append(MetricsBucket(
                    key=p["key"],
                    dims=p.get("dims", []),
                    bucket_start=p["bucket_start"],
                    bucket_seconds=p["bucket_seconds"],
                    value=p["value"],
                    n=p.get("n"),
                ))
            if next_offset is None:
                break
            offset = next_offset

        return results

    async def validate_features(self, config: CollectionConfig) -> None:
        """
        Check if the database supports all features required by the collection configuration.
        
        Args:
            config: Collection configuration to validate against database capabilities
        """
        # Qdrant supports all vector types and features, so no validation needed
        pass
