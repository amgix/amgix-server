from __future__ import annotations

import asyncio
import argparse
import logging
import os
import signal
import time
import random
from typing import List, Dict, Any

from src.core.vector.vectorizer import Vectorizer
from src.core.common.constants import APP_NAME, APP_PREFIX, RPC_TIMEOUT_SECONDS
from src.core.common import CACHE_BASE_DIR, HF_CACHE_DIR, CUDA_CACHE_DIR, get_user_collection_name

from .encoder_base import EncoderBase, EncoderServiceRunner
from .embed_router_service import EmbedRouterService

from src.core.models.vector import CollectionConfigInternal, SearchQuery, VectorConfigInternal, VectorSearchWeight, ModelValidationResponse, ModelValidationResult
from src.core.common.metrics_definitions import MetricKey
from src.core.common.metrics_service import MetricsService
from src.core.common import VectorType, QueuedDocumentStatus, QueueOperationType, MAX_QUEUE_DELIVERY_ATTEMPTS, MAX_DB_RETRIES
from src.core.models.document import Document, SearchResult, QueueDocument
from src.core.common.bunny_talk import BunnyTalk
from src.core.common.lock_manager import LockService, LockClient
from src.core.database.base import AmgixNotFound
from src.core.database.common import validate_metadata_filter
from datetime import datetime, timezone

# Set HuggingFace Hub etag timeout to 2s for faster fallback to cache
os.environ['HF_HUB_ETAG_TIMEOUT'] = '2'

AMGIX_VERSION = os.getenv("AMGIX_VERSION", "1.0.0-dev")

HOSTNAME = os.getenv('HOSTNAME', 'unknown')
AMGIX_AMQP_URL = os.getenv("AMGIX_AMQP_URL", "pyamqp://guest:guest@rabbitmq//")
AMGIX_ENCODER_ROLE = os.getenv("AMGIX_ENCODER_ROLE", "all")


# Cap the backoff sleep to avoid excessively long sleeps
_MAX_RETRY_SLEEP_SECONDS = 20.0

class VectorizationException(Exception):
    """Exception raised when vectorization fails."""
    pass

class EncoderService(EncoderBase):
    def __init__(self, logger, database, bunny_talk, router=None, lock_client=None):
        super().__init__(logger, database, bunny_talk, router, lock_client)
        self.index_metrics = MetricsService(
            amqp_url=AMGIX_AMQP_URL,
            logger=self.logger,
            hostname=HOSTNAME,
            source="index",
            role=AMGIX_ENCODER_ROLE,
            windows=[30, 60],
            database=database,
        )
    
    async def startup(self):
        await self.bunny_talk.listen(
            routing_key="documents",
            handler=self.document_upsert,
            prefetch_count=4
        )
        await self.bunny_talk.listen(
            routing_key="documents-bulk",
            handler=self.document_upsert_bulk,
            prefetch_count=1
        )
        await self.bunny_talk.listen(
            routing_key="collection-stats",
            handler=self.update_collection_stats,
            prefetch_count=1,
            single_active_consumer=True
        )
        await self.bunny_talk.listen(
            routing_key="document-delete-sync",
            handler=self.document_delete_sync,
            prefetch_count=2
        )
        await self.bunny_talk.listen(
            routing_key="ping-encoder",
            handler=self.ping
        )
        self.index_metrics.start_reporting()
        self.logger.info("Registered Encoder handlers (document_upsert, document_upsert_bulk, document_delete_sync)")

    async def ping(self) -> bool:
        return True

    async def document_upsert(self, queue_id: str) -> None:
        try_count = 0  # Initialize try_count outside try/except
        t0 = time.perf_counter_ns()

        self.bunny_talk.log_trace_context(f"EncoderService: queue_id {queue_id}")

        try:
            # Get the document from the queue
            try:
                queue_docs = await self.database.get_from_queue([queue_id])
            except AmgixNotFound as e:
                self.logger.error(str(e))
                return
            queue_doc = queue_docs[0]
            document = queue_doc.document
            collection_name = queue_doc.collection_name
            doc_id = queue_doc.doc_id
            try_count = queue_doc.try_count  # Set try_count from queue document

            if queue_doc.op_type == QueueOperationType.DELETE:
                await self.document_delete_sync(collection_name, doc_id, queue_doc.doc_timestamp)
                try:
                    await self.database.delete_from_queue([queue_id])
                except Exception as queue_error:
                    self.logger.error(f"Failed to delete queue entry {queue_id} after delete: {str(queue_error)}")
                return

            # Retrieve collection config (cached) and vectorize
            collection_config, from_cache = await EncoderBase.get_collection_info_cached(self.database, collection_name)
            if not collection_config:
                error_msg = f"Collection configuration not found for {get_user_collection_name(collection_name)}"
                self.logger.error(error_msg)
                new_try_count = try_count + 1
                try:
                    await self.database.update_queue_status([queue_id], QueuedDocumentStatus.FAILED, new_try_count, error_msg)
                except Exception as queue_error:
                    self.logger.error(f"Failed to update queue status for queue entry {queue_id}: {str(queue_error)}")
                return
            
            # CRITICAL: Verify collection_id matches to prevent processing against changed collection config
            if collection_config.collection_id != queue_doc.collection_id:
                # If config came from cache, invalidate and retry once
                if from_cache:
                    self.logger.warning(
                        f"Collection id mismatch for {collection_name} (queued {queue_doc.collection_id} vs cached {collection_config.collection_id}); invalidating cache and refetching."
                    )
                    EncoderBase.invalidate_collection_cache(collection_name)
                    collection_config, _ = await EncoderBase.get_collection_info_cached(self.database, collection_name)
                
                # Re-check after potential refresh
                error_msg = None
                if not collection_config:
                    error_msg = f"Collection configuration not found for {collection_name}"
                elif collection_config.collection_id != queue_doc.collection_id:
                    error_msg = (
                        f"Collection configuration changed during processing. Document was queued for collection_id {queue_doc.collection_id}, "
                        f"but current collection has collection_id {collection_config.collection_id}. Please re-upload the document."
                    )
                
                if error_msg:
                    self.logger.error(error_msg)
                    new_try_count = try_count + 1
                    try:
                        await self.database.update_queue_status([queue_id], QueuedDocumentStatus.FAILED, new_try_count, error_msg)
                    except Exception as queue_error:
                        self.logger.error(f"Failed to update queue status for queue entry {queue_id}: {str(queue_error)}")
                    return

            # Use distributed locking to prevent race conditions in multi-encoder setup
            lock_name = f"doc-{self.database._string_to_uuid(f"{collection_name}-{doc_id}")}"
            async with self.lock_client.acquire(lock_name, timeout=5.0):
                # Verify queue entry still exists — a concurrent delete may have drained it
                if not await self.database.get_from_queue([queue_id], suppress_not_found=True):
                    self.logger.info(f"Queue entry {queue_id} already removed (concurrent delete), skipping upsert")
                    return

                # Check if document already exists (now safely locked)
                existing_document = (await self.database.get_documents(collection_name, [doc_id], suppress_not_found=True))[0]
                
                if existing_document is not None:
                    # Compare timestamps - only upsert if new document is newer
                    if document.timestamp <= existing_document.timestamp:
                        # Skip upsert - existing document is newer or same timestamp
                        self.logger.info(f"Skipping upsert for document {doc_id}: existing timestamp {existing_document.timestamp} >= new timestamp {document.timestamp}")
                        # Delete from queue since processing is complete
                        try:
                            await self.database.delete_from_queue([queue_id])
                        except Exception as queue_error:
                            self.logger.error(f"Failed to delete queue entry {queue_id} after successful processing: {str(queue_error)}")
                        self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_SKIPPED_STALE)
                        return
                
                try:
                    stats = await self.database.get_collection_stats(collection_name)
                    avgdls = stats.get("avgdls", {})
                    
                    for config in collection_config.vectors:
                        if config.type in VectorType.custom_tokenization():
                            for field in config.index_fields:
                                field_vector_name = f"{field}_{config.name}"
                                if field_vector_name not in avgdls:
                                    avgdls[field_vector_name] = 50.0
                    
                    avgdl_dict = avgdls
                    
                    # Generate vectors synchronously
                    result = await Vectorizer.vectorize_documents(self.router, [document], collection_config.vectors, avgdl_dict=avgdl_dict)
                    doc_with_vectors = result[0]
                except Exception as e:
                    # Re-raise as VectorizationException to distinguish from database errors
                    raise VectorizationException(f"Vectorization failed: {str(e)}") from e
                
                is_new = existing_document is None
                await self.database.add_documents(collection_name, [doc_with_vectors], is_new=is_new, store_content=collection_config.store_content, collection_config=collection_config, lock_client=self.lock_client)
                
                updates: Dict[str, Dict[str, int]] = {}
                for field_vector_name, token_length in doc_with_vectors.token_lengths.items():
                    if field_vector_name not in updates:
                        updates[field_vector_name] = {
                            "new_doc_count": 1 if is_new else 0,
                            "new_sum_token_lengths": 0,
                            "update_doc_count": 0 if is_new else 1,
                            "update_sum_token_lengths": 0,
                            "old_sum_token_lengths": 0
                        }
                    
                    if is_new:
                        updates[field_vector_name]["new_sum_token_lengths"] += token_length
                    else:
                        updates[field_vector_name]["update_sum_token_lengths"] += token_length
                        if existing_document and field_vector_name in existing_document.token_lengths:
                            updates[field_vector_name]["old_sum_token_lengths"] += existing_document.token_lengths[field_vector_name]
                
                if is_new:
                    self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_NEW)
                else:
                    self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_UPDATED)

                if updates:
                    await self.bunny_talk.talk("collection-stats", collection_name=collection_name, updates=updates)
            
            # Processing successful - delete from queue
            try:
                await self.database.delete_from_queue([queue_id])
            except Exception as queue_error:
                self.logger.error(f"Failed to delete queue entry {queue_id} after successful processing: {str(queue_error)}")
            # self.logger.info(f"Successfully processed document {doc_id} for collection {collection_name}")
            
        except Exception as e:
            # Log the full error for debugging
            self.logger.error(f"Failed to process queue entry {queue_id}: {str(e)}")
            
            # Update queue status with error message (no stack trace)
            error_msg = str(e)
            new_try_count = try_count + 1
            
            # Validation errors (ValueError) should fail immediately - no retries
            if isinstance(e, ValueError):
                status = QueuedDocumentStatus.FAILED
            elif isinstance(e, VectorizationException):
                # Vectorization error: cap by MAX_QUEUE_DELIVERY_ATTEMPTS
                status = (
                    QueuedDocumentStatus.REQUEUED
                    if new_try_count < MAX_QUEUE_DELIVERY_ATTEMPTS
                    else QueuedDocumentStatus.FAILED
                )
            else:
                # Non-vectorization error: retry until age cap
                status = (
                    QueuedDocumentStatus.REQUEUED
                    if new_try_count < MAX_DB_RETRIES
                    else QueuedDocumentStatus.FAILED
                )
                
            try:
                await self.database.update_queue_status([queue_id], status, new_try_count, error_msg)
                if status == QueuedDocumentStatus.FAILED:
                        self.index_metrics.record(MetricKey.INDEX_QUEUE_FAILED)
                elif status == QueuedDocumentStatus.REQUEUED:
                        self.index_metrics.record(MetricKey.INDEX_QUEUE_REQUEUED)
            except Exception as queue_error:
                self.logger.error(f"Failed to update queue status for queue entry {queue_id}: {str(queue_error)}")

            if status == QueuedDocumentStatus.REQUEUED:
                delay = min(2 * new_try_count + random.uniform(0.1, 1.5), _MAX_RETRY_SLEEP_SECONDS)
                await asyncio.sleep(delay)
                raise e
        finally:
            self.index_metrics.record(MetricKey.INDEX_QUEUE_JOB_MS, (time.perf_counter_ns() - t0) / 1_000_000.0, n=1)

    async def document_upsert_bulk(self, queue_ids: List[str]) -> None:
        if not queue_ids:
            return

        self.bunny_talk.log_trace_context(f"EncoderService: bulk processing {len(queue_ids)} queue_ids")

        t0 = time.perf_counter_ns()
        try:
            try:
                queue_docs = await self.database.get_from_queue(queue_ids)
            except AmgixNotFound as e:
                self.logger.error(str(e))
                return

            collection_name = queue_docs[0].collection_name

            try:
                collection_config, from_cache = await EncoderBase.get_collection_info_cached(self.database, collection_name)
                if not collection_config:
                    error_msg = f"Collection configuration not found for {get_user_collection_name(collection_name)}"
                    self.logger.error(error_msg)
                    new_try_count = queue_docs[0].try_count + 1
                    try:
                        await self.database.update_queue_status(queue_ids, QueuedDocumentStatus.FAILED, new_try_count, error_msg)
                    except Exception as queue_error:
                        self.logger.error(f"Failed to update queue status: {str(queue_error)}")
                    return

                if collection_config.collection_id != queue_docs[0].collection_id:
                    if from_cache:
                        self.logger.warning(
                            f"Collection id mismatch for {get_user_collection_name(collection_name)}; invalidating cache and refetching."
                        )
                        EncoderBase.invalidate_collection_cache(collection_name)
                        collection_config, _ = await EncoderBase.get_collection_info_cached(self.database, collection_name)

                    if not collection_config:
                        error_msg = f"Collection configuration not found for {get_user_collection_name(collection_name)}"
                        new_try_count = queue_docs[0].try_count + 1
                        try:
                            await self.database.update_queue_status(queue_ids, QueuedDocumentStatus.FAILED, new_try_count, error_msg)
                        except Exception as queue_error:
                            self.logger.error(f"Failed to update queue status: {str(queue_error)}")
                        return

                    if collection_config.collection_id != queue_docs[0].collection_id:
                        error_msg = (
                            f"Collection configuration changed during processing. Documents were queued for collection_id {queue_docs[0].collection_id}, "
                            f"but current collection has collection_id {collection_config.collection_id}. Please re-upload the documents."
                        )
                        new_try_count = queue_docs[0].try_count + 1
                        try:
                            await self.database.update_queue_status(queue_ids, QueuedDocumentStatus.FAILED, new_try_count, error_msg)
                        except Exception as queue_error:
                            self.logger.error(f"Failed to update queue status: {str(queue_error)}")
                        return

                # Acquire all document locks in single batch
                lock_names = [f"doc-{self.database._string_to_uuid(f"{collection_name}-{qd.doc_id}")}" for qd in queue_docs]
                async with self.lock_client.acquire(lock_names, timeout=5.0):

                    # Re-verify queue entries still exist — concurrent deletes may have drained some
                    still_queued = await self.database.get_from_queue([qd.queue_id for qd in queue_docs], suppress_not_found=True)
                    still_queued_ids = {qd.queue_id for qd in still_queued}
                    drained = [qd.queue_id for qd in queue_docs if qd.queue_id not in still_queued_ids]
                    if drained:
                        self.logger.info(f"Bulk: {len(drained)} queue entries already removed (concurrent delete), skipping: {drained}")
                    queue_docs = [qd for qd in queue_docs if qd.queue_id in still_queued_ids]
                    queue_ids = [qd.queue_id for qd in queue_docs]
                    if not queue_docs:
                        return

                    existing_docs = await self.database.get_documents(collection_name, [qd.doc_id for qd in queue_docs], suppress_not_found=True)

                    existing_docs_map = {}
                    for queue_doc, existing_doc in zip(queue_docs, existing_docs):
                        if existing_doc is not None:
                            existing_docs_map[queue_doc.doc_id] = existing_doc

                    documents_to_process = []
                    documents_to_skip = []
                    is_new_flags = []
                
                    for queue_doc in queue_docs:
                        existing_doc = existing_docs_map.get(queue_doc.doc_id)
                        if existing_doc is not None and queue_doc.document.timestamp <= existing_doc.timestamp:
                            documents_to_skip.append(queue_doc)
                        else:
                            documents_to_process.append(queue_doc)
                            is_new_flags.append(existing_doc is None)

                    if documents_to_skip:
                        self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_SKIPPED_STALE, float(len(documents_to_skip)))
                        await self.database.delete_from_queue([qd.queue_id for qd in documents_to_skip])

                    if not documents_to_process:
                        return

                    documents = [qd.document for qd in documents_to_process]
                
                    stats = await self.database.get_collection_stats(collection_name)
                    avgdls = stats.get("avgdls", {})
                
                    for config in collection_config.vectors:
                        if config.type in VectorType.custom_tokenization():
                            for field in config.index_fields:
                                field_vector_name = f"{field}_{config.name}"
                                if field_vector_name not in avgdls:
                                    avgdls[field_vector_name] = 50.0
                
                    avgdl_dict = avgdls
                
                    docs_with_vectors = await Vectorizer.vectorize_documents(self.router, documents, collection_config.vectors, avgdl_dict=avgdl_dict)

                    # Count docs once for the batch
                    new_doc_count_batch = sum(1 for is_new in is_new_flags if is_new)
                    update_doc_count_batch = len(is_new_flags) - new_doc_count_batch

                    updates: Dict[str, Dict[str, int]] = {}
                    for doc_idx, doc_with_vectors in enumerate(docs_with_vectors):
                        is_new = is_new_flags[doc_idx]
                        doc_id = documents_to_process[doc_idx].doc_id
                        existing_doc = existing_docs_map.get(doc_id)
                    
                        for field_vector_name, token_length in doc_with_vectors.token_lengths.items():
                            if field_vector_name not in updates:
                                updates[field_vector_name] = {
                                    "new_doc_count": new_doc_count_batch,
                                    "new_sum_token_lengths": 0,
                                    "update_doc_count": update_doc_count_batch,
                                    "update_sum_token_lengths": 0,
                                    "old_sum_token_lengths": 0
                                }
                        
                            if is_new:
                                updates[field_vector_name]["new_sum_token_lengths"] += token_length
                            else:
                                updates[field_vector_name]["update_sum_token_lengths"] += token_length
                                if existing_doc and field_vector_name in existing_doc.token_lengths:
                                    updates[field_vector_name]["old_sum_token_lengths"] += existing_doc.token_lengths[field_vector_name]

                    new_docs = []
                    existing_docs_list = []
                    for i, is_new in enumerate(is_new_flags):
                        if is_new:
                            new_docs.append(docs_with_vectors[i])
                        else:
                            existing_docs_list.append(docs_with_vectors[i])

                    if new_docs:
                        await self.database.add_documents(collection_name, new_docs, is_new=True, store_content=collection_config.store_content, collection_config=collection_config, lock_client=self.lock_client)
                    if existing_docs_list:
                        await self.database.add_documents(collection_name, existing_docs_list, is_new=False, store_content=collection_config.store_content, collection_config=collection_config, lock_client=self.lock_client)
                
                    if updates:
                        await self.bunny_talk.talk("collection-stats", collection_name=collection_name, updates=updates)
                        # await self.update_collection_stats(collection_name=collection_name, updates=updates)
                
                    await self.database.delete_from_queue([qd.queue_id for qd in documents_to_process])

                    if new_doc_count_batch:
                        self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_NEW, float(new_doc_count_batch))
                    if update_doc_count_batch:
                        self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_UPDATED, float(update_doc_count_batch))
                    nq = len(queue_ids)
                    if nq > 0:
                        self.index_metrics.record(MetricKey.INDEX_BULK_BATCHES)
                        self.index_metrics.record(MetricKey.INDEX_BULK_BATCH_SIZE, float(nq), n=1)

            except Exception as e:
                self.logger.error(f"Failed to process bulk documents: {str(e)}")
                
                new_try_count = queue_docs[0].try_count + 1
                
                if isinstance(e, ValueError):
                    status = QueuedDocumentStatus.FAILED
                elif isinstance(e, VectorizationException):
                    status = (
                        QueuedDocumentStatus.REQUEUED
                        if new_try_count < MAX_QUEUE_DELIVERY_ATTEMPTS
                        else QueuedDocumentStatus.FAILED
                    )
                else:
                    status = (
                        QueuedDocumentStatus.REQUEUED
                        if new_try_count < MAX_DB_RETRIES
                        else QueuedDocumentStatus.FAILED
                    )
                    
                try:
                    await self.database.update_queue_status(queue_ids, status, new_try_count, str(e))
                    if status == QueuedDocumentStatus.FAILED:
                        self.index_metrics.record(MetricKey.INDEX_BULK_FAILED)
                    elif status == QueuedDocumentStatus.REQUEUED:
                        self.index_metrics.record(MetricKey.INDEX_BULK_REQUEUED)
                except Exception as queue_error:
                    self.logger.error(f"Failed to update queue status: {str(queue_error)}")

                if status == QueuedDocumentStatus.REQUEUED:
                    delay = min(2 * new_try_count + random.uniform(0.1, 1.5), _MAX_RETRY_SLEEP_SECONDS)
                    await asyncio.sleep(delay)
                    raise e
        finally:
            self.index_metrics.record(MetricKey.INDEX_BULK_JOB_MS, (time.perf_counter_ns() - t0) / 1_000_000.0, n=1)

    async def update_collection_stats(
        self, 
        collection_name: str, 
        updates: Dict[str, Dict[str, int]]
    ) -> None:
        """
        Update collection statistics for multiple field_vector_names.
        
        Args:
            collection_name: Name of the collection
            updates: Dictionary mapping field_vector_name to {
                "new_doc_count": int,
                "new_sum_token_lengths": int,
                "update_doc_count": int,
                "update_sum_token_lengths": int,
                "old_sum_token_lengths": int
            }
        """

        # lock_name = f"collection-stats-{self.database._string_to_uuid(collection_name)}"
        # async with self.lock_client.acquire(lock_name, timeout=10.0):
        
        stats = await self.database.get_collection_stats(collection_name)

        old_doc_count = stats.get("doc_count", 0)
        avgdls = stats.get("avgdls", {})
        
        # All fields have the same doc_count values, so take from first field
        first_update = next(iter(updates.values()))
        new_docs_in_batch = first_update.get("new_doc_count", 0)
        new_doc_count = old_doc_count + new_docs_in_batch
        
        for field_vector_name, update_data in updates.items():
            old_avgdl = avgdls.get(field_vector_name, 0.0)
            
            new_sum_token_lengths = update_data.get("new_sum_token_lengths", 0)
            update_sum_token_lengths = update_data.get("update_sum_token_lengths", 0)
            old_sum_token_lengths = update_data.get("old_sum_token_lengths", 0)
            
            new_avgdl = (old_avgdl * old_doc_count - old_sum_token_lengths + new_sum_token_lengths + update_sum_token_lengths) / new_doc_count
            avgdls[field_vector_name] = new_avgdl
        
        await self.database.set_collection_stats(collection_name, {"doc_count": new_doc_count, "avgdls": avgdls})

    async def document_delete_sync(self, collection_name: str, document_id: str, request_timestamp: datetime) -> None:
        """Delete a document synchronously."""

        self.bunny_talk.log_trace_context(f"RpcService: document_id {document_id}")
        t0 = time.perf_counter_ns()
        try:
            # Use distributed locking to prevent race conditions in multi-encoder setup
            lock_name = f"doc-{self.database._string_to_uuid(f"{collection_name}-{document_id}")}"
            async with self.lock_client.acquire(lock_name, timeout=RPC_TIMEOUT_SECONDS):

                # Get document before deleting to retrieve token_lengths for stats update
                docs = await self.database.get_documents(collection_name, [document_id], suppress_not_found=True)
                doc_with_vectors = docs[0] if docs else None

                if doc_with_vectors is None:
                    self.logger.warning(f"Document {document_id} not found in {collection_name}, skipping delete")
                    return

                await self.database.delete_document(collection_name, document_id)
                await self.database.delete_upserts_from_queue(collection_name, document_id, request_timestamp)
                self.index_metrics.record(MetricKey.INDEX_QUEUE_DOCS_DELETED)

                # Send stats update with negative values
                updates = {}
                for field_vector_name, token_length in doc_with_vectors.token_lengths.items():
                    updates[field_vector_name] = {
                        "new_doc_count": -1,
                        "new_sum_token_lengths": -token_length,
                        "update_doc_count": 0,
                        "update_sum_token_lengths": 0,
                        "old_sum_token_lengths": 0
                    }

                if updates:
                    await self.bunny_talk.talk("collection-stats", collection_name=collection_name, updates=updates)
        finally:
            self.index_metrics.record(MetricKey.INDEX_QUEUE_DELETE_JOB_MS, (time.perf_counter_ns() - t0) / 1_000_000.0, n=1)


class RpcService(EncoderBase):
    """Service for handling RPC calls."""

    async def startup(self):
        await self.bunny_talk.listen(
            routing_key="search",
            handler=self.search,
            prefetch_count=2
        )
        await self.bunny_talk.listen(
            routing_key="validate-models",
            handler=self.validate_models
        )
        await self.bunny_talk.listen(
            routing_key="document-sync",
            handler=self.document_upsert_sync,
            prefetch_count=2
        )
        await self.bunny_talk.listen(
            routing_key="ping-rpc",
            handler=self.ping
        )
        self.logger.info("Registered RPC handlers (search, validate_models, document_sync)")

    async def ping(self) -> bool:
        return True


    async def search(self, collection_name: str, query: SearchQuery) -> List[SearchResult]:

        self.bunny_talk.log_trace_context(f"RpcService: search in {collection_name}")

        # Retrieve config (cached)
        collection_config, from_cache = await EncoderBase.get_collection_info_cached(self.database, collection_name)
        if not collection_config:
            error_msg = f"Collection configuration not found for {get_user_collection_name(collection_name)}"
            self.logger.error(error_msg)
            raise AmgixNotFound(error_msg)
        
        try:
            # Validate metadata filter against indexed metadata keys/types.
            if query.metadata_filter:
                validate_metadata_filter(collection_config, query.metadata_filter)

            # Run async operations safely using the background event loop
            query_with_vectors = await Vectorizer.vectorize_search_query(self.router, query, collection_config.vectors)
            results = await self.database.search(collection_name, query_with_vectors, collection_config)
            return results
        except Exception as e:
            # If config came from cache and operation failed, invalidate cache and retry once
            if from_cache:
                self.logger.warning(f"Search failed with cached config for {collection_name}, invalidating cache and retrying: {str(e)}")
                EncoderBase.invalidate_collection_cache(collection_name)
                # Refetch fresh config
                collection_config, _ = await EncoderBase.get_collection_info_cached(self.database, collection_name)
                if not collection_config:
                    error_msg = f"Collection configuration not found for {get_user_collection_name(collection_name)}"
                    self.logger.error(error_msg)
                    raise AmgixNotFound(error_msg)
                # Retry once with fresh config
                if query.metadata_filter:
                    validate_metadata_filter(collection_config, query.metadata_filter)
                query_with_vectors = await Vectorizer.vectorize_search_query(self.router, query, collection_config.vectors)
                results = await self.database.search(collection_name, query_with_vectors, collection_config)
                return results
            # If config was fresh or retry already failed, re-raise
            raise

    async def validate_models(self, vector_configs: List[VectorConfigInternal]) -> ModelValidationResponse:
        """Validate vector models and return dimensions for dense vectors."""
        try:
            
            self.bunny_talk.log_trace_context(f"RpcService: validate_models")
            
            # Create a dummy search query with all vector configs
            # Create VectorSearchWeight objects for each config
            vector_weights = []
            for config in vector_configs:
                for field in config.index_fields:
                    vector_weights.append(VectorSearchWeight(
                        vector_name=config.name,
                        weight=1.0,
                        field=field
                    ))
            
            dummy_query = SearchQuery(
                query="x",  # Minimal dummy text
                vector_weights=vector_weights
            )
            
            query_with_vectors = await Vectorizer.vectorize_search_query(self.router, dummy_query, vector_configs, validation_mode=True)
            
            results = {}
            for config in vector_configs:
                if config.type == VectorType.DENSE_MODEL:
                    # Find the dense vector for this config
                    dense_vector = None
                    for vector_data in query_with_vectors.vectors:
                        if vector_data.vector_name == config.name and vector_data.dense_vector is not None:
                            dense_vector = vector_data.dense_vector
                            break
                    
                    if dense_vector is not None:
                        results[config.name] = ModelValidationResult(
                            valid=True,
                            dimension=len(dense_vector)
                        )
                    else:
                        results[config.name] = ModelValidationResult(
                            valid=False,
                            error="No dense vector generated"
                        )
                elif config.type == VectorType.SPARSE_MODEL:
                    # Just confirm it worked (sparse vectors have variable dimensions)
                    results[config.name] = ModelValidationResult(valid=True)
                else:
                    # Other vector types (trigrams, full_text) don't need validation
                    results[config.name] = ModelValidationResult(valid=True)
            
            return ModelValidationResponse(results=results)
        except Exception as e:
            # Return error information for debugging
            return ModelValidationResponse(error=str(e))

    async def document_upsert_sync(self, collection_name: str, document: Document) -> None:
        """Upsert a document synchronously."""

        self.bunny_talk.log_trace_context(f"RpcService: document_id {document.id}")

        # Retrieve collection config and vectorize
        collection_config = await self.database.get_collection_info_internal(collection_name)
        if not collection_config:
            error_msg = f"Collection configuration not found"
            raise AmgixNotFound(error_msg)
        
        # Use distributed locking to prevent race conditions in multi-encoder setup
        lock_name = f"doc-{self.database._string_to_uuid(f"{collection_name}-{document.id}")}"
        async with self.lock_client.acquire(lock_name, timeout=RPC_TIMEOUT_SECONDS):
            # Check if document already exists (now safely locked)
            existing_document = (await self.database.get_documents(collection_name, [document.id], suppress_not_found=True))[0]
            
            if existing_document is not None:
                # Compare timestamps - only upsert if new document is newer
                if document.timestamp <= existing_document.timestamp:
                    # Skip upsert - existing document is newer or same timestamp
                    raise ValueError(f"Document {document.id} already exists with timestamp {existing_document.timestamp} >= new timestamp {document.timestamp}")
            
            stats = await self.database.get_collection_stats(collection_name)
            avgdls = stats.get("avgdls", {})
            
            for config in collection_config.vectors:
                if config.type in VectorType.custom_tokenization():
                    for field in config.index_fields:
                        field_vector_name = f"{field}_{config.name}"
                        if field_vector_name not in avgdls:
                            avgdls[field_vector_name] = 50.0
            
            avgdl_dict = avgdls
            
            # Generate vectors synchronously
            result = await Vectorizer.vectorize_documents(self.router, [document], collection_config.vectors, avgdl_dict=avgdl_dict)
            doc_with_vectors = result[0]
            
            is_new = existing_document is None
            await self.database.add_documents(collection_name, [doc_with_vectors], is_new=is_new, store_content=collection_config.store_content, collection_config=collection_config, lock_client=self.lock_client)
            
            updates: Dict[str, Dict[str, int]] = {}
            for field_vector_name, token_length in doc_with_vectors.token_lengths.items():
                if field_vector_name not in updates:
                    updates[field_vector_name] = {
                        "new_doc_count": 1 if is_new else 0,
                        "new_sum_token_lengths": 0,
                        "update_doc_count": 0 if is_new else 1,
                        "update_sum_token_lengths": 0,
                        "old_sum_token_lengths": 0
                    }
                
                if is_new:
                    updates[field_vector_name]["new_sum_token_lengths"] += token_length
                else:
                    updates[field_vector_name]["update_sum_token_lengths"] += token_length
                    if existing_document and field_vector_name in existing_document.token_lengths:
                        updates[field_vector_name]["old_sum_token_lengths"] += existing_document.token_lengths[field_vector_name]
            
            if updates:
                await self.bunny_talk.talk("collection-stats", collection_name=collection_name, updates=updates)


async def main():
    """Main entry point for the encoder service."""

    global AMGIX_AMQP_URL
    
    # Set up cache directories
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["CUDA_CACHE_PATH"] = CUDA_CACHE_DIR
    
    os.makedirs(CACHE_BASE_DIR, exist_ok=True)
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.makedirs(CUDA_CACHE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description=f"{APP_PREFIX} Service")
    parser.add_argument(
        "--service",
        choices=["index", "query", "all"],
        default="all",
        help="Encoder role: index (document ingestion), query (search/validation), or all (default)"
    )
    args = parser.parse_args()

    service_classes = []

    # must be first in the list because other services depend on it
    service_classes.append(EmbedRouterService)

    if args.service in ["index", "all"]:
        service_classes.append(EncoderService)

    if args.service in ["query", "all"]:
        service_classes.append(RpcService)

    if "?" not in AMGIX_AMQP_URL:
        AMGIX_AMQP_URL = f"{AMGIX_AMQP_URL}?name={APP_PREFIX}-node-{args.service}-{HOSTNAME}"

    logging.info(f"Starting {APP_NAME} Encoder v{AMGIX_VERSION}")

    service = EncoderServiceRunner(AMGIX_AMQP_URL)
    await service.run_forever(service_classes)


if __name__ == "__main__":
    asyncio.run(main())


