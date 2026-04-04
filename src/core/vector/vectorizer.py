from typing import List, Dict, Type, Union, Optional
import asyncio

from ..models.document import Document, DocumentWithVectors
from ..models.vector import VectorConfigInternal, SearchQuery, SearchQueryWithVectors, VectorData, VectorSearchWeight
from ..common import VectorType, EmbedRouter, WMTR_DEFAULT_TRIGRAM_WEIGHT
from .dense_custom import CustomDenseVector
from .sparse_custom import CustomSparseVector


class Vectorizer:
    """
    Vectorizer class that generates vectors for documents and search queries.
    
    This class manages different vector generation strategies based on VectorConfig
    and provides a unified interface for vector generation.
    """
    
    @staticmethod
    async def vectorize_documents(
        router: EmbedRouter,
        documents: List[Document],
        vector_configs: List[VectorConfigInternal],
        avgdl_dict: Optional[Dict[str, Dict[str, Union[int, float]]]] = None
    ) -> List[DocumentWithVectors]:
        """
        Generate vectors for a document based on the provided vector configurations.
        
        Args:
            document: The document to vectorize
            vector_configs: List of vector configurations to apply
            
        Returns:
            DocumentWithVectors: Document with pre-calculated vectors
            
        Raises:
            ValueError: If a vector type is not supported or configuration is invalid
        """
        vectors_per_doc: List[List[VectorData]] = [[] for _ in range(len(documents))]
        token_lengths_per_doc: List[Dict[str, int]] = [{} for _ in range(len(documents))]

        for config in vector_configs:
            try:
                if config.type == VectorType.DENSE_MODEL:
                    texts: List[str] = []
                    for doc in documents:
                        for field in config.index_fields:
                            texts.append(Vectorizer._get_field_text(doc, field))

                    dense_vectors = await router(
                        config,
                        texts,
                        trigram_weight=WMTR_DEFAULT_TRIGRAM_WEIGHT,
                    )

                    idx = 0
                    for doc_idx, _doc in enumerate(documents):
                        for field in config.index_fields:
                            dense_vector = dense_vectors[idx]
                            if config.dimensions is not None and len(dense_vector) != config.dimensions:
                                raise ValueError(f"Specified dimensions {config.dimensions} don't match generated dimensions {len(dense_vector)} for vector '{config.name}' field '{field}'")
                            vectors_per_doc[doc_idx].append(VectorData(
                                vector_name=config.name,
                                field=field,
                                vector_type=config.type,
                                dense_vector=dense_vector
                            ))
                            idx += 1

                elif config.type == VectorType.DENSE_CUSTOM:
                    per_doc = CustomDenseVector.extract_for_documents(config, documents)
                    for doc_idx, field_map in per_doc.items():
                        for field in config.index_fields:
                            vec = field_map.get(field)
                            if vec is None:
                                raise ValueError(f"Custom dense vector '{config.name}' for field '{field}' not provided")
                            vectors_per_doc[doc_idx].append(VectorData(
                                vector_name=config.name,
                                field=field,
                                vector_type=config.type,
                                dense_vector=vec
                            ))

                elif config.type == VectorType.SPARSE_CUSTOM:
                    per_doc = CustomSparseVector.extract_for_documents(config, documents)
                    for doc_idx, field_map in per_doc.items():
                        for field in config.index_fields:
                            pair = field_map.get(field)
                            if pair is None:
                                raise ValueError(f"Custom sparse vector '{config.name}' for field '{field}' not provided")
                            indices, values = pair
                            field_vector_name = f"{field}_{config.name}"
                            token_length = len(indices)
                            vectors_per_doc[doc_idx].append(VectorData(
                                vector_name=config.name,
                                field=field,
                                vector_type=config.type,
                                sparse_indices=indices,
                                sparse_values=values
                            ))
                            token_lengths_per_doc[doc_idx][field_vector_name] = token_length

                else:  # All other sparse vector types (SPARSE_MODEL, TRIGRAMS, FULL_TEXT, WHITESPACE, WMTR)
                    texts: List[str] = []
                    avgdls: List[float] = []
                    is_custom = config.type in VectorType.custom_tokenization()
                    for doc in documents:
                        for field in config.index_fields:
                            texts.append(Vectorizer._get_field_text(doc, field))
                            if is_custom:
                                field_vector_name = f"{field}_{config.name}"
                                avgdls.append(avgdl_dict[field_vector_name])

                    sparse_vectors = await router(
                        config,
                        texts,
                        avgdls=avgdls,
                        trigram_weight=WMTR_DEFAULT_TRIGRAM_WEIGHT,
                    )

                    idx = 0
                    for doc_idx, _doc in enumerate(documents):
                        for field in config.index_fields:
                            indices, values = sparse_vectors[idx]
                            field_vector_name = f"{field}_{config.name}"
                            token_length = len(indices)
                            vectors_per_doc[doc_idx].append(VectorData(
                                vector_name=config.name,
                                field=field,
                                vector_type=config.type,
                                sparse_indices=indices,
                                sparse_values=values
                            ))
                            if config.type in VectorType.sparse_types():
                                token_lengths_per_doc[doc_idx][field_vector_name] = token_length
                            idx += 1
            except Exception as e:
                # Preserve helpful error context by vector config
                if config.type in (VectorType.DENSE_CUSTOM, VectorType.SPARSE_CUSTOM):
                    # For custom types, errors include field-level messages above
                    raise
                raise ValueError(f"Failed to generate vector '{config.name}' for fields {config.index_fields}: {str(e)}") from e

        return [
            DocumentWithVectors(
                **doc.model_dump(),
                vectors=vectors_per_doc[i],
                token_lengths=token_lengths_per_doc[i]
            )
            for i, doc in enumerate(documents)
        ]
    
    @staticmethod
    async def vectorize_search_query(
        router: EmbedRouter,
        query: SearchQuery, 
        vector_configs: List[VectorConfigInternal],
        validation_mode: bool = False
    ) -> SearchQueryWithVectors:
        """
        Generate vectors for a search query based on the provided vector configurations.
        
        Only generates vectors for the specific vector_name + field combinations
        specified in query.vector_weights. If no vector_weights are specified,
        generates vectors for all available vector configurations.
        
        Args:
            query: The search query to vectorize
            vector_configs: List of vector configurations to apply
            validation_mode: If True, validates custom vector configs without processing custom vectors from query
            
        Returns:
            SearchQueryWithVectors: Search query with pre-calculated vectors
            
        Raises:
            ValueError: If a vector type is not supported or configuration is invalid
        """
        # Validate that search query is not empty
        if not query.query:
            raise ValueError("Search query cannot be empty")
        
        # In validation mode, validate custom vector configs first
        if validation_mode:
            for config in vector_configs:
                if config.type == VectorType.DENSE_CUSTOM:
                    if config.dimensions is None:
                        raise ValueError(f"Dense custom vector '{config.name}' requires dimensions to be specified")
                elif config.type == VectorType.SPARSE_CUSTOM:
                    if config.top_k is None:
                        raise ValueError(f"Sparse custom vector '{config.name}' requires top_k to be specified")
        
        vectors = []
        
        # Determine which vectors to generate based on query.vector_weights
        if query.vector_weights:
            # Generate only the vectors specified in the query, excluding weight 0 vectors
            vectors_to_generate = set()
            for weight in query.vector_weights:
                # Skip vectors with weight 0 - they contribute nothing
                if weight.weight != 0:
                    vectors_to_generate.add((weight.vector_name, weight.field))
        else:
            # No specific weights specified, generate all available vectors
            vectors_to_generate = set()
            for config in vector_configs:
                for field in config.index_fields:
                    vectors_to_generate.add((config.name, field))
            
            # Create equal weights for all generated vectors
            if vectors_to_generate:
                equal_weight = 1.0 / len(vectors_to_generate)
                query.vector_weights = [
                    VectorSearchWeight(
                        vector_name=name,
                        field=field,
                        weight=equal_weight
                    )
                    for name, field in vectors_to_generate
                ]
        
        # Create a map of configs by name for quick lookup
        config_map = {config.name: config for config in vector_configs}
        
        # Group by vector_name to avoid encoding the same text multiple times with same config
        vector_groups = {}
        for vector_name, field in vectors_to_generate:
            if vector_name not in vector_groups:
                vector_groups[vector_name] = []
            vector_groups[vector_name].append(field)
        
        # Process vector configs in parallel using asyncio.gather
        tasks = []
        vector_name_to_task = {}
        
        for vector_name, fields in vector_groups.items():
            config = config_map.get(vector_name)
            if not config:
                # Fail if vector config not found - this is a configuration error
                raise ValueError(f"Vector configuration '{vector_name}' not found. Available vectors: {list(config_map.keys())}")
            
            # Validate that all fields are configured for this vector
            for field in fields:
                if field not in config.index_fields:
                    raise ValueError(f"Field '{field}' is not configured for vector '{vector_name}'. Available fields: {config.index_fields}")
            
            # Create async task for vector generation
            task = asyncio.create_task(Vectorizer._generate_vector_for_query(router, config, query, fields, validation_mode))
            tasks.append(task)
            vector_name_to_task[task] = vector_name
        
        # Wait for all tasks to complete - if ANY task fails, the whole operation fails
        try:
            results = await asyncio.gather(*tasks)
            
            # Process results
            for i, task in enumerate(tasks):
                vector_name = vector_name_to_task[task]
                vector_data_list = results[i]
                vectors.extend(vector_data_list)
                
        except Exception as e:
            # Find which task failed and provide better error context
            for task in tasks:
                if task.done() and task.exception():
                    vector_name = vector_name_to_task[task]
                    raise ValueError(f"Failed to generate vector '{vector_name}': {str(task.exception())}") from task.exception()
            # If we can't identify the specific task, re-raise the original error
            raise
        
        # Create SearchQueryWithVectors
        return SearchQueryWithVectors(
            **query.model_dump(),
            vectors=vectors
        )
    
    @staticmethod
    async def _generate_vector_for_query(
        router: EmbedRouter,
        config: VectorConfigInternal, 
        query: SearchQuery, 
        fields: List[str], 
        validation_mode: bool
    ) -> List[VectorData]:
        """
        Generate vectors for a single vector config and query.
        This method is designed to be called in parallel by ThreadPoolExecutor.
        
        Args:
            config: The vector configuration to use
            query: The search query to vectorize
            fields: List of fields to generate vectors for
            validation_mode: If True, validates custom vector configs without processing custom vectors
            
        Returns:
            List[VectorData]: List of generated vector data for all fields
        """
        vector_data_list = []
        
        # Generate vector ONCE for this vector type, then reuse for all fields
        if config.type == VectorType.DENSE_MODEL:
            # Use query_model/query_revision if specified in config, otherwise use model/revision
            effective_config = config
            if config.query_model is not None:
                dumped = config.model_dump()
                dumped["model"] = config.query_model
                dumped["revision"] = config.query_revision
                effective_config = VectorConfigInternal(**dumped)

            # Generated dense vector - generate once, reuse for all fields
            result = await router(
                effective_config,
                [query.query],
                trigram_weight=WMTR_DEFAULT_TRIGRAM_WEIGHT,
            )
            dense_vector = result[0]
            
            # If dimensions were specified in config, validate they match detected dimensions
            if config.dimensions is not None and len(dense_vector) != config.dimensions:
                raise ValueError(f"Specified dimensions {config.dimensions} don't match generated dimensions {len(dense_vector)} for vector '{config.name}'")
            
            # Create VectorData for all fields using the same generated vector
            for field in fields:
                vector_data = VectorData(
                    vector_name=config.name,
                    field=field,
                    vector_type=config.type,
                    dense_vector=dense_vector
                )
                vector_data_list.append(vector_data)
                
        elif config.type == VectorType.DENSE_CUSTOM:
            if validation_mode:
                return vector_data_list
            vec = CustomDenseVector.extract_for_query(config, query)
            for field in fields:
                vector_data = VectorData(
                    vector_name=config.name,
                    field=field,
                    vector_type=config.type,
                    dense_vector=vec
                )
                vector_data_list.append(vector_data)
                
        elif config.type == VectorType.SPARSE_CUSTOM:
            if validation_mode:
                return vector_data_list
            indices, values = CustomSparseVector.extract_for_query(config, query)
            for field in fields:
                vector_data = VectorData(
                    vector_name=config.name,
                    field=field,
                    vector_type=config.type,
                    sparse_indices=indices,
                    sparse_values=values
                )
                vector_data_list.append(vector_data)
                
        else:  # All other sparse vector types (SPARSE_MODEL, TRIGRAMS, FULL_TEXT, WHITESPACE, WMTR)
            # Use query_model/query_revision if specified in config, otherwise use model/revision
            effective_config = config
            if config.type == VectorType.SPARSE_MODEL and config.query_model is not None:
                dumped = config.model_dump()
                dumped["model"] = config.query_model
                dumped["revision"] = config.query_revision
                effective_config = VectorConfigInternal(**dumped)
            
            # Generated sparse vector - generate once, reuse for all fields
            if config.type in VectorType.custom_tokenization():
                result = await router(
                    effective_config,
                    [query.query],
                    avgdls=[5.0],
                    trigram_weight=query.wmtr_trigram_weight,
                )
            else:
                result = await router(
                    effective_config,
                    [query.query],
                    trigram_weight=query.wmtr_trigram_weight,
                )
            indices, values = result[0]
            
            # Create VectorData for all fields using the same generated vector
            for field in fields:
                vector_data = VectorData(
                    vector_name=config.name,
                    field=field,
                    vector_type=config.type,
                    sparse_indices=indices,
                    sparse_values=values
                )
                vector_data_list.append(vector_data)
        
        return vector_data_list
    
    @staticmethod
    def _get_field_text(document: Document, field: str) -> str:
        """
        Extract text content from a document field.
        
        Args:
            document: The document to extract text from
            field: The field name to extract
            
        Returns:
            str: The text content of the field, or empty string if not found
        """
        if field == "name":
            return document.name or ""
        elif field == "description":
            return document.description or ""
        elif field == "content":
            return document.content or ""
        else:
            # For unknown fields, return empty string
            return ""
