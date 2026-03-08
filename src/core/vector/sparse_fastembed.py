from __future__ import annotations

import os
import logging
from typing import List, Optional, Tuple

import re

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal
from ..common import AMGIXCache, MODEL_CACHE_SIZE, MODEL_CACHE_TTL, DEFAULT_SPARSE_TOP_K, SPARSE_MODEL_BATCH_SIZE, FASTEMBED_CACHE_DIR

AMGIX_CUDA = os.getenv("AMGIX_CUDA", "false").lower() == "true"
AMGIX_BUILD_FE = os.getenv("AMGIX_BUILD_FE", "").lower()


class SparseFastEmbedVector(VectorBase):
    """
    Sparse vector generation using FastEmbed sparse models.

    Produces sparse vectors using FastEmbed's sparse embedding models.
    - Uses FastEmbed's built-in sparse embedding generation
    - Applies top-K filtering based on config.top_k
    - Returns indices and values for each document
    """

    # Cache loaded models by model name
    _MODEL_CACHE: AMGIXCache[str, object] = AMGIXCache("ttl_lru", "sparse_fe_models", maxsize=MODEL_CACHE_SIZE, ttl=MODEL_CACHE_TTL)  # Cache up to 20 models
    
    # Cache fastembed imports at class level
    _SparseTextEmbedding = None
    _fe_available = None

    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        super().__init__(trusted_organizations, logger)
        
        # Lazy import fastembed, cache at class level
        if SparseFastEmbedVector._fe_available is None:
            try:
                from fastembed import SparseTextEmbedding
                SparseFastEmbedVector._SparseTextEmbedding = SparseTextEmbedding
                SparseFastEmbedVector._fe_available = True
            except ImportError:
                SparseFastEmbedVector._fe_available = False

        self.cuda = True if AMGIX_CUDA and AMGIX_BUILD_FE == "gpu" else False

    def get_sparse_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[Tuple[List[int], List[float]]]:
        if not config.model or not config.model.strip():
            raise ValueError("SparseFastEmbedVector requires 'model' to be specified in VectorConfig")
        
        if not SparseFastEmbedVector._fe_available:
            raise ValueError("FastEmbed is not available")

        processed_docs = self.preprocess_text(docs, keep_case=config.keep_case)

        results: List[Tuple[List[int], List[float]]] = []

        model = self._load_model(config.model)
        embeddings = list(model.embed(processed_docs, batch_size=SPARSE_MODEL_BATCH_SIZE, parallel=4))

        for embedding in embeddings:
            # FastEmbed returns SparseEmbedding objects with values and indices
            indices = embedding.indices.tolist()
            values = embedding.values.tolist()
            
            # Apply top_k filtering using base class method
            indices, values = self.top_k(list(zip(indices, values)), config.top_k)
            
            results.append((indices, values))

        return results

    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:  # type: ignore[override]
        raise NotImplementedError("SparseFastEmbedVector does not produce dense vectors")

    def _get_sparse_vector(self, config: VectorConfigInternal, docs: str) -> Tuple[List[int], List[float]]:
        raise NotImplementedError("SparseFastEmbedVector does not use _get_sparse_vector")

    def _load_model(self, model_name: str):
        key = model_name
        
        def load_model():
            try:
                model = SparseFastEmbedVector._SparseTextEmbedding(model_name=model_name, cuda=self.cuda, cache_dir=FASTEMBED_CACHE_DIR)
                return model
            except Exception as e:
                # Wrap the error with the specific model name for better debugging
                raise ValueError(f"Failed to load model '{model_name}': {str(e)}") from e
        
        return self._MODEL_CACHE.get_or_add(key, load_model)
    
    def unload_model(self, model_name: str, revision: str = None) -> None:
        """
        Unload a model from the cache.
        
        Args:
            model_name: Name of the model to unload
            revision: Model revision (ignored for FastEmbed models)
        """
        key = model_name
        del self._MODEL_CACHE[key]


