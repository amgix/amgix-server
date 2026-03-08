from __future__ import annotations

import os
import logging
import re
from typing import List, Optional, Tuple

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal
from ..common import AMGIXCache, MODEL_CACHE_SIZE, MODEL_CACHE_TTL, DENSE_MODEL_BATCH_SIZE, FASTEMBED_CACHE_DIR

AMGIX_CUDA = os.getenv("AMGIX_CUDA", "false").lower() == "true"
AMGIX_BUILD_FE = os.getenv("AMGIX_BUILD_FE", "").lower()

class DenseFastEmbedVector(VectorBase):
    """
    Dense vector generation using a FastEmbed model.

    - Requires `config.model` (FastEmbed model id) and `config.dimensions`.
    - No fallbacks: if model loading or inference fails, the error is propagated.
    - Uses the model's built-in encode() method for consistent dimensions and reliable output.
    - L2 normalization is configurable via config.normalization (defaults to True for dense vectors).
    """

    # Cache loaded models by model name
    # Use thread-safe LRU cache to avoid importing fastembed at module import time
    _MODEL_CACHE: AMGIXCache[str, object] = AMGIXCache("ttl_lru", "dense_fe_models", maxsize=MODEL_CACHE_SIZE, ttl=MODEL_CACHE_TTL)  # Cache up to 20 models
    
    # Cache fastembed imports at class level
    _TextEmbedding = None
    _fe_available = None

    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        super().__init__(trusted_organizations, logger)
        
        # Lazy import fastembed, cache at class level
        if DenseFastEmbedVector._fe_available is None:
            try:
                from fastembed import TextEmbedding
                DenseFastEmbedVector._TextEmbedding = TextEmbedding
                DenseFastEmbedVector._fe_available = True
            except ImportError:
                DenseFastEmbedVector._fe_available = False

        self.cuda = True if AMGIX_CUDA and AMGIX_BUILD_FE == "gpu" else False

    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        if not config.model or not config.model.strip():
            raise ValueError("DenseVector requires 'model' to be specified in VectorConfig")
        
        if not DenseFastEmbedVector._fe_available:
            raise ValueError("FastEmbed is not available")

        processed_docs = self.preprocess_text(docs, keep_case=config.keep_case)

        # Use the model's built-in encoding method for consistent dimensions
        # This ensures the same output dimensions as validation
        # Use built-in normalization if configured
        model = self._load_model(config.model)
        embeddings = list(model.embed(processed_docs, batch_size=DENSE_MODEL_BATCH_SIZE, parallel=4))
        return [emb.tolist() for emb in embeddings]

    def _get_sparse_vector(self, config: VectorConfigInternal, docs: List[str]):
        raise NotImplementedError("DenseFastEmbedVector does not produce sparse vectors")

    def _load_model(self, model_name: str):
        key = model_name
        
        def load_model():
            try:
                # FastEmbed handles model loading internally
                return DenseFastEmbedVector._TextEmbedding(model_name=model_name, cuda=self.cuda, cache_dir=FASTEMBED_CACHE_DIR)
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


