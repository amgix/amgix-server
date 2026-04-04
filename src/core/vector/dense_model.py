from __future__ import annotations

import os
import logging
import re
from typing import List, Optional, Tuple

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal
from ..common import AMGIXCache, MODEL_CACHE_SIZE, MODEL_CACHE_TTL, DENSE_MODEL_BATCH_SIZE, HF_CACHE_DIR

# Check safetensors enforcement setting once at module level
REQUIRE_SAFETENSORS = os.getenv('AMGIX_SAFETENSORS', 'false').lower() == 'true'

# Set the environment variable once at module level
if REQUIRE_SAFETENSORS:
    os.environ['TRANSFORMERS_SAFE_SERIALIZATION'] = '1'

class DenseModelVector(VectorBase):
    """
    Dense vector generation using a Hugging Face Transformers model.

    - Requires `config.model` (Hugging Face model id) and `config.dimensions`.
    - No fallbacks: if model loading or inference fails, the error is propagated.
    - Uses the model's built-in encode() method for consistent dimensions and reliable output.
    - L2 normalization is configurable via config.normalization (defaults to True for dense vectors).
    """

    # Cache loaded models by (model, revision)
    # Use thread-safe LRU cache to avoid importing transformers at module import time
    _MODEL_CACHE: AMGIXCache[Tuple[str, Optional[str]], object] = AMGIXCache("ttl_lru", "dense_models", maxsize=MODEL_CACHE_SIZE, ttl=MODEL_CACHE_TTL)  # Cache up to 20 models
    
    # Cache torch and SentenceTransformer imports at class level
    _torch = None
    _SentenceTransformer = None
    _st_available = None

    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        super().__init__(trusted_organizations, logger)
        
        # Lazy import torch and sentence_transformers, cache at class level
        if DenseModelVector._st_available is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                DenseModelVector._torch = torch
                DenseModelVector._SentenceTransformer = SentenceTransformer
                DenseModelVector._st_available = True
            except ImportError:
                DenseModelVector._st_available = False
        
        # Select device
        if DenseModelVector._st_available:
            self._device = DenseModelVector._torch.device("cuda" if DenseModelVector._torch.cuda.is_available() else "cpu")
        else:
            self._device = None


    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        if not config.model or not config.model.strip():
            raise ValueError("DenseVector requires 'model' to be specified in VectorConfig")
        if not DenseModelVector._st_available:
            raise ValueError("SentenceTransformer is not available in this environment")

        processed_docs = self.preprocess_text(docs, keep_case=config.keep_case)

        # Use the model's built-in encoding method for consistent dimensions
        # This ensures the same output dimensions as validation
        # Use built-in normalization if configured
        model = self._load_model(config.model, config.revision)
        embeddings = model.encode(
            processed_docs,
            normalize_embeddings=config.normalization,
            show_progress_bar=False,
            batch_size=DENSE_MODEL_BATCH_SIZE,
        )
        return [emb.tolist() for emb in embeddings]

    def _get_sparse_vector(
        self,
        config: VectorConfigInternal,
        docs: List[str],
        avgdl: float,
        trigram_weight: float,
    ):
        raise NotImplementedError("DenseVector does not produce sparse vectors")

    def _load_model(self, model_name: str, revision: Optional[str]):
        """
        Load sentence transformer model with SafeTensors for security.
        Requires models to be available in SafeTensors format.
        """
        key = (model_name, revision)
        
        def load_model():
            try:
                # Check if trusted organizations are enabled and validate model
                if self._trusted_organizations is not None:
                    if not self.is_trusted_model(model_name):
                        raise ValueError(f"Model '{model_name}' is not from a trusted organization. Trusted organizations: {', '.join(sorted(self._trusted_organizations))}")
                
                try:
                    self.logger.debug(f"[DenseModel] About to load model '{model_name}' (revision={revision}) with local_files_only=True")
                    # Try loading from cache only (no HTTP calls)
                    model = DenseModelVector._SentenceTransformer(
                        model_name, 
                        revision=revision,
                        device=self._device,
                        cache_folder=HF_CACHE_DIR,
                        local_files_only=True
                    )
                    self.logger.debug(f"[DenseModel] Successfully loaded model '{model_name}' from cache")
                    return model
                except Exception as e:
                    self.logger.debug(f"[DenseModel] Failed to load from cache: {type(e).__name__}: {str(e)}")
                    self.logger.debug(f"[DenseModel] Attempting to download model '{model_name}' (revision={revision})")
                    # Model not in cache, download it
                    model = DenseModelVector._SentenceTransformer(
                        model_name, 
                        revision=revision,
                        device=self._device,
                        cache_folder=HF_CACHE_DIR
                    )
                    self.logger.debug(f"[DenseModel] Successfully downloaded model '{model_name}'")
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
            revision: Model revision
        """
        key = (model_name, revision)
        del self._MODEL_CACHE[key]


