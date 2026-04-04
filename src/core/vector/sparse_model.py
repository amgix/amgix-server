from __future__ import annotations

import os
import logging
from typing import List, Optional, Tuple

import re

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal
from ..common import AMGIXCache, MODEL_CACHE_SIZE, MODEL_CACHE_TTL, SPARSE_MODEL_BATCH_SIZE, HF_CACHE_DIR

# Check safetensors enforcement setting once at module level
REQUIRE_SAFETENSORS = os.getenv('AMGIX_SAFETENSORS', 'false').lower() == 'true'

class SparseModelVector(VectorBase):
    """
    Sparse vector generation using a Hugging Face masked language model (e.g., BERT).

    Produces sparse vectors over tokenizer vocabulary ids using model logits.
    Aggregation strategy:
    - Compute per-token logits across sequence
    - Apply attention mask to ignore padding
    - Aggregate by taking max over positions
    - Apply ReLU and log1p to compress dynamic range
    - Exclude special token ids
    - Keep top-K indices (K from config.top_k)
    - L2-normalize values for consistency with other sparse vectors
    """

    # Cache loaded models/tokenizers by (model, revision)
    _MODEL_CACHE: AMGIXCache[Tuple[str, Optional[str]], Tuple[object, object]] = AMGIXCache("ttl_lru", "sparse_models", maxsize=MODEL_CACHE_SIZE, ttl=MODEL_CACHE_TTL)  # Cache up to 20 models
    
    # Cache torch and transformers imports at class level
    _torch = None
    _AutoTokenizer = None
    _AutoModelForMaskedLM = None
    _transformers_available = None

    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        super().__init__(trusted_organizations, logger)
        
        # Lazy import torch and transformers, cache at class level
        if SparseModelVector._transformers_available is None:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                SparseModelVector._torch = torch
                SparseModelVector._AutoTokenizer = AutoTokenizer
                SparseModelVector._AutoModelForMaskedLM = AutoModelForMaskedLM
                SparseModelVector._transformers_available = True
            except ImportError:
                SparseModelVector._transformers_available = False
        
        # Select device
        if SparseModelVector._transformers_available:
            self._device = SparseModelVector._torch.device("cuda" if SparseModelVector._torch.cuda.is_available() else "cpu")
        else:
            self._device = None

    def get_sparse_vector(
        self,
        config: VectorConfigInternal,
        docs: List[str],
        trigram_weight: float,
    ) -> List[Tuple[List[int], List[float]]]:
        if not config.model or not config.model.strip():
            raise ValueError("SparseModelVector requires 'model' to be specified in VectorConfig")
        
        if not SparseModelVector._transformers_available:
            raise ValueError("transformers is not available in this environment")

        processed_docs = self.preprocess_text(docs, keep_case=config.keep_case)

        results: List[Tuple[List[int], List[float]]] = []

        tokenizer, model = self._load_model(config.model, config.revision)
        model.to(self._device)
        model.eval()

        # Get special token IDs to exclude from all documents
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            special_ids.add(pad_id)

        # Micro-batch to limit memory usage
        for start in range(0, len(processed_docs), SPARSE_MODEL_BATCH_SIZE):
            chunk = processed_docs[start:start + SPARSE_MODEL_BATCH_SIZE]

            # Tokenize chunk (no truncation change per request)
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with SparseModelVector._torch.no_grad():
                outputs = model(**encoded)

            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise ValueError("Model outputs do not contain 'logits'. Use a masked language model (e.g., AutoModelForMaskedLM).")

            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                raise ValueError("Tokenizer outputs do not contain 'attention_mask'.")

            # Process each document in the chunk
            chunk_results: List[Tuple[List[int], List[float]]] = []
            for i in range(logits.shape[0]):
                doc_logits = logits[i:i+1]
                doc_mask = attention_mask[i:i+1].unsqueeze(-1)

                masked_logits = doc_logits.masked_fill(doc_mask == 0, float("-inf"))

                per_token_scores = masked_logits.max(dim=1).values.squeeze(0)
                per_token_scores = SparseModelVector._torch.relu(per_token_scores)
                per_token_scores = SparseModelVector._torch.log1p(per_token_scores)

                if special_ids:
                    idx_tensor = SparseModelVector._torch.tensor(list(special_ids), device=per_token_scores.device)
                    per_token_scores.index_fill_(0, idx_tensor, 0.0)

                positive_mask = per_token_scores > 0
                if not SparseModelVector._torch.any(positive_mask):
                    chunk_results.append(([], []))
                    continue

                scores = per_token_scores[positive_mask]
                indices = SparseModelVector._torch.nonzero(positive_mask, as_tuple=False).squeeze(1)

                k = min(config.top_k, scores.numel())
                topk = SparseModelVector._torch.topk(scores, k=k, largest=True, sorted=True)
                top_scores = topk.values
                top_indices = indices[topk.indices]

                values = top_scores.detach().cpu().numpy().astype("float32").tolist()
                idxs = top_indices.detach().cpu().numpy().astype("int64").tolist()

                chunk_results.append((idxs, values))

            results.extend(chunk_results)

        return results

    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:  # type: ignore[override]
        raise NotImplementedError("SparseModelVector does not produce dense vectors")

    def _get_sparse_vector(
        self,
        config: VectorConfigInternal,
        docs: str,
        avgdl: float,
        trigram_weight: float,
    ) -> Tuple[List[int], List[float]]:
        raise NotImplementedError("SparseModelVector does use _get_sparse_vector")

    def _load_model(self, model_name: str, revision: Optional[str]):
        key = (model_name, revision)
        
        def load_model():
            try:
                # Check if trusted organizations are enabled and validate model
                if self._trusted_organizations is not None:
                    if not self.is_trusted_model(model_name):
                        raise ValueError(f"Model '{model_name}' is not from a trusted organization. Trusted organizations: {', '.join(sorted(self._trusted_organizations))}")
                
                try:
                    self.logger.debug(f"[SparseModel] About to load model '{model_name}' (revision={revision}) with local_files_only=True")
                    # Try loading from cache only (no HTTP calls)
                    tokenizer = SparseModelVector._AutoTokenizer.from_pretrained(
                        model_name, 
                        revision=revision, 
                        use_fast=True, 
                        cache_dir=HF_CACHE_DIR,
                        local_files_only=True
                    )
                    self.logger.debug(f"[SparseModel] Tokenizer loaded from cache for '{model_name}'")
                    model = SparseModelVector._AutoModelForMaskedLM.from_pretrained(
                        model_name, 
                        revision=revision, 
                        use_safetensors=REQUIRE_SAFETENSORS,
                        cache_dir=HF_CACHE_DIR,
                        local_files_only=True
                    )
                    self.logger.debug(f"[SparseModel] Successfully loaded model '{model_name}' from cache")
                    return (tokenizer, model)
                except Exception as e:
                    self.logger.debug(f"[SparseModel] Failed to load from cache: {type(e).__name__}: {str(e)}")
                    self.logger.debug(f"[SparseModel] Attempting to download model '{model_name}' (revision={revision})")
                    # Model not in cache, download it
                    tokenizer = SparseModelVector._AutoTokenizer.from_pretrained(
                        model_name, 
                        revision=revision, 
                        use_fast=True, 
                        cache_dir=HF_CACHE_DIR
                    )
                    self.logger.debug(f"[SparseModel] Tokenizer downloaded for '{model_name}'")
                    model = SparseModelVector._AutoModelForMaskedLM.from_pretrained(
                        model_name, 
                        revision=revision, 
                        use_safetensors=REQUIRE_SAFETENSORS,
                        cache_dir=HF_CACHE_DIR
                    )
                    self.logger.debug(f"[SparseModel] Successfully downloaded model '{model_name}'")
                    return (tokenizer, model)
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


