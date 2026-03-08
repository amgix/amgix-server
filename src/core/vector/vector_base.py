"""
Abstract base class for vector generation implementations.
"""
import hashlib
import numpy as np
import re
import math
import logging
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
from langid.langid import LanguageIdentifier, model
import mmh3

# Suppress pkg_resources deprecation warning coming from stopwordsiso
# import warnings
# warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# from stopwordsiso import stopwords

from ..common.constants import MAX_SPARSE_VECTOR_THREADS
from ..models.vector import VectorConfigInternal

# Large prime number for token hash range
TOKEN_HASH_RANGE = 2147483647  # 2^31 - 1 (Mersenne prime)


class VectorBase(ABC):
    """
    Abstract base class for vector generation.
    
    This class defines the interface for all vector generation implementations.
    Concrete implementations should inherit from this class and implement
    all abstract methods.
    """
    
    # Trusted organizations passed in constructor
    _trusted_organizations = None
    # Cache for stopwords as lists per language code
    _stopwords_cache = {}
    
    # Singleton language identifier - eagerly initialized for thread-safety
    _language_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    
    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        """
        Initialize the vector generator with configuration.
        
        Args:
            trusted_organizations: Set of trusted organization names that can use .bin weights
            logger: Logger instance for this vector generator
        """
        self._trusted_organizations = trusted_organizations
        self.logger = logger
    
    def preprocess_text(self, docs: List[str], keep_case: bool = False) -> List[str]:
        """Normalize text: lowercase and normalize multiple spaces to single space."""
        new_docs = []
        for doc in docs:
            if keep_case:
                new_docs.append(re.sub(r'\s+', ' ', doc.strip()))
            else:
                new_docs.append(re.sub(r'\s+', ' ', doc.casefold().strip()))
        return new_docs
    
    def top_k(self, token_weights: List[Tuple[int, float]], top_k: int) -> Tuple[List[int], List[float]]:
        """
        Apply top-k filtering to token weights and return indices and values.
        
        Args:
            token_weights: List of (token_id, weight) tuples
            
        Returns:
            Tuple of (indices, values) for the sparse vector
        """

        # Always sort by weight (descending) for consistent, high-quality results
        token_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top-k filtering if needed
        if len(token_weights) > top_k:
            token_weights = token_weights[:top_k]
        
        # Extract indices and values
        indices = [item[0] for item in token_weights]
        values = [item[1] for item in token_weights]
        
        return indices, values
    
    def get_sparse_vector(self, config: VectorConfigInternal, docs: List[str], avgdls: Optional[List[float]]) -> List[Tuple[List[int], List[float]]]:
        return [self._get_sparse_vector(config, doc, avgdl=avgdl) for doc, avgdl in zip(self.preprocess_text(docs), avgdls)]
        # processed_docs = self.preprocess_text(docs)
        # if not processed_docs:
        #     return []
        
        # # Single doc: avoid overhead
        # if len(processed_docs) == 1:
        #     return [await self._get_sparse_vector(processed_docs[0])]
        
        # # Run async worker per doc inside separate processes (bypass GIL for pure Python CPU-bound code)
        # def run_in_process(doc: str) -> Tuple[List[int], List[float]]:
        #     return asyncio.run(self._get_sparse_vector(doc))
        
        # # Use process pool with picklable worker function
        # results: List[Tuple[List[int], List[float]]] = [([], [])] * len(processed_docs)
        # with ProcessPoolExecutor(max_workers=MAX_SPARSE_PROCESSES) as executor:
        #     future_to_idx = {executor.submit(run_in_process, doc): idx for idx, doc in enumerate(processed_docs)}
        #     for future in as_completed(future_to_idx):
        #         idx = future_to_idx[future]
        #         results[idx] = future.result()
        
        # return results

    @abstractmethod
    def _get_sparse_vector(self, config: VectorConfigInternal, docs: str, avgdl: float) -> Tuple[List[int], List[float]]:
        """
        Generate a sparse vector from text.
        
        Args:
            text: The input text to generate a sparse vector for
            avgdl: Average document length for BM25 calculation
            
        Returns:
            Tuple[List[int], List[float]]: A tuple containing (indices, values) for the sparse vector
        """
        pass
    
    @abstractmethod
    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        """
        Generate a dense vector from text.
        
        Args:
            text: The input text to generate a dense vector for
            
        Returns:
            List[float]: The dense vector as a list of floats
        """
        pass
    
    def unload_model(self, model_name: str, revision: str = None) -> None:
        """
        Unload a model from cache.
        
        Default implementation raises NotImplementedError.
        Subclasses that support model caching should override this method.
        
        Args:
            model_name: Name of the model to unload
            revision: Model revision (optional)
        """
        raise NotImplementedError(f"unload_model not implemented for {self.__class__.__name__}") 

    def get_token(self, feature: str) -> int:
        """
        Generate a token ID from a feature string using MurmurHash3.
        
        Args:
            feature: The feature string to hash
            
        Returns:
            int: A token ID in the range [0, TOKEN_HASH_RANGE)
        """
        # Use MurmurHash3 32-bit signed version (same as scikit-learn)
        hash_value = mmh3.hash(feature, signed=True)
        # Map to positive range [0, TOKEN_HASH_RANGE)
        return hash_value % TOKEN_HASH_RANGE

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the given text.
        
        Uses a singleton language identifier to avoid repeated initialization.
        The identifier is created only once when first needed.
        
        Args:
            text: The text to detect language for
            
        Returns:
            Tuple[str, float]: A tuple containing (language_code, confidence)
                              Language code is a two-letter ISO code (e.g., 'en', 'es')
                              Confidence is a float between 0.0 and 1.0
        """
        lang_code, confidence = VectorBase._language_identifier.classify(text)
        return (lang_code, confidence)

    def is_trusted_model(self, model_name: str) -> bool:
        """
        Check if a model is from a trusted organization.
        
        Trusted organizations can use .bin weights (non-safetensors) safely.
        
        Args:
            model_name: The model name (e.g., 'microsoft/DialoGPT-medium')
            
        Returns:
            bool: True if the model is from a trusted organization
        """
        if not model_name or '/' not in model_name:
            return False
        
        org = model_name.split('/')[0].lower()
        return org in self._trusted_organizations

    def get_stopwords(self, lang_code: str) -> List[str]:
        """
        Get stopwords for the specified language code.
        
        Args:
            lang_code: Two-letter language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            List[str]: List of stopwords for the language, or empty list if not found
        """
        cached = VectorBase._stopwords_cache.get(lang_code)
        if cached is not None:
            return cached
        sw = stopwords(lang_code)
        # Convert to list once and store
        sw_list = list(sw) if sw else []
        VectorBase._stopwords_cache[lang_code] = sw_list
        return sw_list

    def get_count_weights(self, tokens: List[str], base_weight: float = 1.0) -> List[Tuple[int, float]]:
        """
        Convert tokens to (token_id, weight) pairs using logarithmic weighting.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of (token_id, weight) tuples
        """
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Convert to list of (token_id, weight) pairs
        token_weights = []
        for token, count in token_counts.items():
            token_id = self.get_token(token)
            weight = base_weight * math.log(1 + count)
            token_weights.append((token_id, weight))
        
        return token_weights

    def _validate_text(self, text: str) -> bool:
        """
        Validate that text is not empty or only whitespace.
        
        Args:
            text: The text to validate
            
        Returns:
            bool: True if text is valid (not empty/whitespace), False otherwise
        """
        return True if text else False

    def l2_norm(self, vector: List[float]) -> List[float]:
        """
        Perform L2 normalization on a vector using NumPy.
        
        Args:
            vector: List of float values to normalize
            
        Returns:
            List[float]: L2 normalized vector (or original vector if norm is zero)
        """
        if not vector:
            return []
        
        # Convert to numpy array for efficient computation
        np_vector = np.array(vector, dtype=np.float32)
        
        # Calculate L2 norm
        norm = np.linalg.norm(np_vector)
        
        # If norm is zero (all elements are zero), return original vector
        if norm == 0:
            return vector
        
        # Normalize the vector
        normalized = np_vector / norm
        
        # Convert back to list
        return normalized.tolist()

    def dedup_sparse(self, sparse_vector: Tuple[List[int], List[float]]) -> Tuple[List[int], List[float]]:
        """
        Deduplicate sparse vector by combining weights for duplicate indices.
        
        When multiple tokens hash to the same index, their weights are summed together.
        This is necessary because sparse vectors cannot have duplicate indices.
        
        Args:
            sparse_vector: Tuple of (indices, values) where indices may contain duplicates
            
        Returns:
            Tuple[List[int], List[float]]: Deduplicated (indices, values) with combined weights
        """
        indices, values = sparse_vector
        
        # Use Counter to sum weights for duplicate indices
        index_weights = Counter()
        for idx, weight in zip(indices, values):
            index_weights[idx] += weight
        
        return list(index_weights.keys()), list(index_weights.values())

    def get_language_code(self, config: VectorConfigInternal, text: str) -> str:
        """
        Determine the language code to use based on vector configuration.
        
        Args:
            text: The text to analyze (for language detection if needed)
            
        Returns:
            str: The language code to use
            
        Raises:
            ValueError: If no language code could be determined
        """
        # If language detection is enabled
        if config.language_detect:
            detected_lang, confidence = self.detect_language(text)
            # Use detected language only if confidence is high enough
            if confidence >= config.language_confidence:
                return detected_lang
        
        # Use default language code if specified
        if config.language_default_code:
            return config.language_default_code
        
        # No language code could be determined
        raise ValueError(
            f"No language code could be determined for vector config '{config.name}'. "
            f"Either enable language_detect=True or specify language_default_code."
        )
