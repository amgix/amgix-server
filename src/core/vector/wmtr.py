"""
Weighted Multilevel Token Representation (WMTR) vector generation.
"""
import logging
from typing import Counter, List, Tuple, Optional

import amgix_analyzers as aa

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal


class WMTRVector(VectorBase):
    """
    Weighted Multilevel Token Representation (WMTR) vector generation.
    
    This class generates sparse vectors based on a combination of whitespace,
    language-aware tokens, and character n-grams with different weighting schemes.
    """
    
    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        super().__init__(trusted_organizations, logger)
    
    def _get_sparse_vector(self, config: VectorConfigInternal, text: str, avgdl: float) -> Tuple[List[int], List[float]]:
        """
        Generate a sparse vector from text using WMTR approach.
        
        Args:
            text: The input text to generate a sparse vector for
            avgdl: Average document length for BM25 calculation
            
        Returns:
            Tuple[List[int], List[float]]: A tuple containing (indices, values) for the sparse vector
        """
        if not self._validate_text(text):
            return [], []

        lang_code = self.get_language_code(config, text)

        return aa.tokenize_wmtr(
            text=text,
            lang_code=lang_code,
            top_k_limit=config.top_k,
            word_weight_percentage=config.wmtr_word_weight,
            use_stopwords=True,
            avgdl=avgdl,
        )
    
    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("WMTRVector does not support dense vectors")
    
