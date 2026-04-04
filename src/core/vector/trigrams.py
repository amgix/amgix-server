"""
Trigrams vector implementation.
Generates sparse vectors using trigram tokenization.
"""
import logging
from typing import List, Tuple

import amgix_analyzers as aa

from .vector_base import VectorBase
from ..common import VectorType
from ..models.vector import VectorConfigInternal


class TrigramsVector(VectorBase):
    """
    Trigrams vector generator.
    
    Generates sparse vectors using trigram tokenization.
    This is a language-agnostic approach that creates vectors based on character-level trigrams.
    """
    
    def __init__(self, trusted_organizations: set = None, logger: logging.Logger = None):
        super().__init__(trusted_organizations, logger)
        
    def _get_sparse_vector(
        self,
        config: VectorConfigInternal,
        text: str,
        avgdl: float,
        trigram_weight: float,
    ) -> Tuple[List[int], List[float]]:
        """
        Generate a sparse vector using trigram tokenization.
        
        Args:
            text: The input text to generate a sparse vector for
            avgdl: Average document length for BM25 calculation
            
        Returns:
            Tuple[List[int], List[float]]: A tuple containing (indices, values) for the sparse vector
        """

        if not self._validate_text(text):
            return [], []
        
        return aa.tokenize_trigrams(
            text=text,
            top_k_limit=config.top_k,
            avgdl=avgdl,
        )
    
    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("TrigramsVector only supports sparse vectors") 