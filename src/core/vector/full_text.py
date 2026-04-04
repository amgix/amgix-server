"""
Full-text vector generation using Whoosh library for tokenization.
"""
import logging
from typing import List, Tuple, Optional

import amgix_analyzers as aa

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal


class FullTextVector(VectorBase):
    """
    Full-text vector generation using Whoosh library for tokenization.
    
    This class generates sparse vectors based on full-text search tokens,
    using Whoosh's tokenization capabilities for language-aware text processing.
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
        Generate a sparse vector from text using Whoosh tokenization.
        
        Args:
            text: The input text to generate a sparse vector for
            avgdl: Average document length for BM25 calculation
            
        Returns:
            Tuple[List[int], List[float]]: A tuple containing (indices, values) for the sparse vector
        """
        if not self._validate_text(text):
            return [], []

        lang_code = self.get_language_code(config, text)

        return aa.tokenize_fulltext(
            text=text,
            lang_code=lang_code,
            top_k_limit=config.top_k,
            use_stopwords=True,
            avgdl=avgdl,
        )
    
    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("FullTextVector does not support dense vectors")
    
