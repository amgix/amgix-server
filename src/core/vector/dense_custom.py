from __future__ import annotations

from typing import Dict, List, Tuple

from .vector_base import VectorBase
from ..models.document import Document
from ..models.vector import SearchQuery, VectorConfigInternal


class CustomDenseVector(VectorBase):
    """
    Handler for custom dense vectors supplied with documents or queries.
    """

    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("CustomDenseVector does not generate from text")

    def _get_sparse_vector(
        self,
        config: VectorConfigInternal,
        docs: str,
        avgdl: float,
        trigram_weight: float,
    ) -> Tuple[List[int], List[float]]:
        raise NotImplementedError("CustomDenseVector does not produce sparse vectors")

    @staticmethod
    def extract_for_documents(
        config: VectorConfigInternal,
        documents: List[Document]
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Return per-document, per-field dense vectors from Document.custom_vectors.

        Returns: {doc_idx: {field: dense_vector}}
        """
        per_doc: Dict[int, Dict[str, List[float]]] = {}
        for idx, doc in enumerate(documents):
            per_field: Dict[str, List[float]] = {}
            if not doc.custom_vectors:
                raise ValueError(f"Custom dense vector '{config.name}' requires custom vectors but document has none")
            for field in config.index_fields:
                cv = next((cv for cv in doc.custom_vectors if cv.vector_name == config.name and cv.field == field), None)
                if not cv:
                    raise ValueError(f"Custom dense vector '{config.name}' for field '{field}' not provided")
                if config.dimensions is not None and len(cv.vector) != config.dimensions:
                    raise ValueError(
                        f"Custom dense vector '{config.name}' has {len(cv.vector)} dimensions, expected {config.dimensions}"
                    )
                per_field[field] = cv.vector
            per_doc[idx] = per_field
        return per_doc

    @staticmethod
    def extract_for_query(config: VectorConfigInternal, query: SearchQuery) -> List[float]:
        """Extract a single dense vector from SearchQuery.custom_vectors for this config."""

        if not query.custom_vectors:
            raise ValueError(f"Custom dense vector '{config.name}' requires custom vectors but query has none")

        cv = next((cv for cv in query.custom_vectors if cv.vector_name == config.name), None)
        if not cv:
            raise ValueError(f"Custom dense vector '{config.name}' not provided in query")
        if config.dimensions is not None and len(cv.vector) != config.dimensions:
            raise ValueError(
                f"Custom dense vector '{config.name}' has {len(cv.vector)} dimensions, expected {config.dimensions}"
            )
            
        return cv.vector


