from __future__ import annotations

from typing import Dict, List, Tuple

from .vector_base import VectorBase
from ..models.document import Document
from ..models.vector import SearchQuery, VectorConfigInternal


class CustomSparseVector(VectorBase):
    """
    Handler for custom sparse vectors supplied with documents or queries.
    """

    def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("CustomSparseVector does not produce dense vectors")

    def _get_sparse_vector(self, config: VectorConfigInternal, docs: str) -> Tuple[List[int], List[float]]:
        raise NotImplementedError("CustomSparseVector does not generate from text")

    @staticmethod
    def extract_for_documents(
        config: VectorConfigInternal,
        documents: List[Document]
    ) -> Dict[int, Dict[str, Tuple[List[int], List[float]]]]:
        """
        Return per-document, per-field sparse vectors from Document.custom_vectors.

        Returns: {doc_idx: {field: (indices, values)}}
        """
        per_doc: Dict[int, Dict[str, Tuple[List[int], List[float]]]] = {}
        for idx, doc in enumerate(documents):
            per_field: Dict[str, Tuple[List[int], List[float]]] = {}
            if not doc.custom_vectors:
                raise ValueError(f"Custom sparse vector '{config.name}' requires custom vectors but document has none")
            for field in config.index_fields:
                cv = next((cv for cv in doc.custom_vectors if cv.vector_name == config.name and cv.field == field), None)
                if not cv:
                    raise ValueError(f"Custom sparse vector '{config.name}' for field '{field}' not provided")
                if len(cv.vector) > config.top_k:
                    raise ValueError(
                        f"Custom sparse vector '{config.name}' has {len(cv.vector)} entries, max allowed: {config.top_k}"
                    )
                indices = [it[0] for it in cv.vector]
                values = [it[1] for it in cv.vector]
                per_field[field] = (indices, values)
            per_doc[idx] = per_field
        return per_doc

    @staticmethod
    def extract_for_query(config: VectorConfigInternal, query: SearchQuery) -> Tuple[List[int], List[float]]:
        """Extract a single sparse vector from SearchQuery.custom_vectors for this config."""

        if not query.custom_vectors:
            raise ValueError(f"Custom sparse vector '{config.name}' requires custom vectors but query has none")

        cv = next((cv for cv in query.custom_vectors if cv.vector_name == config.name), None)
        if not cv:
            raise ValueError(f"Custom sparse vector '{config.name}' not provided in query")
        if len(cv.vector) > config.top_k:
            raise ValueError(
                f"Custom sparse vector '{config.name}' has {len(cv.vector)} entries, max allowed: {config.top_k}"
            )
        indices = [it[0] for it in cv.vector]
        values = [it[1] for it in cv.vector]
        return indices, values


