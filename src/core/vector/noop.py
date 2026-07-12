from typing import List, Tuple

from .vector_base import VectorBase
from ..models.vector import VectorConfigInternal


class NoopVector(VectorBase):
    """Always returns an empty sparse vector. Used for payload-only collections."""

    def _get_sparse_vector(
        self,
        config: VectorConfigInternal,
        text: str,
        avgdl: float,
        trigram_weight: float,
    ) -> Tuple[List[int], List[float]]:
        return [], []

    async def get_dense_vector(self, config: VectorConfigInternal, docs: List[str]) -> List[List[float]]:
        raise NotImplementedError("NoopVector only supports sparse vectors")
