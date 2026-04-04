from typing import List, Protocol, Tuple, Union, Optional

from src.core.models.vector import VectorConfigInternal


class EmbedRouter(Protocol):
    """Callable for embedding documents."""

    async def __call__(
        self,
        vector_config: VectorConfigInternal,
        docs: List[str],
        trigram_weight: float,
        avgdls: Optional[List[float]] = None,
    ) -> Union[List[List[float]], List[Tuple[List[int], List[float]]]]:
        pass
