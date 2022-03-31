from abc import ABC, abstractmethod
from typing import List, Tuple


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, batch: List[Tuple[str, List]]) -> List[Tuple[List, List]]:
        pass
