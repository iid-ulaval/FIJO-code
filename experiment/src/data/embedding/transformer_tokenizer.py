from typing import List, Tuple
from abc import ABC, abstractmethod


class TransformerTokenizer(ABC):
    @abstractmethod
    def tokenize_and_align(
            self, batch: List[Tuple[str, List]]) -> Tuple[List, List, List]:
        pass
