from typing import List, Tuple

from fasttext import load_model

from . import EmbeddingModel


class FasttextVectorizer(EmbeddingModel):
    def __init__(self, embeddings_path: str, tags_to_idx: dict) -> None:
        super().__init__()

        self.embedding_model = load_model(embeddings_path)
        self.tags_to_idx = tags_to_idx

    def embed(self, batch: List[Tuple[List, List]]) -> List[Tuple[List, List]]:
        embedded_batch = []

        for pair in batch:
            embedded_sequence = []
            for word in pair[0]:
                embedded_sequence.append(self.embedding_model[word])

            target_label = [self.tags_to_idx[tag] for tag in pair[1]]

            embedded_batch.append((embedded_sequence, target_label))

        return embedded_batch
