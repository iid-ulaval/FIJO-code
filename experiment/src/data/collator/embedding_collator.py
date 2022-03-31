from typing import Tuple, List

import torch
from torch.nn.utils.rnn import pad_sequence

from ..embedding import EmbeddingModel


class EmbeddingCollator:
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 padding_value: int = -32,
                 ignore_index: int = -100):
        self.embedding_model = embedding_model
        self.padding_value = padding_value
        self.ignore_index = ignore_index

    def collate_batch(
        self, batch: Tuple[List, List]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        input_tensor, target_tensor, lengths = zip(*[(
            torch.Tensor(embeded_sequence), torch.Tensor(target),
            len(embeded_sequence)
        ) for embeded_sequence, target in self.embedding_model.embed(batch)])

        input_tensor = pad_sequence(input_tensor,
                                    batch_first=True,
                                    padding_value=self.padding_value)

        target_tensor = pad_sequence(target_tensor,
                                     batch_first=True,
                                     padding_value=self.ignore_index)

        lengths_tensor = torch.Tensor(lengths)

        return ((input_tensor, lengths_tensor), target_tensor)
