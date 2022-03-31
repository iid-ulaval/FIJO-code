from typing import List, Tuple

import torch

from ..embedding import TransformerTokenizer


class TokenCollator:
    def __init__(self, tokenizer: TransformerTokenizer) -> None:
        self.tokenizer = tokenizer

    def collate_batch(
        self, batch: List[Tuple[str, List]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        tokenized_sequences, attention_mask, targets = self.tokenizer.tokenize_and_align(
            batch)

        return (torch.tensor(tokenized_sequences),
                torch.tensor(attention_mask)), torch.tensor(targets)
