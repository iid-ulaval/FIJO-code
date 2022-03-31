from typing import List, Tuple

from transformers import CamembertTokenizerFast

from . import TransformerTokenizer


class CamembertTokenizer(TransformerTokenizer):
    def __init__(self,
                 tags_to_idx: dict,
                 ignore_subwords: bool = True,
                 ignore_index: int = -100) -> None:
        self.tokenizer = CamembertTokenizerFast.from_pretrained(
            "camembert-base")
        self.tags_to_idx = tags_to_idx
        self.ignore_subwords = ignore_subwords
        self.ignore_index = ignore_index

    def tokenize_and_align(
            self, batch: List[Tuple[List, List]]) -> Tuple[List, List, List]:
        sequences, targets = zip(*batch)

        tokenized_sequence = self.tokenizer(list(sequences),
                                            return_attention_mask=True,
                                            padding="longest",
                                            is_split_into_words=True)
        targets_labels = []

        for i, target in enumerate(targets):
            word_indices = tokenized_sequence.word_ids(batch_index=i)

            target_labels = []

            previous_word_idx = None
            for word_idx in word_indices:
                if word_idx is None:
                    target_labels.append(self.ignore_index)
                elif word_idx != previous_word_idx:
                    target_labels.append(self.tags_to_idx[target[word_idx]])
                else:
                    target_labels.append(
                        self.ignore_index if self.ignore_subwords else self.
                        tags_to_idx[target[word_idx]])
                previous_word_idx = word_idx

            targets_labels.append(target_labels)

        return tokenized_sequence["input_ids"], tokenized_sequence[
            "attention_mask"], targets_labels
