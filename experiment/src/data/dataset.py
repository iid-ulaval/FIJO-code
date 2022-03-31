from typing import List, Tuple, Union

from torch.utils.data import Dataset as torchDataset


class Dataset(torchDataset):
    def __init__(
        self,
        data: List[Tuple[str, List]],
        transform=None,
    ):
        self.data = data

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        idx: Union[int,
                   slice]) -> Union[List[Tuple[str, List]], Tuple[str, List]]:
        result = None
        if isinstance(idx, slice):
            result = self.data[idx.start:idx.stop]
        else:
            result = self.data[idx]

        return result
