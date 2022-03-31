import torch
from torch.nn import CrossEntropyLoss


def loss(prediction: torch.Tensor,
         ground_truth: torch.Tensor,
         device: torch.device,
         ignore_idx: int = -100) -> torch.Tensor:

    if isinstance(prediction, tuple):
        prediction = prediction[0]

    criterion = CrossEntropyLoss(ignore_index=ignore_idx)

    return criterion(
        prediction.transpose(-1, 1).to(device),
        ground_truth.type(torch.LongTensor).to(device))
