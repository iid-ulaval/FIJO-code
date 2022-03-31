import torch
import torch.nn as nn
from transformers import CamembertConfig, CamembertModel


class CamembertClassifier(nn.Module):
    def __init__(self,
                 output_size: int,
                 freeze_camembert: bool = True,
                 camembert_model_name: str = "camembert-base",
                 **camembert_config_kwargs) -> None:
        super().__init__()

        camembert_config = CamembertConfig(**camembert_config_kwargs)

        self.camembert = CamembertModel.from_pretrained(
            camembert_model_name, config=camembert_config)

        if freeze_camembert:
            for param in self.camembert.parameters():
                param.requires_grad = False

        self.projection_layer = nn.Linear(
            camembert_config_kwargs["hidden_size"], output_size)

    def forward(self, batch_inputs: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        batch_outputs = self.camembert(batch_inputs,
                                       attention_mask=attention_mask)[0]

        batch_outputs = self.projection_layer(batch_outputs)

        return batch_outputs
