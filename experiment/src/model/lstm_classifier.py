import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LstmClassifier(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_lstm_layers: int,
                 output_size: int,
                 bidirectional: bool = False,
                 padding_value: int = -32) -> None:
        super().__init__()

        self.padding_value = padding_value
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            bidirectional=bidirectional)

        self.projection_input_size = hidden_size if not bidirectional else 2 * hidden_size
        self.projection_layer = nn.Linear(self.projection_input_size,
                                          output_size)

    def forward(self, input_: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        packed_sequence = pack_padded_sequence(input_,
                                               lengths.cpu(),
                                               batch_first=True,
                                               enforce_sorted=False)

        lstm_output, _ = self.lstm(packed_sequence)

        lstm_output, _ = pad_packed_sequence(lstm_output,
                                             batch_first=True,
                                             padding_value=self.padding_value)

        output = self.projection_layer(lstm_output)

        return output
