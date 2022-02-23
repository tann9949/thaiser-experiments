from collections import OrderedDict
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model.base_model import BaseModel


class CNNLSTM(BaseModel):
    
    def __init__(
        self, 
        hparams,
        schedule_learning_rate = False,
        **kwargs):
        super().__init__(hparams, schedule_learning_rate, **kwargs);
        in_channel: int = hparams.get("in_channel", 64)
        sequence_length: int = hparams.get("sequence_length", 300)
        n_channels: List[int] = hparams.get("n_channels", [64, 64, 128, 128])
        kernel_size: List[int] = hparams.get("kernel_size", [5, 3, 3, 3])
        pool_size: List[int] = hparams.get("pool_size", [2, 2, 2, 2])
        lstm_unit: int = hparams.get("lstm_unit", 128)
        n_classes: int = hparams.get("n_classes", 4)

        self.in_channel: int = in_channel
        self.sequence_length: int = sequence_length
        assert len(n_channels) == len(kernel_size) == len(pool_size), "Size of `n_channels`, `kernel_size`, and " \
                                                                      "`pool_size` must equal "

        # configure cnn parameters
        in_channels: List[int] = [in_channel] + n_channels[:-1]
        out_channels: List[int] = n_channels
        seq_lens: List[int] = []
        for p in pool_size:
            seq_lens.append(sequence_length)
            sequence_length = sequence_length // p
        assert len(in_channels) == len(out_channels) == len(seq_lens)

        self.cnn_layers: nn.Sequential = nn.Sequential(OrderedDict([
            (f"conv{i}", nn.Sequential(
                nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=k, padding=(k - 1) // 2),
                nn.LeakyReLU(),
                nn.LayerNorm(normalized_shape=[oc, seq]),
                nn.MaxPool1d(kernel_size=p)
            )) for i, (ic, oc, k, seq, p) in enumerate(zip(in_channels, out_channels, kernel_size, seq_lens, pool_size))
        ]))
        self.lstm: nn.LSTM = nn.LSTM(input_size=out_channels[-1], hidden_size=lstm_unit, bidirectional=True, batch_first=True)
        self.logits: nn.Linear = nn.Linear(lstm_unit * 2, n_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        for cnn in self.cnn_layers:
            x = cnn(x)  # (batch_size, freq, time_seq)
        x = x.transpose(1, 2)  # (batch_size, time_seq, freq)
        _, (x, _) = self.lstm(x)  # (num_layers * num_directions, batch, hidden_size)
        x = x.transpose(0, 1)  # (batch_size, num_layers * num_directions, hidden_size)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # flatten -> (batch_size, feat_dim)
        x = self.logits(x)
        return x
