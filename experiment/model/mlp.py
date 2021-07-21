from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model.base_model import BaseModel


class MLP(BaseModel):
    
    def __init__(
        self, 
        **kwargs):
        super().__init__(**kwargs);
        input_dim: int = self.hparams.get("input_dim", 40);
        n_classes: int = self.hparams.get("n_classes", 4);
        self.linear: nn.Linear = nn.Linear(input_dim, n_classes);
        
    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = x.mean(dim=-1);
        x = self.linear(x);
        return x;
