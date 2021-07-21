from typing import List, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model.base_model import BaseModel


class Evaluator:

    def __init__(
        self,
        model: BaseModel):
        self.model: BaseModel = model;

    def __call__(self, test_loader: DataLoader) -> Dict[str, Tensor]:
        correct: List[Dict[str, Tensor]] = [
            self.model.test_step(batch, batch_idx)
            for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader))
        ];
        metrics: List[str] = list(correct[0].keys());
        correct: Tensor = Tensor([list(c.values()) for c in correct]);
        return {k: v for k, v in zip(metrics, correct.mean(dim=0))};
