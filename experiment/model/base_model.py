from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, CyclicLR


class BaseModel(pl.LightningModule):

    def __init__(
            self, 
            hparams: Dict[str, Any] = {},
            schedule_learning_rate = False,
            **kwargs
        ) -> None:
        super().__init__();
        self.schedule_learning_rate = schedule_learning_rate
        self.hyperparams: Dict[str, Any] = hparams;
        self.learning_rate: float = self.hyperparams.get("learning_rate", 1e-4);

    def set_learning_rate(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate;

    def forward(self, x: Dict[str, Union[str, Tensor]], *args, **kwargs) -> Tensor:
        raise NotImplementedError();

    def compute_loss(self, y_hat: Tensor, y: Tensor, *args, **kwargs) -> float:
        assert y_hat.shape[0] == y.shape[0];
        return F.cross_entropy(y_hat, y.argmax(dim=-1));

    def compute_metric(self, y_hat: Tensor, y: Tensor, *args, **kwargs) -> float:
        assert y_hat.shape[0] == y.shape[0];
        if torch.cuda.is_available():
            y_hat = y_hat.cuda();
            y = y.cuda();
        return torch.sum(y_hat == y) / y_hat.shape[0];

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        x, y = batch["feature"], batch["emotion"];
        y_hat: Tensor = self(x);
        loss: float = self.compute_loss(y_hat, y);
        acc: float = self.compute_metric(y_hat.argmax(dim=-1), y.argmax(dim=-1));
        metrics: Dict[str, float] = {"train_loss": loss, "train_acc": acc};
        self.log_dict(metrics, prog_bar=True, logger=True);
        return loss;

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        emotion: Tensor = batch["emotion"];
        feature: List[Tensor] = batch["feature"];

        with torch.no_grad():
            # old methods
            """
            # compute logits of each chunk
            probs: Tensor = torch.stack([F.softmax(self(x), dim=-1) for x in feature]);
            # compute loss by average loss from each chunk
            loss: float = torch.stack([self.compute_loss(prob, emotion) for prob in probs]).mean();
            # compute emotion index and compute acc
            tmp_count: Tensor = torch.zeros([probs.shape[-1],], dtype=torch.int);  # count of each emotion
            tmp_score: Tensor = torch.zeros([probs.shape[-1],]);  # score of each emotion
            for prob in probs:
                pred_emotion: Tensor = prob.argmax();
                pred_score: Tensor = prob.max();
                    
                count: Tensor = tmp_count[pred_emotion];
                score: Tensor = tmp_score[pred_emotion];
                tmp_count[pred_emotion] = count + 1;
                tmp_score[pred_emotion] = score + pred_score;

            unique, count = tmp_count.unique(return_counts=True);

            if sum([1 if c == count.max() else 0 for c in count]) > 1:
                scores: Tensor = torch.zeros([len(count),]);
                for i, c in enumerate(count):
                    if c == count.max():
                        scores[i] += sum([s for c, s in zip(tmp_count, tmp_score) if c == unique[i]])
                prediction: Tensor = unique[scores.argmax()];
            else:
                prediction: Tensor = unique[count.argmax()];
            prediction = prediction.unsqueeze(dim=0);
            acc: Tensor = self.compute_metric(prediction, emotion.argmax(dim=-1));
            """
            # average distance
            logits: Tensor = torch.stack([self(x) for x in feature]).mean(dim=0)
            probs: Tensor = F.softmax(logits, dim=-1)
            loss: float = self.compute_loss(probs, emotion)
            acc: Tensor = self.compute_metric(probs.argmax(-1), emotion.argmax(-1))
            
            metrics: Dict[str, float] = {"val_acc": acc, "val_loss": loss};
            self.log_dict(metrics);
        return metrics;

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        metrics: Dict[str, float] = self.validation_step(batch, batch_idx)
        metrics: Dict[str, float] = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics;

    def configure_optimizers(self) -> Optimizer:
        if self.schedule_learning_rate:
            opt = Adam(self.parameters(), lr=self.learning_rate)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(
                        opt,
                        factor=0.1,
                        patience=3,
                        verbose=True
                    ),
                    "monitor": "val_acc"
                },
            }
        else:
            opt: Adam = Adam(self.parameters(), lr=self.learning_rate)
            return opt;
