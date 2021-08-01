from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model.base_model import BaseModel


class Evaluator:

    def __init__(
        self,
        model: BaseModel):
        self.model: BaseModel = model;

        if torch.cuda.is_available():
            self.model.cuda();

    @staticmethod
    def compute_weighted_accuracy(y_true: List[int], y_pred: List[int]) -> float:
        assert len(y_true) == len(y_pred);
        return sum([1. if y == y_hat else 0. for y, y_hat in zip(y_true, y_pred)]) / len(y_true);

    @staticmethod
    def compute_unweighted_accuracy(y_true: List[int], y_pred: List[int]) -> float:
        assert len(y_true) == len(y_pred);
        cm = confusion_matrix(y_true, y_pred);
        return (np.diag(cm) / cm.sum(axis=1)).mean();

    def __call__(
        self, 
        test_loader: DataLoader, 
        return_prediction: bool = False, 
        verbose: bool = True) -> Dict[str, Tensor]:
        if verbose:
            print("Evaluating...")

        names: List[str] = [];
        y_true: List[int] = [];
        y_pred: List[int] = [];
        test_iter: DataLoader = tqdm(test_loader) if verbose else test_loader;
        for batch in test_iter:
            name: str = batch["name"];
            emotion: Tensor = batch["emotion"].argmax(dim=-1);
            feature: List[Tensor] = batch["feature"];

            with torch.no_grad():
                # compute logits of each chunk
                probs: Tensor = torch.stack([F.softmax(self.model(x), dim=-1) for x in feature]);
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

                if count[[u==tmp_count.max() for u in unique].index(True)] != 1:
                    # if utterance votes are not consensus
                    # e.g. 1 utterance = 4 chunks and their votes are [2, 2, 1, 1]
                    scores: Tensor = torch.zeros([len(count),]);
                    for i, c in enumerate(count):
                        if c == count.max():
                            scores[i] += sum([s for c, s in zip(tmp_count, tmp_score) if c == unique[i]])
                    prediction: Tensor = tmp_score.argmax();
                else:
                    # if majority vote is consensus
                    prediction: Tensor = tmp_count.argmax();
                prediction = prediction.unsqueeze(dim=0);
            y_true.append(emotion.numpy()[0]);
            y_pred.append(prediction.numpy()[0]);
            names += name;
        
        weighted_accuracy: float = self.compute_weighted_accuracy(y_true, y_pred);
        unweighted_accuracy: float = self.compute_unweighted_accuracy(y_true, y_pred);
        assert len(names) == len(y_pred) == len(y_true);
        metrics = {"weighted_accuracy": weighted_accuracy, "unweighted_accuracy": unweighted_accuracy};
        results: List[Dict[str, Any]] = [{"name": name, "prediction": int(y_hat), "label": int(y)} for name, y_hat, y in zip(names, y_pred, y_true)];
        if return_prediction:
            return metrics, results;
        return metrics;          
