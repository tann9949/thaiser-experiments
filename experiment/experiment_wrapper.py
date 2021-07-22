from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader

from .evaluate import Evaluator
from .model.base_model import BaseModel


class ExperimentWrapper:
    """
    Experiment wrapper, automatically run experiment and 
    return results of n_iteration of experiment and calculate
    average metrics and its standard deviation
    """
    def __init__(
        self, 
        ModelClass: BaseModel,
        hparams: Dict[str, Any],
        trainer_params: Dict[str, Any],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        n_iteration: int,
        checkpoint_path: str):
        """
        Experiment Wrapper Constructor

        Data loader must be in a form of iterator where
        each element must be {
            "name": <file-name>,
            "feature": <input-feature>,
            "emotion": <target-emotion-vector>
        }

        train_dataloader: DataLoader
            Data loader of training data
        val_dataloader: DataLoader
            Data loader of validation data
        test_dataloader: DataLoader
            Data loader of testing data
        """
        self.train_dataloader: DataLoader = train_dataloader;
        self.val_dataloader: DataLoader = val_dataloader;
        self.test_dataloader: DataLoader = test_dataloader;
        self.n_iteration: int = n_iteration;

        self.ModelClass: BaseModel = ModelClass;
        self.hparams: Dict[str, Any] = hparams;

        self.trainer_params: Dict[str, Any] = trainer_params;
        self.checkpoint_path: str = checkpoint_path;

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run experiment from config file

        Return
        ------
        exp_result: Dict[str, Dict[str, Any]]
            Return a dictonary contain experiments results and its statistic
        """
        # train and evaluate self.n_iteration times
        results: Dict[str, List[Tensor]] = {};
        for i in range(self.n_iteration):
            print(f"Running Iteration [{i+1}/{self.n_iteration}]");
            # instantiate model, trainer
            model: BaseModel = self.ModelClass(self.hparams);

            callbacks: List[Callback] = [
                ModelCheckpoint(dirpath=f"{self.checkpoint_path}/{i}/", monitor="val_loss")
            ];
            logger: TensorBoardLogger = TensorBoardLogger(save_dir=f"{self.checkpoint_path}", version=1, name="lightning_logs")
            trainer: pl.Trainer = pl.Trainer(**self.trainer_params, callbacks=callbacks, logger=logger);

            # train model using trainer
            trainer.fit(
                model, 
                train_dataloader=self.train_dataloader, 
                val_dataloaders=self.val_dataloader
            );

            weight_path: str = f"{self.checkpoint_path}/{i}/final.ckpt";
            trainer.save_checkpoint(weight_path);

            # evaluate and stored in results list
            evaluator: Evaluator = Evaluator(model);
            result: Dict[str, Tensor] = evaluator(self.test_dataloader);

            for metric in result.keys():
                if metric not in results.keys():
                    results[metric] = [];
                results[metric].append(result[metric].numpy());
                print(f"{metric}: {result[metric]:.4f}");

        # compute results statistics
        results_stats: Dict[str, Tensor] = {}
        print(f"-"*20);
        for metric in results:
            result: np.ndarray = np.array(results[metric]);
            metric_mean: Tensor = result.mean();
            metric_std: Tensor = result.std();
            results_stats[metric] = { "mean": metric_mean, "std": metric_std };
            
            print(f"{metric}: {metric_mean:.4f} +- {metric_std:.4f}")
        print(f"-"*20);
        
        return {
            "statistics": results_stats,
            "experiment_results": {k: np.array(v) for k, v in results.items()}
        };