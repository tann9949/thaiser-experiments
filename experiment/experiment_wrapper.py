import json
import os
from typing import Any, Dict, List, Union

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
        test_dataloader: Union[DataLoader, Dict[str, DataLoader]],
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

        ModelClass: BaseModel
            A class instantiator that inherited from BaseModel class. 
            Use to instantiate model for training
        hparams: Dict[str, Any]
            Hyperparameters of ModelClass when instantiating
        trainer_params: Dict[str, Any]
            Keyword arguments of pytorch lightning's Trainer 
            (see more at https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html)
        train_dataloader: DataLoader
            Data loader of training data
        val_dataloader: DataLoader
            Data loader of validation data
        test_dataloader: Union[DataLoader, List[DataLoader]]
            Data loader of testing data. Can be either a DataLoader of List of DataLoader.
            If List of Data Loader is parsed, all test dataloader will be use as an evaluation set
        n_iteration : int
            Number of training iteration to be done
        checkpoint_path: str
            Path to model checkpoints
        extra_dataloader
        """
        self.train_dataloader: DataLoader = train_dataloader;
        self.val_dataloader: DataLoader = val_dataloader;
        self.test_dataloader: Dict[str, DataLoader] = test_dataloader if isinstance(test_dataloader, dict) else {"TEST": test_dataloader};
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
            logger: TensorBoardLogger = TensorBoardLogger(save_dir=f"{self.checkpoint_path}/{i}", version=1, name="lightning_logs")
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
            for test_name, test_dl in self.test_dataloader.items():  # iterate over test loaders
                if test_name not in results.keys():
                    results[test_name] = {"experiment_results": {}, "statistics": {}}
                result, predictions = evaluator(test_dl, return_prediction=True);
                
                if not os.path.exists(f"{self.checkpoint_path}/{i}/pred_{test_name}"):
                    os.makedirs(f"{self.checkpoint_path}/{i}/pred_{test_name}");
                with open(f"{self.checkpoint_path}/{i}/pred_{test_name}/prediction.json", "w") as f:
                    json.dump(predictions, f);

                for metric in result.keys():  # iterate over
                    if metric not in results[test_name]["experiment_results"].keys():
                        results[test_name]["experiment_results"][metric] = [];
                    results[test_name]["experiment_results"][metric].append(result[metric]);
                    print(f"[{test_name}]\t{metric}: {result[metric]:.4f}");

        # compute results statistics
        print("\n" + "-"*20);
        print("Summary");
        print(f"-"*20);
        for name in results:  # iterate over test loaders
            print("*"*5, name, "*"*5);
            for metric in results[name]["experiment_results"]:  # iterate over metric
                result: np.ndarray = np.array(results[name]["experiment_results"][metric]);
                metric_mean: Tensor = result.mean();
                metric_std: Tensor = result.std();
                results[name]["statistics"][metric] = { "mean": metric_mean, "std": metric_std };
            
                print(f"{metric}: {metric_mean:.4f} ± {metric_std:.4f}")
        print(f"-"*20);
        
        return results;
