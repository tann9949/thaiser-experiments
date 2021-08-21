from numpy.core.numeric import cross
from experiment.data.feature.feature_packer import FeaturePacker
from experiment.data.feature.featurizer import Featurizer
import json
import os
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader

from .data.thaiser import ThaiSERLoader
from .evaluate import Evaluator
from .model.base_model import BaseModel


class CirriculumWrapper:
    """
    Cirriculum Traininng Wrapper proposed in the paper. This experiment
    train model progressively from high agreemennt to low while smoothing
    soft label moving average for each iteration
    """
    def __init__(
        self,
        fold: int,
        featurizer: Featurizer,
        packer: FeaturePacker,
        frame_size: int,
        test_mics: List[str],
        dataloader_param: Dict[str, Any],
        agreement_range: Tuple[float, float],
        cross_corpus: Dict[str, DataLoader],
        n_iteration: int,
        batch_size: int,
        checkpoint_path: str):
        """
        """
        self.fold: int = fold;
        self.featurizer: Featurizer = featurizer;
        self.packer: FeaturePacker = packer;
        self.frame_size: int = frame_size;

        self.dataloader_param: Dict[str, Any] = dataloader_param;
        self.test_mics: List[str] = test_mics;

        self.min_agreement: float = agreement_range[0];
        self.max_agreement: float = agreement_range[1];

        self.cross_corpus: Dict[str, DataLoader] = cross_corpus;

        self.n_iteration: int = n_iteration;
        self.batch_size: int = batch_size;
        self.checkpoint_path: str = checkpoint_path;

    def run(self):
        """
        Run experiment from config file

        Return
        ------
        exp_result: Dict[str, Dict[str, Any]]
            Return a dictonary contain experiments results and its statistic
        """
        results: Dict[str, List[Tensor]] = {};
        for i in range(self.n_iteration):
            print(f"Running Iteration [{i+1}/{self.n_iteration}]");

            # instantiate model
            model: BaseModel = self.ModelClass(self.hparams);
            callbacks: List[Callback] = [
                ModelCheckpoint(dirpath=f"{self.checkpoint_path}/{i}/", monitor="val_loss")
            ];
            logger: TensorBoardLogger = TensorBoardLogger(save_dir=f"{self.checkpoint_path}/{i}", version=1, name="lightning_logs")
            trainer: pl.Trainer = pl.Trainer(**self.trainer_params, callbacks=callbacks, logger=logger);

            # iterate over agreement
            for agreement in np.arange(self.min_agreement, self.max_agreement, 0.1):
                print(f">"*5, f"Training on agreement = {agreement}");

                dataloader: ThaiSERLoader = ThaiSERLoader(
                    featurizer=self.featurizer,
                    packer=self.packer,
                    agreement=agreement,
                    **self.dataloader_param
                );

                dataloader.setup();
                train_dataloader, val_dataloader = dataloader.prepare(frame_size=self.frame_size, batch_size=batch_size);

                test_loaders: Dict[str, DataLoader] = {};
                for mic in self.test_mics:
                    test_loaders = {
                        **test_loaders,
                        f"TEST-{mic}": dataloader.prepare_test(frame_size=self.frame_size, mic_type=mic),
                    }
                test_loaders = {
                    **test_loaders,
                    **self.cross_corpus
                }

                # train model using trainer
                trainer.fit(
                    model, 
                    train_dataloader=train_dataloader, 
                    val_dataloaders=val_dataloader
                );

                weight_path: str = f"{self.checkpoint_path}/{i}/ag-{agreement}/final.ckpt";
                trainer.save_checkpoint(weight_path);

                # evaluate model
                evaluator: Evaluator = Evaluator(model);
                if agreement not in results.keys():
                    results[agreement] = {}
                for test_name, test_dl in self.test_dataloader.items():  # iterate over test loaders
                    if test_name not in results[agreement].keys():
                        results[agreement][test_name] = {"experiment_results": {}, "statistics": {}}
                    result, predictions = evaluator(test_dl, return_prediction=True);
                    
                    if not os.path.exists(f"{self.checkpoint_path}/{i}/ag-{agreement}/pred_{test_name}"):
                        os.makedirs(f"{self.checkpoint_path}/{i}/ag-{agreement}/pred_{test_name}");
                    with open(f"{self.checkpoint_path}/{i}/ag-{agreement}/pred_{test_name}/prediction.json", "w") as f:
                        json.dump(predictions, f);

                    for metric in result.keys():  # iterate over
                        if metric not in results[test_name]["experiment_results"].keys():
                            results[test_name]["experiment_results"][metric] = [];
                        results[test_name]["experiment_results"][metric].append(result[metric]);
                        print(f"[{test_name}]\t{metric}: {result[metric]:.4f}");

                # update model