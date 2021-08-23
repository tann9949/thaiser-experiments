from numpy.core.numeric import cross
from .data.feature.feature_packer import FeaturePacker
from .data.feature.featurizer import Featurizer
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
from .data.feature.featurizer import Featurizer
from .data.feature.feature_packer import FeaturePacker
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
        ModelClass: BaseModel,
        fold: int,
        eta: float,
        config: Dict[str, Any],
        cross_corpus: Dict[str, DataLoader],
        checkpoint_path: str,
        agreement_range: Tuple[float, float] = (0.3, 1.0),
        agreement_step: float = 0.1,
        learning_rate_decay_factor: float = 0.5):
        """
        Cirriculum Training Wrapper Constructor. This wrapper run cirriculum learning as proposed
        in a paper for `n_iteration` times stated in prompted config dictionary

        ModelClass: BaseModel
            A class instantiator that inherited from BaseModel class. 
            Use to instantiate model for training
        fold: int
            Experiment fold to run on. Must be an integer within range [1, 8]
        eta: float
            Pseudo label weight. A weight for pseudo label to used when smoothing a target label
        config: Dict[str, Any]
            A config dictionary read from config.yaml
        cross_corpus: Dict[str, DataLoader]
            A dictionary mapping cross corpus name and its DataLoader. Used
            when evaluating the model
        checkpoint_path: str
            Path to stored each training log and model
        agreement_range: Tuple[float, float]
            Min and Max (exclusive) value of agreement value to generate a cirriculum
        agreement_step: float
            Step for generating range of cirriculum
        learning_rate_decay_factor: float
            A factor used to scale learning rate once finish training on one agreement value
        """
        self.fold: int = fold;
        assert 0. <= eta <= 1., "eta must be within [0, 1]"
        self.eta: float = eta;

        self.ModelClass = ModelClass
        self.learning_rate_decay_factor: float = learning_rate_decay_factor;

        self.min_agreement: float = agreement_range[0];  # inclusive
        self.max_agreement: float = agreement_range[1];  # exclusive
        self.agreement_step: float = agreement_step;

        self.cross_corpus: Dict[str, DataLoader] = cross_corpus;
        self.checkpoint_path: str = checkpoint_path;

        # unpack config
        self.featurizer: Featurizer = Featurizer(**config["featurizer"]);
        self.packer: FeaturePacker = FeaturePacker(**config["packer"]);
        self.hparams: Dict[str, Any] = config["model_hparams"];
        self.trainer_params: Dict[str, Any] = config["trainer_params"];
        self.dataloader_param: Dict[str, Any] = config["dataloader"];

        self.frame_size: int = config["frame_size"];
        self.test_mics: List[str] = config["test_mics"];
        self.n_iteration: int = config["n_iteration"];
        self.batch_size: int = config["batch_size"];

    def run(self) -> Dict[str, Any]:
        """
        Run experiment from config file

        Return
        ------
        exp_result: Dict[str, Dict[str, Any]]
            Return a dictonary contain experiments results and its statistic
        """
        results: Dict[str, List[Tensor]] = {};
        for i in range(self.n_iteration):
            print("="*10, f"Running Iteration [{i+1}/{self.n_iteration}]", "="*10);
            
            # instantiate model
            model: BaseModel = self.ModelClass(self.hparams);

            # iterate over agreement in reversal order as high agreement = easy samples
            for agreement in np.arange(self.min_agreement, self.max_agreement, self.agreement_step)[::-1]:

                #### Prepare train/val/test dataloader ####
                dataloader: ThaiSERLoader = ThaiSERLoader(
                    featurizer=self.featurizer,
                    packer=self.packer,
                    agreement=agreement,
                    **self.dataloader_param
                );

                # prepare data
                dataloader.setup();
                train_dataloader, val_dataloader = dataloader.prepare(
                    frame_size=self.frame_size, 
                    batch_size=self.batch_size,
                    model=model,
                    eta=self.eta if agreement != self.max_agreement else 0.  # set weight to zero if at the start of iteration
                );

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

                #### train model using trainer ####
                # declare trainer
                callbacks: List[Callback] = [
                    ModelCheckpoint(dirpath=f"{self.checkpoint_path}/{i}/ag-{agreement}", monitor="val_loss")
                ];
                logger: TensorBoardLogger = TensorBoardLogger(save_dir=f"{self.checkpoint_path}/{i}/ag-{agreement}", version=1, name="lightning_logs")
                if agreement == self.max_agreement:
                    new_lr: float = model.learning_rate;
                else:
                    new_lr: float = model.decay_learning_rate(self.learning_rate_decay_factor);  # update trainer learning rate
                trainer: pl.Trainer = pl.Trainer(**self.trainer_params, callbacks=callbacks, logger=logger);  # declare Trainer

                # fit model
                print(">"*5, f"Training on Agreement value = {agreement}, learning rate = {new_lr}")
                trainer.fit(
                    model, 
                    train_dataloader=train_dataloader, 
                    val_dataloaders=val_dataloader
                );

                # save weight after finished training
                weight_path: str = f"{self.checkpoint_path}/{i}/ag-{agreement}/final.ckpt";
                trainer.save_checkpoint(weight_path);

                #### Evaluate Model from agreement training ####
                evaluator :Evaluator = Evaluator(model);
                agreement: str = str(round(agreement, 2));  # format float -> str for key storing
                if agreement not in results.keys():
                    results[agreement] = {}
                for test_name, test_dl in test_loaders.items():
                    if test_name not in results[agreement].keys():
                        results[agreement][test_name] = {"experiment_results": {}, "statistics": {}}
                    result, predictions = evaluator(test_dl, return_prediction=True);

                    if not os.path.exists(f"{self.checkpoint_path}/{i}/ag-{agreement}/pred_{test_name}"):
                        os.makedirs(f"{self.checkpoint_path}/{i}/ag-{agreement}/pred_{test_name}")
                    with open(f"{self.checkpoint_path}/{i}/ag-{agreement}/pred_{test_name}/prediction.json", "w") as f:
                        json.dump(predictions, f);

                    for metric in result.keys():
                        if metric not in results[agreement][test_name]["experiment_results"].keys():
                            results[agreement][test_name]["experiment_results"][metric] = [];
                        results[agreement][test_name]["experiment_results"][metric].append(result[metric]);
                        print(f"[{test_name}]\t{metric}: {result[metric]:.4f}");

        # compute results statistics
        print("\n" + "-"*20);
        print("Summary");
        print(f"-"*20);

        for ag, ag_result in results.items():
            print(f">>>>> AGREEMENT: {ag}")
            for name, test_result in ag_result.items():
                print("*"*5, name, "*"*5);
                for metric, exp_result in test_result["experiment_results"].items():
                    exp_result: np.ndarray = np.array(exp_result);
                    metric_mean: float = exp_result.mean();
                    metric_std: float = exp_result.std();
                    results[ag][name]["statistics"][metric] = { "mean": metric_mean, "std": metric_std };
                    print(f"{metric}: {metric_mean:.4f} ± {metric_std:.4f}")
        print(f"-"*20);

        return results;