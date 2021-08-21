from argparse import ArgumentParser, Namespace
from experiment.cirriculum_wrapper import CirriculumWrapper
import json
import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from experiment.data.feature.feature_packer import FeaturePacker
from experiment.data.feature.featurizer import Featurizer
from experiment.data.thaiser import ThaiSERLoader
from experiment.experiment_wrapper import ExperimentWrapper
from experiment.model.cnnlstm import CNNLSTM
from experiment.utils import notify_line, read_config

warnings.filterwarnings("ignore")
pl.utilities.distributed.log.setLevel(logging.ERROR)


def run_parser() -> Namespace:
    """
    Run argument parser

    Return
    ------
    args: Namespace
        Arguments of the program
    """
    parser: ArgumentParser = ArgumentParser();
    parser.add_argument("--config-path", required=True, type=str, help="Path to training config file");
    return parser.parse_args();


def main(args: Namespace) -> None:
    # unpack config
    config: Dict[str, Any] = read_config(args.config_path);
    frame_size: float = config["frame_size"];
    test_mics: List[str] = config["test_mics"];
    test_zoom: bool = config["test_zoom"];
    batch_size: int = config["batch_size"];
    n_iteration: int = config["n_iteration"];
    exp_path: str = config["exp_path"];

    # format config such that compatible
    del config["dataloader"]["agreement"]

    #### Featurizer, Packer ####
    featurizer: Featurizer = Featurizer(**config["featurizer"]);
    packer: FeaturePacker = FeaturePacker(**config["packer"]);

    # init result statistics
    fold_stats: Dict[str, Any] = {};

    # preload cross-corpus
    dataloader: ThaiSERLoader = ThaiSERLoader(
        featurizer=featurizer,
        packer=packer,
        **config["dataloader"]
    );

    cross_corpus: Dict[str, DataLoader] = {};
    if test_zoom:
        cross_corpus = {
            **cross_corpus,
            "ZOOM": dataloader.prepare_zoom(frame_size=frame_size)
        }

    if dataloader.cross_corpus is not None:
        for dataset_name, _ in dataloader.cross_corpus.items():
            if dataset_name.lower().strip() == "iemocap":
                cross_corpus = {
                    **cross_corpus,
                    "IEMOCAP_IMPRO": dataloader.prepare_iemocap(frame_size=frame_size, turn_type="impro"),
                    "IEMOCAP_SCRIPT": dataloader.prepare_iemocap(frame_size=frame_size, turn_type="script"),
                    "IEMOCAP": dataloader.prepare_iemocap(frame_size=frame_size),
                }
            elif dataset_name.lower().strip() == "emodb":
                cross_corpus = {
                    **cross_corpus,
                    "EMODB": dataloader.prepare_emodb(frame_size=frame_size),
                }
            elif dataset_name.lower().strip() == "emovo":
                cross_corpus = {
                    **cross_corpus,
                    "EMOVO": dataloader.prepare_emovo(frame_size=frame_size),
                }

    # iterate over fold ( 8 folds for THAI SER)
    for fold in range(8):

        print("*"*20);
        print(f"Running Fold {fold}");
        print("*"*20);

        checkpoint_path: str = f"{exp_path}/fold{fold}";
        if packer.stats_path is not None:
            packer.set_stat_path(f"{checkpoint_path}/{packer.stats_path}");

        # iterate over agreement for cirriculum learning
        wrapper: CirriculumWrapper = CirriculumWrapper(
            ModelClass=CNNLSTM,
            hparams=config["model_hparams"],
            agreement_range=(0.3, 1.0),
            cross_corpus=cross_corpus,
            n_iteration=n_iteration,
            checkpoint_path=checkpoint_path
        )

        