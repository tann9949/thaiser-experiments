from argparse import ArgumentParser, Namespace

import json
import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from experiment.cirriculum_wrapper import CirriculumWrapper
from experiment.data.feature.feature_packer import FeaturePacker
from experiment.data.feature.featurizer import Featurizer
from experiment.data.thaiser import ThaiSERLoader
from experiment.model.cnnlstm import CNNLSTM
from experiment.utils import read_config

warnings.filterwarnings("ignore")


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
    eta: List[str] = config["eta"];
    test_zoom: bool = config["test_zoom"];
    exp_path: str = config["exp_path"];
    learning_rates: List[float] = config["learning_rates"];

    # format config such that compatible
    del config["dataloader"]["agreement"]

    # init result statistics
    fold_stats: Dict[str, Any] = {};

    # preload cross-corpus
    dataloader: ThaiSERLoader = ThaiSERLoader(
        featurizer=Featurizer(**config["featurizer"]),
        packer=FeaturePacker(**config["packer"]),
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

        # iterate over agreement for cirriculum learning
        wrapper: CirriculumWrapper = CirriculumWrapper(
            ModelClass=CNNLSTM,
            fold=fold,
            eta=eta,
            config=config,
            cross_corpus=cross_corpus,
            checkpoint_path=checkpoint_path,
            learning_rates=learning_rates,
            agreement_range=(0.3, 1.0),
            agreement_step=0.1,
        );
        exp_results: Dict[str, Any] = wrapper.run();
        fold_stats[fold] = exp_results;
    
    # print results
    metrics: List[str] = list(list(list(fold_stats[0].values())[0].values())[0]["statistics"].keys());
    names: List[str] = list(list(fold_stats[0].values())[0].keys());
    template: str = "fold\t\t"+"\t\t".join(metrics)+"\n";

    # compute global variance of all stats
    global_stats: Dict[str, Any] = dict();
    for fold, results in fold_stats.items():
        ag_results = list(results.values())[-1]  # get last agreement result trained on all agreement values
        for test_name, test_results in ag_results.items():
            if test_name not in global_stats.keys():
                global_stats[test_name] = {};
            for metric_name, metric_results in test_results["experiment_results"].items():
                if metric_name not in global_stats[test_name].keys():
                    global_stats[test_name][metric_name] = [];
                global_stats[test_name][metric_name] += metric_results;

    # compute average of all folds to get final results
    avgs: Dict[str, Any] = {name: {metric: [] for metric in metrics} for name in names};  # initialize template
    for f, results in fold_stats.items(): # iterate over fold
        ag_results = list(results.values())[-1]
        for i, (name, name_data) in enumerate(ag_results.items()):  # iterate over test loader
            # name: test loader name
            # name_data: {}
            line = "";  # init line
            if i == 0:
                line += f"{f} - ({name})\t\t";
            else:
                line += f"  - ({name})\t\t";
            for metric, value in name_data["statistics"].items():  # iterate over metric of each test loader
                line += f"{value['mean']:.4f}±{value['std']:.4f}\t\t";
                avgs[name][metric].append(value["mean"]);
            template += f"{line}\n"
    template += "\n"

    # aggregate all fold results
    fold_stats["summary"] = global_stats;

    # pretty print results
    for test_name, test_result in global_stats.items():
        template += f"\n**** {test_name} ****\n"
        line: str = "Avg";
        for i, (metric, values) in enumerate(test_result.items()):
            indent: str = "\t\t" if i == 0 else "\t\t\t";
            mean = np.array(values).mean()
            std = np.array(values).std()
            line += f"{indent}{mean:.4f} ± {std:.4f}";
        template += line

    print("");
    print("*"*20);
    print("All Folds Summary");
    print("*"*20);
    print(template);
    
    # save experiment results
    with open(f"{exp_path}/exp_results.json", "w") as f:
        json.dump(fold_stats, f, indent=4);

        
if __name__ == "__main__":
    args: Namespace = run_parser();
    main(args);
