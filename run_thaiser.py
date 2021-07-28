from argparse import ArgumentParser, Namespace
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import DataLoader

from experiment.data.feature.feature_packer import FeaturePacker
from experiment.data.feature.featurizer import Featurizer
from experiment.data.thaiser import ThaiSERLoader
from experiment.experiment_wrapper import ExperimentWrapper
from experiment.model.cnnlstm import CNNLSTM
from experiment.utils import read_config

warnings.filterwarnings("ignore")
pl.utilities.distributed.log.setLevel(logging.ERROR)


def run_parser() ->  Namespace:
    """
    Run argument parser

    Return
    ------
    args: Namespace
        Arguments of the program
    """
    parser = ArgumentParser();
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

    #### Featurizer, Packer ####
    featurizer: Featurizer = Featurizer(**config["featurizer"]);
    packer: FeaturePacker = FeaturePacker(**config["packer"]);

    # init result statistics
    fold_stats: Dict[str, Any] = {};

    # iterate over fold ( 8 folds for THAI SER)
    for fold in range(8):
        
        print("*"*20);
        print(f"Running Fold {fold}");
        print("*"*20);

        dataloader: ThaiSERLoader = ThaiSERLoader(**config["dataloader"]);

        # prepare train, val data loaders
        dataloader.setup();
        train_dataloader, val_dataloader = dataloader.prepare(frame_size=frame_size, batch_size=batch_size);

        # prepare test loaders
        test_loaders: Dict[str, DataLoader] = {};
        for mic in test_mics:
            test_loaders = { 
                **test_loaders,
                f"TEST-{mic}": dataloader.prepare_test(frame_size=frame_size, mic_type=mic),
            }
        if test_zoom:
            test_loaders = { 
                **test_loaders,
                "ZOOM": dataloader.prepare_zoom(frame_size=frame_size)
            }
        if dataloader.cross_corpus is not None:
            for dataset_name, _ in dataloader.cross_corpus.items():
                if dataset_name.lower().strip() == "iemocap":
                    test_loaders = {
                        **test_loaders,
                        "IEMOCAP_IMPRO": dataloader.prepare_iemocap(frame_size=frame_size, turn_type="impro"),
                        "IEMOCAP_SCRIPT": dataloader.prepare_iemocap(frame_size=frame_size, turn_type="script"),
                        "IEMOCAP": dataloader.prepare_iemocap(frame_size=frame_size),
                    }
                elif dataset_name.lower().strip() == "emodb":
                    test_loaders = {
                        **test_loaders,
                        "EMODB": dataloader.prepare_emodb(frame_size=frame_size),
                    }
                elif dataset_name.lower().strip() == "emovo":
                    test_loaders = {
                        **test_loaders,
                        "EMOVO": dataloader.prepare_emovo(frame_size=frame_size),
                    }

        wrapper: ExperimentWrapper = ExperimentWrapper(
            ModelClass=CNNLSTM,
            hparams=config["model_hparams"],
            trainer_params=config["trainer_params"],
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_loaders,
            n_iteration=n_iteration,
            checkpoint_path=f"{exp_path}/fold{fold}"
        );

        exp_results: Dict[str, Any] = wrapper.run();
        fold_stats[fold] = exp_results;

    # print results
    metrics: List[str] = list(list(fold_stats[0].values())[0]["statistics"].keys());
    names: List[str] = list(fold_stats[0].keys());
    template: str = "fold\t\t"+"\t\t".join(metrics)+"\n";

    # compute average of all folds to get final results
    avgs: Dict[str, Any] = {name: {metric: [] for metric in metrics} for name in names};  # initialize template
    for f, result in fold_stats.items(): # iterate over fold
        for i, (name, name_data) in enumerate(result.items()):  # iterate over test loader
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
    aggregated_result: Dict[str, float] = {
        test_name: {
            k: sum(v)/len(v) 
            for k, v in test_result.items()
        } for test_name, test_result in avgs.items()
    };
    fold_stats["summary"] = aggregated_result;

    # pretty print results
    for test_name, test_result in aggregated_result.items():
        template += f"\n**** {test_name} ****\n"
        line: str = "Avg";
        for i, (metric, values) in enumerate(test_result.items()):
            indent: str = "\t\t" if i == 0 else "\t\t\t";
            line += f"{indent}{values:.4f}";
        template += line

    print("");
    print("*"*20);
    print("All Folds Summary");
    print("*"*20);
    print(template)
    
    # save experiment results
    with open(f"{exp_path}/exp_results.json", "w") as f:
        json.dump(fold_stats, f, indent=4);


if __name__ == "__main__":
    args = run_parser();
    main(args);
