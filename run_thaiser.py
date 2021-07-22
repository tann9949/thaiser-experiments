from typing import Dict, Optional, Tuple, Any
import warnings
import logging

import pytorch_lightning as pl

from experiment.experiment_wrapper import ExperimentWrapper
from experiment.data.feature.feature_packer import FeaturePacker
from experiment.data.feature.featurizer import Featurizer
from experiment.data.thaiser import ThaiSERLoader
from experiment.model.cnnlstm import CNNLSTM

warnings.filterwarnings("ignore")
pl.utilities.distributed.log.setLevel(logging.ERROR)


#### HYPER PARAMETERS ####
label_path: str = "dataset/wav/THAISER/labels.csv";

agreement: float = 0.71;
smoothing_param: float = 0.;
use_soft_target: bool = True;
include_fru: bool = False;
include_zoom: bool = False;
train_mic: str = "con";
test_mic: Optional[str] = None;
    
num_mel_bins: int = 40;
max_len: int = 3
frame_length: int = 25;
frame_shift: int = 10;
vtlp_range: Optional[Tuple[float, float]] = (0.9, 1.1);
n_class: int = 5 if include_fru else 4;
pad_mode: str = "dup";
stats_path: Optional[str] = "stats.pckl";

hparams: Dict[str, Any] = {
    "in_channel": num_mel_bins,
    "sequence_length": int(max_len * (1000 / frame_shift)),
    "n_channels": [64, 64, 128, 128],
    "kernel_size": [5, 3, 3, 3],
    "pool_size": [2, 2, 2, 2],
    "lstm_unit": 128,
    "n_classes": n_class
}
    
batch_size: int = 16;
max_epoch: int = 3;
n_iteration: int = 2;
checkpoint_path: str = "log/cnnlstm_exp";
    

def main():
    #### Featurizer, Packer ####
    featurizer: Featurizer = Featurizer(
        feature_type="fbank",
        feature_param={
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "num_mel_bins": num_mel_bins,
            "low_freq": 0,
            "high_freq": 8000,
            "sample_frequency": 16000
        },
        vtlp_range=vtlp_range
    );
        
    packer: FeaturePacker = FeaturePacker(
        pad_mode=pad_mode,
        max_len=max_len,
        stats_path=stats_path
    );


    fold_stats: Dict[str, Any] = {};

    # iterate over fold ( 8 folds for THAI SER)
    for fold in range(8):
        
        print("*"*20);
        print(f"Running Fold {fold}");
        print("*"*20);

        dataloader: ThaiSERLoader = ThaiSERLoader(
            fold=fold,
            use_soft_target=use_soft_target,
            train_mic=train_mic,
            test_mic=test_mic,
            smoothing_param=smoothing_param,
            label_path=label_path,
            agreement=agreement,
            featurizer=featurizer,
            packer=packer
        );

        dataloader.setup();
        train_dataloader, val_dataloader, test_dataloader = dataloader.prepare(frame_size=frame_shift/1000, batch_size=batch_size);

        wrapper: ExperimentWrapper = ExperimentWrapper(
            ModelClass=CNNLSTM,
            hparams=hparams,
            trainer_params={
                "max_epochs": max_epoch, 
                "progress_bar_refresh_rate": 0,
                "weights_summary": None
            },
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            n_iteration=n_iteration,
            checkpoint_path=checkpoint_path
        );

        exp_results: Dict[str, Any] = wrapper.run();
        fold_stats[fold] = exp_results;

    # print results
    template: str = "fold\t\t"+"\t\t".join(fold_stats[0]["statistics"].keys())+"\n";
        
    avgs: Dict[str, Any] = {metric: [] for metric in fold_stats[0]["statistics"].keys()}
    for f, result in fold_stats.items():
        line: str = f"{f}\t\t";
        for metric, value in result["statistics"].items():
            line += f"{value['mean']:.4f}±{value['std']:.4f}\t\t";
            avgs[metric].append(value["mean"]);
        template += f"{line}\n"
    template += "\n"

    avgs = {k: sum(v)/len(v) for k, v in avgs.items()};
    fold_stats["summary"] = avgs;

    line: str = "Avg";
    for i, (metric, values) in enumerate(avgs.items()):
        indent: str = "\t\t" if i == 0 else "\t\t\t"
        line += f"{indent}{values:.4f}";
    template += line

    print("");
    print("*"*20);
    print("All Folds Summary");
    print("*"*20);
    print(template);


if __name__ == "__main__":
    main();