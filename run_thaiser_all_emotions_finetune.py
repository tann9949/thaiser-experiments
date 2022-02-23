import json
import logging
import warnings
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, CyclicLR

from experiment.data.feature.feature_packer import FeaturePacker
from experiment.data.feature.featurizer import Featurizer
from experiment.data.thaiser import ThaiSERLoader
from experiment.experiment_wrapper import ExperimentWrapper
from experiment.model.base_model import BaseModel
from experiment.model.cnnlstm import CNNLSTM
from experiment.utils import notify_line, read_config
from experiment.evaluate import Evaluator

warnings.filterwarnings("ignore")
pl.utilities.distributed.log.setLevel(logging.ERROR)

ROOT_DIR = "playground/finetune-emotion"


class FineTuneCNNLSTM(BaseModel):
    
    def __init__(
        self, 
        pretrained: CNNLSTM, 
        n_classes: int,
        schedule_lr: str = 'reduce_lr_plateau',
        cyclic_lr_step_size: Optional[int] = None,
        **kwargs):
        super().__init__(**kwargs);
        self.cnn_layers, self.lstm, old_linear = list(pretrained.children())
        lstm_unit: int = int(old_linear.weight.shape[-1] / 2)
        self.logits: nn.Linear = nn.Linear(lstm_unit * 2, n_classes)
            
        self.schedule_lr = schedule_lr
        if self.schedule_lr == "triangle":
            assert cyclic_lr_step_size is not None and isinstance(cyclic_lr_step_size, int)
        self.cyclic_lr_step_size = cyclic_lr_step_size
            
    def freeze_base(self):
        for name, p in self.cnn_layers.named_parameters():
            p.requires_grad = False
        for name, p in self.lstm.named_parameters():
            p.requires_grad = False
        
    def unfreeze_base(self):
        for name, p in self.cnn_layers.named_parameters():
            p.requires_grad = True
        for name, p in self.lstm.named_parameters():
            p.requires_grad = True
        
    def forward(self, x: Tensor) -> Tensor:
        for cnn in self.cnn_layers:
            x = cnn(x)  # (batch_size, freq, time_seq)
        x = x.transpose(1, 2)  # (batch_size, time_seq, freq)
        _, (x, _) = self.lstm(x)  # (num_layers * num_directions, batch, hidden_size)
        x = x.transpose(0, 1)  # (batch_size, num_layers * num_directions, hidden_size)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # flatten -> (batch_size, feat_dim)
        x = self.logits(x)
        return x
    
    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.learning_rate)
        if self.schedule_lr == "reduce_lr_plateau":
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(
                        opt,
                        factor=0.1,
                        patience=3
                    ),
                    "monitor": "val_acc"
                },
            }
        elif self.schedule_lr == "triangle":
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": CyclicLR(
                        opt,
                        1e-6,
                        1e-4,
                        cycle_momentum=False,
                        step_size_up=self.cyclic_lr_step_size,
                        mode="triangular2",
                        
                    )
                }
            }
        else:
            return opt


def get_dataloader(include_fru):
    featurizer: Featurizer = Featurizer(
        feature_type="fbank",
        feature_param={
            "num_mel_bins": 64,
            "frame_length": 25,
            "frame_shift": 10,
            "low_freq": 0,
            "high_freq": 8000,
            "sample_frequency": 16000
        },
        vtlp_range=[0.9, 1.1]
    )
        
    packer: FeaturePacker = FeaturePacker(
        max_len=3,
        pad_mode="dup",
        stats_path=None
    )
        
    dataloader: ThaiSERLoader = ThaiSERLoader(
        featurizer=featurizer,
        packer=packer,
        agreement=0.71,
        label_path="dataset/wav/THAISER/labels.csv",
        use_soft_target=False,
        include_fru=include_fru,
        include_zoom=False,
        train_mic="con"
    )
    dataloader.setup()
    
    return dataloader


def get_trainer(root_dir, log_dir, i = None):
    if i is None:
        i = ""
    logger: TensorBoardLogger = TensorBoardLogger(
        save_dir=f"{root_dir}/{i}", 
        version=1, 
        name=log_dir
    )
        
    callbacks: List[Callback] = [
        ModelCheckpoint(
            dirpath=f"{root_dir}/{i}", 
            monitor="val_acc"
        ),
#         EarlyStopping(
#             monitor="val_acc",
#             patience=7
#         )
    ];

    trainer: pl.Trainer = pl.Trainer(
        max_epochs=30,
        gpus=[0], 
        callbacks=callbacks, 
        logger=logger
    );
    return trainer


def get_model(n_classes):
    model: CNNLSTM = CNNLSTM(
        hparams = {
            "in_channel": 64,
            "sequence_length": 300,
            "n_classes": n_classes,
            "n_channels": [64, 64, 128, 128],
            "kernel_size": [5, 3, 3, 3],
            "pool_size": [2, 2, 2, 2],
            "lstm_unit": 128
        },
        schedule_learning_rate=True
    )
    model.set_learning_rate(1e-4);
    model.hparams.lr = 1e-4
    return model


def evaluate_model(model, test_loader):
    evaluator: Evaluator = Evaluator(model);
    result, predictions = evaluator(test_loader, return_prediction=True);

    y_true = [x["label"] for x in predictions]
    y_pred = [x["prediction"] for x in predictions]
    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true"
    )
    wa = accuracy_score(y_true, y_pred)
    ua = np.diag(cm).mean()
    return cm, wa, ua


def train_base_emotion(root_dir, n_iter):
    dataloader = get_dataloader(include_fru=False)
    train_dataloader, val_dataloader = dataloader.prepare(
        frame_size=0.01, 
        batch_size=64
    );
    test_loader: DataLoader = dataloader.prepare_test(frame_size=0.01)
        
    exp_result = {
        "wa": [],
        "ua": []
    }
    for i in range(n_iter):
        print(f"Iteration {i}/10")
        model = get_model(n_classes=4)
        trainer = get_trainer(root_dir, "base", i)
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{root_dir}/{i}/base-final.ckpt")

        cm, wa, ua = evaluate_model(model, test_loader)
        exp_result["wa"].append(wa)
        exp_result["ua"].append(ua)
    
        print(f"Weighted Accuracy: {wa*100:.2f}%")
        print(f"Unweighted Accuracy: {ua*100:.2f}%")
        print(f"Confusion Matrix:")
        print(cm, "\n")
        
        with open(f"{root_dir}/base_model.txt", "a") as f:
            f.write(f"iteration{i} {wa*100:.2f} {ua*100:.2f}\n")
        
        with open(f"{root_dir}/base_confusion-matrix.txt", "a") as f:
            f.write(f"iteration{i}\n{cm}\n")  
    
    with open(f"{root_dir}/base_model.txt", "a") as f:
        wa_results = np.array([exp_result["wa"]])
        ua_results = np.array([exp_result["ua"]])
        
        f.write(f"average      {wa_results.mean()*100:.2f}±{wa_results.std()*100:.2f}% " + \
                f"{ua_results.mean()*100:.2f}±{ua_results.std()*100:.2f}%")
        

def load_base_model(model_path, n_classes):
    model = CNNLSTM.load_from_checkpoint(
        model_path, 
        hparams={
            "in_channel": 64,
            "sequence_length": 300,
            "n_classes": n_classes,
            "n_channels": [64, 64, 128, 128],
            "kernel_size": [5, 3, 3, 3],
            "pool_size": [2, 2, 2, 2],
            "lstm_unit": 128
        },
        schedule_learning_rate=True
    )
    return model

def get_head_model(model_path, base_model, cyclic_lr_step_size):
    return FineTuneCNNLSTM.load_from_checkpoint(
        model_path, 
        pretrained=base_model, 
        n_classes=5,
        schedule_lr="triangle",
        cyclic_lr_step_size=cyclic_lr_step_size,
    ).cuda()
        
        
def finetune_head(root_dir, n_iter):
    dataloader = get_dataloader(include_fru=True)
    train_dataloader, val_dataloader = dataloader.prepare(
        frame_size=0.01, 
        batch_size=64
    );
    test_loader: DataLoader = dataloader.prepare_test(frame_size=0.01)
    
    exp_result_head = {
        "wa": [],
        "ua": []
    }
    for i in range(n_iter):
        print(f"Iteration {i}/10")
        base_model = load_base_model(f"{root_dir}/{i}/base-final.ckpt", 4)
        model = FineTuneCNNLSTM(base_model, n_classes=5)
        model.freeze_base()
        model.set_learning_rate(1e-4);
        model.hparams.lr = 1e-4
        
        trainer = get_trainer(root_dir, "finetune_head", i)
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{root_dir}/{i}/head-final.ckpt")

        cm, wa, ua = evaluate_model(model, test_loader) 
        exp_result_head["wa"].append(wa)
        exp_result_head["ua"].append(ua)
    
        print(f"Weighted Accuracy: {wa*100:.2f}%")
        print(f"Unweighted Accuracy: {ua*100:.2f}%")
        print(f"Confusion Matrix:")
        print(cm, "\n")
        
        with open(f"{root_dir}/head_model.txt", "a") as f:
            f.write(f"iteration{i} {wa*100:.2f} {ua*100:.2f}\n")     
        
        with open(f"{root_dir}/head_confusion-matrix.txt", "a") as f:
            f.write(f"iteration{i}\n{cm}\n")  
        
    with open(f"{root_dir}/head_model.txt", "a") as f:
        wa_results = np.array([exp_result_head["wa"]])
        ua_results = np.array([exp_result_head["ua"]])
        
        f.write(f"average      {wa_results.mean()*100:.2f}±{wa_results.std()*100:.2f}% " + \
                f"{ua_results.mean()*100:.2f}±{ua_results.std()*100:.2f}%")
        
        
def finetune_all(root_dir, n_iter):
    dataloader = get_dataloader(include_fru=True)
    train_dataloader, val_dataloader = dataloader.prepare(
        frame_size=0.01, 
        batch_size=64
    );
    test_loader: DataLoader = dataloader.prepare_test(frame_size=0.01)
    
    exp_result_head = {
        "wa": [],
        "ua": []
    }
    exp_result_full = {
        "wa": [],
        "ua": []
    }
    for i in range(n_iter):
        print(f"Iteration {i}/10")
        base_model = load_base_model(f"{root_dir}/{i}/base-final.ckpt", 4)
        model = get_head_model(
            f"{root_dir}/{i}/head-final.ckpt", 
            base_model, 
            len(train_dataloader)
        )
        model.unfreeze_base()
        
        trainer = get_trainer(root_dir, "finetune_all", i)
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{root_dir}/{i}/all-final.ckpt")

        cm, wa, ua = evaluate_model(model, test_loader) 
        exp_result_full["wa"].append(wa)
        exp_result_full["ua"].append(ua)
    
        print(f"Weighted Accuracy: {wa*100:.2f}%")
        print(f"Unweighted Accuracy: {ua*100:.2f}%")
        print(f"Confusion Matrix:")
        print(cm, "\n")
        
        with open(f"{root_dir}/all_model.txt", "a") as f:
            f.write(f"iteration{i} {wa*100:.2f} {ua*100:.2f}\n")  
            
        with open(f"{root_dir}/all_confusion-matrix.txt", "a") as f:
            f.write(f"iteration{i}\n{cm}\n")  
        
    with open(f"{root_dir}/all_model.txt", "a") as f:
        wa_results = np.array([exp_result_full["wa"]])
        ua_results = np.array([exp_result_full["ua"]])
        
        f.write(f"average      {wa_results.mean()*100:.2f}±{wa_results.std()*100:.2f}% " + \
                f"{wa_results.mean()*100:.2f}±{wa_results.std()*100:.2f}%")

        
def main():       
    n_iter = 10
    
    train_base_emotion(ROOT_DIR, n_iter)
    finetune_head(ROOT_DIR, n_iter)
    finetune_all(ROOT_DIR, n_iter)
        
        
if __name__ == "__main__":
    main()
