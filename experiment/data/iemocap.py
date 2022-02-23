import itertools
import os
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_dataloader import BaseDataLoader


class IEMOCAPLoader(BaseDataLoader):
    
    def __init__(self,
        fold: int = 0,
        use_soft_target: bool = True,
        use_all_emotions: bool = False,
        turn_type: str = "impro",
        oversampling: List[str] = ["happy", "sad"],
        cross_corpus: Optional[Dict[str, str]] = None,
        **kwargs) -> None:
        """
        """
        super().__init__(**kwargs)
        self.fold: int = fold
        self.use_soft_target: bool = use_soft_target
        self.use_all_emotions: bool = use_all_emotions
        self.oversampling: List[str] = oversampling
        self.turn_type: str = turn_type.lower().strip()
        self.cross_corpus: Optional[Dict[str, str]] = cross_corpus
        
        self.score_cols: List[str] = [
            "neutral_score", 
            "angry_score", 
            "happy_score",
            "sad_score"
        ]
        if self.use_all_emotions:
            self.score_cols += [
                "excited_score",
                "frustrated_score",
                "surprise_score",
                "fear_score",
                "other_score",
                "disgust_score"
            ]
        
        sessions: List[str] = [
            f"Ses0{i[0]}{i[1]}"
            for i in itertools.product(
                range(1, 6),
                ["F", "M"]
            )
        ]
        self.fold_mapping: Dict[int, List[int]] = {
            0: [list(set(sessions) - {"Ses01M", "Ses01F"}), ["Ses01M"], ["Ses01F"]],
            1: [list(set(sessions) - {"Ses01F", "Ses01M"}), ["Ses01F"], ["Ses01M"]],
            
            2: [list(set(sessions) - {"Ses02M", "Ses02F"}), ["Ses02M"], ["Ses02F"]],
            3: [list(set(sessions) - {"Ses02F", "Ses02M"}), ["Ses02F"], ["Ses02M"]],
            
            4: [list(set(sessions) - {"Ses03M", "Ses03F"}), ["Ses03M"], ["Ses03F"]],
            5: [list(set(sessions) - {"Ses03F", "Ses03M"}), ["Ses03F"], ["Ses03M"]],
            
            6: [list(set(sessions) - {"Ses04M", "Ses04F"}), ["Ses04M"], ["Ses04F"]],
            7: [list(set(sessions) - {"Ses04F", "Ses04M"}), ["Ses04F"], ["Ses04M"]],            
            
            8: [list(set(sessions) - {"Ses05M", "Ses05F"}), ["Ses05M"], ["Ses05F"]],
            9: [list(set(sessions) - {"Ses05F", "Ses05M"}), ["Ses05F"], ["Ses05M"]],
        }
            
        self.train_studios, self.val_studios, self.test_studios = self.fold_mapping[self.fold]
            
    def set_fold(self, fold: int) -> None:
        self.fold = fold
        self.train_studios, self.val_studios, self.test_studios = self.fold_mapping[self.fold]
        
    def setup_train(self) -> None:
        label: pd.DataFrame = self.label
            
        if self.turn_type == "impro":
            label = label[label["name"].str.contains("impro")]
        elif self.turn_type == "script":
            label = label[label["name"].str.contains("script")]
        
        if not self.use_all_emotions:
            label = label[label["dominant_emotion"].isin(["neutral", "angry", "happy", "sad"])]
            
        train: pd.DataFrame = label[
            label["name"].map(lambda x: x.split("_")[0] in self.train_studios)
        ]
            
        scores: np.ndarray = train[self.score_cols].values.astype(float)
        paths: np.ndarray = train["path"].values
            
        if self.use_soft_target:
            self.train: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f,
                    "emotion": e.astype(float) / e.astype(float).sum()
                }
                for f, e in zip(paths, scores)
                if e.astype(float).sum() != 0.
            ]
        else:
            self.train: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
                } 
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
        
    def setup_val(self) -> None:
        label: pd.DataFrame = self.label
            
        if self.turn_type == "impro":
            label = label[label["name"].str.contains("impro")]
        elif self.turn_type == "script":
            label = label[label["name"].str.contains("script")]
        
        if not self.use_all_emotions:
            label = label[label["dominant_emotion"].isin(["neutral", "angry", "happy", "sad"])]
            
        train: pd.DataFrame = label[
            label["name"].map(lambda x: x.split("_")[0] in self.val_studios)
        ]
            
        scores: np.ndarray = train[self.score_cols].values.astype(float)
        paths: np.ndarray = train["path"].values
            
        if self.use_soft_target:
            self.val: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f,
                    "emotion": e.astype(float) / e.astype(float).sum()
                }
                for f, e in zip(paths, scores)
                if e.astype(float).sum() != 0.
            ]
        else:
            self.val: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
                } 
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
                
    def setup_test(self) -> None:
        label: pd.DataFrame = self.label
            
        if self.turn_type == "impro":
            label = label[label["name"].str.contains("impro")]
        elif self.turn_type == "script":
            label = label[label["name"].str.contains("script")]
        
        if not self.use_all_emotions:
            label = label[label["dominant_emotion"].isin(["neutral", "angry", "happy", "sad"])]
            
        train: pd.DataFrame = label[
            label["name"].map(lambda x: x.split("_")[0] in self.test_studios)
        ]
            
        scores: np.ndarray = train[self.score_cols].values.astype(float)
        paths: np.ndarray = train["path"].values
            
        if self.use_soft_target:
            self.test: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f,
                    "emotion": e.astype(float) / e.astype(float).sum()
                }
                for f, e in zip(paths, scores)
                if e.astype(float).sum() != 0.
            ]
        else:
            self.test: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
                } 
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
                
    def prepare_test(self, frame_size) -> DataLoader:
        # prepare test
        print("Preparing Testing Samples");
        test_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(self.test):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            test_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        test_dataloader: DataLoader = DataLoader(test_samples, batch_size=1, num_workers=1);
        return test_dataloader;
    
    def prepare_emodb(self, frame_size: float) -> DataLoader:
        # unpack label
        label_path: str = None;
        for k in self.cross_corpus.keys():
            if k.lower().strip() == "emodb":
                label_path = self.cross_corpus[k];
        
        # check if label_path exists
        if label_path is None:
            raise ValueError(f"Cannot call `prepare_emodb` if not provided in self.cross_corpus");
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"`label_path` does not exists at {label_path}");

        emodb: pd.DataFrame = pd.read_csv(label_path);
        
        # filter emodb
        emodb = emodb[emodb["emotion"].isin([e.replace("_score", "") for e in self.score_cols])];

        scores: np.ndarray = emodb[self.score_cols].values.astype(float);
        paths: np.ndarray = emodb["path"].values;
        emodb: List[Dict[str, Union[str, Tensor]]] = [
            {
                "feature": f, 
                "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
            } 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];

        print("Preparing EmoDB Samples");
        emodb_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(emodb):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            emodb_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        emodb_dataloader: DataLoader = DataLoader(emodb_samples, batch_size=1, num_workers=1);
        return emodb_dataloader;
    
    def prepare_thaiser(self, frame_size: float, turn_type: str = "all", mic_type: str = "con", agreement: float = 0.71) -> DataLoader:
        assert mic_type.lower().strip() in ["con", "clip"]
        
        label_path: str = None
        for k in self.cross_corpus.keys():
            if k.lower().strip() == "thaiser":
                label_path = self.cross_corpus[k]
                
        if label_path is None:
            raise ValueError(f"Cannot call `prepare_iemocap` if not provided in self.cross_corpus");
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"`label_path` does not exists at {label_path}");
        
        thaiser: pd.DataFrame = pd.read_csv(label_path)
            
        # filter thaiser
        # FIXME:
#         thaiser = thaiser[thaiser["dominant_emotion"].isin([e.replace("_score", "") for e in self.score_cols])];
        thaiser = thaiser[thaiser["mic"] == mic_type]
        thaiser = thaiser[thaiser["agreement"] >= agreement]
        thaiser = thaiser[thaiser["turn_type"] == turn_type] if turn_type.lower().strip() != "all" else thaiser
        
        scores = thaiser[self.score_cols].values.astype(float)
        scores = scores / scores.sum(axis=1, keepdims=True)
        paths = thaiser["path"].values
        
        thaiser: List[Dict[str, Union[str, Tensor]]] = [
            {
                "feature": f, 
                "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
            } 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];

        print("Preparing THAISER Samples");
        thaiser_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(thaiser):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            thaiser_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        thaiser_dataloader: DataLoader = DataLoader(thaiser_samples, batch_size=1, num_workers=1);
        return thaiser_dataloader;
        
        
    def prepare_emovo(self, frame_size: float) -> DataLoader:
        # unpack label
        label_path: str = None;
        for k in self.cross_corpus.keys():
            if k.lower().strip() == "emovo":
                label_path = self.cross_corpus[k];
        
        # check if label_path exists
        if label_path is None:
            raise ValueError(f"Cannot call `prepare_emovo` if not provided in self.cross_corpus");
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"`label_path` does not exists at {label_path}");

        emovo: pd.DataFrame = pd.read_csv(label_path);
        
        # filter emovo
        emovo = emovo[emovo["emotion"].isin([e.replace("_score", "") for e in self.score_cols])];

        scores: np.ndarray = emovo[self.score_cols].values.astype(float);
        paths: np.ndarray = emovo["path"].values;
        emovo: List[Dict[str, Union[str, Tensor]]] = [
            {
                "feature": f, 
                "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
            } 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];

        print("Preparing EMOVO Samples");
        emovo_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(emovo):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            emovo_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        emovo_dataloader: DataLoader = DataLoader(emovo_samples, batch_size=1, num_workers=1);
        return emovo_dataloader;