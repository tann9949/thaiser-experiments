import os
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_dataloader import BaseDataLoader


class ThaiSERLoader(BaseDataLoader):
    """
    THAI SER DataLoader Module use to extract
    training, validation , and testing data
    """
    def __init__(self, 
        agreement: float = 0.71,  # default value
        test_agreement: float = 0.71,  # default test set agreement
        fold: int = 0,
        use_soft_target: bool = True,
        smoothing_param: float = 0.,
        train_mic: str = "con",
        include_fru: bool = False,
        include_zoom: bool = False,
        cross_corpus: Optional[Dict[str, str]] = None,
        **kwargs):
        """
        Thai SER DataLoader constructor
        
        agreement: float
        float: int
        train_mic: str
        include_fru: bool
        label_path: str
        featurizer: Feaurizer
        packer:  FeaturePacker
        """
        super().__init__(**kwargs);
        self.fold = fold;  # fold to load dataset from (see self.fold_mapping)
        self.agreement: float = agreement;  # annotator consensus agreement to filter dataset off
        self.test_agreement: float = test_agreement;  # test set setup for chooinng test set, use 0.71 for default consensus value
        self.train_mic: str = train_mic;  # mic to used for training
        self.include_zoom: bool = include_zoom;  # state whether to include zoom in training set or not
        self.smoothing_param: float = smoothing_param;  # smoothing parameter K for soft labeling
        self.use_soft_target: bool = use_soft_target;  # states whether to use soft label as a target or not
        self.cross_corpus: Optional[Dict[str, str]] = cross_corpus;  # a dictionary containing corpus name and its corresponded labels.csv path

        # emotion idx follow this columns
        if include_fru:
            self.score_cols: List[str] = ["neutral_score", "angry_score", "happy_score", "sad_score", "frustrated_score"];
        else:
            self.score_cols: List[str] = ["neutral_score", "angry_score", "happy_score", "sad_score"];

        self.fold_mapping: Dict[int, List[int]] = {
            0: [list(range(1, 71)), list(range(71, 76)), list(range(76, 81))],
            1: [list(range(1, 61)) + list(range(71, 81)), list(range(61, 66)), list(range(66, 71))],
            2: [list(range(1, 51)) + list(range(61, 81)), list(range(51, 56)), list(range(56, 61))],
            3: [list(range(1, 41)) + list(range(51, 81)), list(range(41, 46)), list(range(46, 51))],
            4: [list(range(1, 31)) + list(range(41, 81)), list(range(31, 36)), list(range(36, 41))],
            5: [list(range(1, 21)) + list(range(31, 81)), list(range(21, 26)), list(range(26, 31))],
            6: [list(range(1, 11)) + list(range(21, 81)), list(range(11, 16)), list(range(16, 21))],
            7: [list(range(11, 81)), list(range(1, 6)), list(range(6, 11))],
        };
        self.train_studios, self.val_studios, self.test_studios = self.fold_mapping[self.fold];

    def set_fold(self, fold: int) -> None:
        """
        Set THAISER loader fold

        Argument
        --------
        fold: int
            Fold to set dataloader to
        """
        self.fold = fold;
        
    def setup_train(self) -> None:
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] >= self.agreement];
        if self.include_zoom:
            zoom: pd.DataFrame = label[label["mic"] == "mic"];
        label = label[label["mic"] == self.train_mic];
        
        # train: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.train_studios)];
        train: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.train_studios)].iloc[:200];  # FIXME:

        if self.include_zoom:
            train = pd.concat([train, zoom], axis=0).reset_index(drop=True);

        scores: np.ndarray = train[self.score_cols].values.astype(float);
        paths: np.ndarray = train["path"].values;
        
        if self.use_soft_target:
            self.train: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": (
                        self.smoothing_param + Tensor(e.astype(float))
                    ).divide(
                        self.smoothing_param * len(Tensor(e.astype(float))) + Tensor(e.astype(float)).sum()
                    )
                }
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
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
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] >= self.test_agreement];
        label = label[label["mic"] == self.train_mic];
        
        # val: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.val_studios)];
        val: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.val_studios)].iloc[:200];  # FIXME:
        scores: np.ndarray = val[self.score_cols].values.astype(float);
        paths: np.ndarray = val["path"].values;
        
        if self.use_soft_target:
            self.val: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": (
                        self.smoothing_param + Tensor(e.astype(float))
                    ).divide(
                        self.smoothing_param * len(Tensor(e.astype(float))) + Tensor(e.astype(float)).sum()
                    )
                }
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
        else:
            self.val: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
                } 
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
        
    def prepare_test(self, frame_size, mic_type: Optional[str]) -> DataLoader:
        # init mic
        if mic_type is None:
            mic_type = self.train_mic;

        # setup test labels
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] >= self.test_agreement];
        label = label[label["mic"] == mic_type];
        
        # test: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.test_studios)];
        test: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.test_studios)].iloc[:200];  # FIXME:
        scores: np.ndarray = test[self.score_cols].values.astype(float);
        paths: np.ndarray = test["path"].values;
        
        if self.use_soft_target:
            self.test: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": (
                        self.smoothing_param + Tensor(e.astype(float))
                    ).divide(
                        self.smoothing_param * len(Tensor(e.astype(float))) + Tensor(e.astype(float)).sum()
                    )
                }
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
        else:
            self.test: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
                } 
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
            
        # prepare test
        print("Preparing Testing Samples");
        test_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(self.test):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            test_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        test_dataloader: DataLoader = DataLoader(test_samples, batch_size=1, num_workers=1);
        return test_dataloader;

    def prepare_zoom(self, frame_size: float) -> DataLoader:
        if self.include_zoom == True:
            raise ValueError(f"Cannot setup zoom as test fold when it is included in training data");

        label: pd.DataFrame = self.label;  # load label
        label = label[label["agreement"] >= self.test_agreement];  # filter agreement
        # zoom: pd.DataFrame = label[label["mic"] == "mic"];  # select zoom item
        zoom: pd.DataFrame = label[label["mic"] == "mic"].iloc[:200];  # FIXME:
        
        # get scores and file path
        scores: np.ndarray = zoom[self.score_cols].values.astype(float);
        paths: np.ndarray = zoom["path"].values;
        
        if self.use_soft_target:
            zoom: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": (
                        self.smoothing_param + Tensor(e.astype(float))
                    ).divide(
                        self.smoothing_param * len(Tensor(e.astype(float))) + Tensor(e.astype(float)).sum()
                    )
                }
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];
        else:
            zoom: List[Dict[str, Union[str, Tensor]]] = [
                {
                    "feature": f, 
                    "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
                } 
                for f, e in zip(paths, scores) 
                if e.astype(float).sum() != 0.
            ];

        # prepare Zoom
        print("Preparing Zoom Samples");
        zoom_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(zoom):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            zoom_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        zoom_dataloader: DataLoader = DataLoader(zoom_samples, batch_size=1, num_workers=1);
        return zoom_dataloader;


    def prepare_iemocap(self, frame_size: float, turn_type: str = "all") -> DataLoader:
        # unpack label
        label_path: str = None;
        for k in self.cross_corpus.keys():
            if k.lower().strip() == "iemocap":
                label_path = self.cross_corpus[k];
        
        # check if label_path exists
        if label_path is None:
            raise ValueError(f"Cannot call `prepare_iemocap` if not provided in self.cross_corpus");
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"`label_path` does not exists at {label_path}");

        iemocap: pd.DataFrame = pd.read_csv(label_path);
        
        # filter iemocap
        # iemocap = iemocap[iemocap["dominant_emotion"].isin([e.replace("_score", "") for e in self.score_cols])];
        iemocap = iemocap[iemocap["dominant_emotion"].isin([e.replace("_score", "") for e in self.score_cols])].iloc[:300];  # FIXME:
        if turn_type == "impro":
            iemocap = iemocap[iemocap["name"].str.contains("impro")];
        elif turn_type == "script":
            iemocap = iemocap[iemocap["name"].str.contains("script")];

        scores: np.ndarray = iemocap[self.score_cols].values.astype(float);
        paths: np.ndarray = iemocap["path"].values;
        iemocap: List[Dict[str, Union[str, Tensor]]] = [
            {
                "feature": f, 
                "emotion": Tensor([1. if p == max(e) else 0. for p in e.astype(float)])
            } 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];

        print("Preparing IEMOCAP Samples");
        iemocap_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(iemocap):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample, test=True);
            iemocap_samples.append(self.packer(feature, frame_size=frame_size, test=True));
      
        iemocap_dataloader: DataLoader = DataLoader(iemocap_samples, batch_size=1, num_workers=1);
        return iemocap_dataloader;
        
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
        
