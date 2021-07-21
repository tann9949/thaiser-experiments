from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from torch import Tensor

from .base_dataloader import BaseDataLoader


class ThaiSERSoftLabelLoader(BaseDataLoader):
    """
    THAI SER DataLoader Module use to extract
    training, validation , and testing data
    Labels are formatted in hard label (e.g. [0.05, 0.01, 0.74, 0.2])
    """
    def __init__(self, 
        agreement: float,
        fold: int = 0,
        train_mic: str = "con",
        test_mic: Optional[str] = None,
        include_fru: bool = False,
        include_zoom: bool = False,
        **kwargs):
        """
        Thai SER DataLoader constructor
        
        train_mic: str
        test_mic: str
        include_fru: bool
        label_path: str
        featurizer: Feaurizer
        packer:  FeaturePacker
        """
        super().__init__(**kwargs);
        self.fold = fold;
        self.agreement: float = agreement;
        self.train_mic: str = train_mic;
        self.include_zoom: bool = include_zoom;

        if include_fru:
            self.score_cols: List[str] = ["neutral_score", "angry_score", "happy_score", "sad_score", "frustrated_score"];
        else:
            self.score_cols: List[str] = ["neutral_score", "angry_score", "happy_score", "sad_score"];

        if test_mic is None:
            self.test_mic: str = self.train_mic;
        else:
            self.test_mic: str = test_mic;

        fold_mapping: Dict[int, List[int]] = {
            0: [list(range(1, 71)), list(range(71, 76)), list(range(76, 81))],
            1: [list(range(1, 61)) + list(range(71, 81)), list(range(61, 66)), list(range(66, 71))],
            2: [list(range(1, 51)) + list(range(61, 81)), list(range(51, 56)), list(range(56, 61))],
            3: [list(range(1, 41)) + list(range(51, 81)), list(range(41, 46)), list(range(46, 51))],
            4: [list(range(1, 31)) + list(range(41, 81)), list(range(31, 36)), list(range(36, 41))],
            5: [list(range(1, 21)) + list(range(31, 81)), list(range(21, 26)), list(range(26, 31))],
            6: [list(range(1, 11)) + list(range(21, 81)), list(range(11, 16)), list(range(16, 21))],
            7: [list(range(11, 81)), list(range(1, 6)), list(range(6, 11))],
        };
        self.train_studios, self.val_studios, self.test_studios = fold_mapping[self.fold];

    def set_fold(self, fold: int) -> None:
        self.fold = fold;
        
    def setup_train(self):
        label: pd.DataFrame = self.label;
        if self.include_zoom:
            zoom: pd.DataFrame = self.label[self.label["mic"] == "mic"]
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.train_mic];
        
        train: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.train_studios)];

        if self.include_zoom:
            train = pd.concat([train, zoom], axis=0).reset_index(drop=True);

        scores: np.ndarray = train[self.score_cols].values.astype(float);
        paths: np.ndarray = train["path"].values;
        
        self.train: List[Dict[str, Union[str, Tensor]]] = [
            {"feature": f, "emotion": Tensor(e.astype(float) / e.astype(float).sum())} 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];
        
    def setup_val(self):
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.train_mic];
        
        val: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.val_studios)];
        scores: np.ndarray = val[self.score_cols].values.astype(float);
        paths: np.ndarray = val["path"].values;
        
        self.val: List[Dict[str, Union[str, Tensor]]] = [
            {"feature": f, "emotion": Tensor(e.astype(float) / e.astype(float).sum())} 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];
        
    def setup_test(self):
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.test_mic];
        
        test: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.test_studios)];
        scores: np.ndarray = test[self.score_cols].values.astype(float);
        paths: np.ndarray = test["path"].values;
        
        self.test: List[Dict[str, Union[str, Tensor]]] = [
            {"feature": f, "emotion": Tensor(e.astype(float) / e.astype(float).sum())} 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];


class ThaiSERHardLabelLoader(ThaiSERSoftLabelLoader):
    """
    THAI SER DataLoader Module use to extract
    training, validation , and testing data.
    Labels are formatted in hard label (e.g. [0., 1., 0., 0.])
    """
    def __init__(self,
        **kwargs):
        """
        Thai SER DataLoader constructor
        
        train_mic: str
        test_mic: str
        include_fru: bool
        label_path: str
        featurizer: Feaurizer
        packer:  FeaturePacker
        """
        super().__init__(**kwargs);      
        
    def setup_train(self):
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.train_mic];
        
        train: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in range(1, 71))];
        scores: np.ndarray = train[self.score_cols].values.astype(float);
        paths: np.ndarray = train["path"].values;
        
        self.train: List[Dict[str, Union[str, Tensor]]] = [
            {"feature": f, "emotion": Tensor([1. if p == max(p) else 0. for p in e.astype(float)])} 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];
        
    def setup_val(self):
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.train_mic];
        
        val: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in range(71, 76))];
        scores: np.ndarray = val[self.score_cols].values.astype(float);
        paths: np.ndarray = val["path"].values;
        
        self.val: List[Dict[str, Union[str, Tensor]]] = [
            {"feature": f, "emotion": Tensor([1. if p == max(p) else 0. for p in e.astype(float)])} 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];
        
    def setup_test(self):
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.test_mic];
        
        test: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in range(76, 81))];
        scores: np.ndarray = test[self.score_cols].values.astype(float);
        paths: np.ndarray = test["path"].values;
        
        self.test: List[Dict[str, Union[str, Tensor]]] = [
            {"feature": f, "emotion": Tensor([1. if p == max(p) else 0. for p in e.astype(float)])} 
            for f, e in zip(paths, scores) 
            if e.astype(float).sum() != 0.
        ];
