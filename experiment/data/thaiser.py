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
    Labels are formatted in hard label (e.g. [0.05, 0.01, 0.74, 0.2])
    """
    def __init__(self, 
        agreement: float,
        fold: int = 0,
        use_soft_target: bool = True,
        smoothing_param: float = 0.,
        train_mic: str = "con",
        test_mic: Optional[str] = None,
        include_fru: bool = False,
        include_zoom: bool = False,
        **kwargs):
        """
        Thai SER DataLoader constructor
        
        agreement: float
        float: int
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
        self.smoothing_param: float = smoothing_param;
        self.use_soft_target: bool = use_soft_target;

        if include_fru:
            self.score_cols: List[str] = ["neutral_score", "angry_score", "happy_score", "sad_score", "frustrated_score"];
        else:
            self.score_cols: List[str] = ["neutral_score", "angry_score", "happy_score", "sad_score"];

        if test_mic is None:
            self.test_mic: str = self.train_mic;
        else:
            self.test_mic: str = test_mic;

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
        self.fold = fold;
        
    def setup_train(self) -> None:
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        if self.include_zoom:
            zoom: pd.DataFrame = label[label["mic"] == "mic"];
        label = label[label["mic"] == self.train_mic];
        
        train: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.train_studios)];
        # train: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.train_studios)].iloc[:50];

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
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.train_mic];
        
        val: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.val_studios)];
        # val: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.val_studios)].iloc[:50];
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
        
    def setup_test(self) -> None:
        label: pd.DataFrame = self.label;
        label = label[label["agreement"] > self.agreement];
        label = label[label["mic"] == self.test_mic];
        
        test: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.test_studios)];
        # test: pd.DataFrame = label[label["studio_id"].map(lambda x: int(x[1:]) in self.test_studios)].iloc[:50];
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

    def prepare_zoom(self, frame_size: float) -> DataLoader:
        if self.include_zoom == True:
            raise ValueError(f"Cannot setup zoom as test fold when it is included in training data");

        label: pd.DataFrame = self.label;  # load label
        label = label[label["agreement"] > self.agreement];  # filter agreement
        zoom: pd.DataFrame = label[label["mic"] == "mic"];  # select zoom item
        # zoom: pd.DataFrame = label[label["mic"] == "mic"].iloc[:50];  # select zoom item
        
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
      
        zoom_dataloader: DataLoader = DataLoader(zoom_samples, batch_size=1, num_workers=0);
        return zoom_dataloader;
