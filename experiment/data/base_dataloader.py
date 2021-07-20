import os
import pickle
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from .feature.feature_packer import FeaturePacker
from .feature.featurizer import Featurizer


class BaseDataLoader:
    """
    Base class for SER data loader
    """
    def __init__(self, 
        label_path: str,
        featurizer: Featurizer,
        packer: FeaturePacker) -> None:
        """
        Base Data Loader constructor
        """
        self.label_path: str;
        self.label: pd.DataFrame = pd.read_csv(label_path);
        self.featurizer = featurizer;
        self.packer = packer;

        # initialize train, val, test
        # must call self.setup() to instantiate these variables
        self.train: List[Dict[str, Union[Tensor, str]]] = None;
        self.val: List[Dict[str, Union[Tensor, str]]] = None;
        self.test: List[Dict[str, Union[Tensor, str]]] = None;

    def setup_train(self):
        """
        Override this method to declare self.train
        Each instance is a list of sample which is a dictionary format as follow

        Ex.
        self.train = [
            { feature: <Tensor-feature>, emotion: <emotion-array> },
            ...
        ]

        Filter data as you wish by override this data and manipulating self.label
        """
        raise NotImplementedError();

    def setup_val(self):
        """
        Override this method to declare self.val
        Each instance is a list of sample which is a dictionary format as follow

        Ex.
        self.train = [
            { feature: <Tensor-feature>, emotion: <emotion-array> },
            ...
        ]

        Filter data as you wish by override this data and manipulating self.label
        """
        raise NotImplementedError();

    def setup_test(self):
        """
        Override this method to declare self.test
        Each instance is a list of sample which is a dictionary format as follow

        Ex.
        self.train = [
            { feature: <Tensor-feature>, emotion: <emotion-array> },
            ...
        ]

        Filter data as you wish by override this data and manipulating self.label
        """
        raise NotImplementedError();

    def compute_global_stats(self, save_path: str) -> None:
        """
        Compute mean, and std deviation from self.train

        Arguments
        ---------
        save_path: str
            Path to saved statistic
        """
        print("Extracting feature for calculating stats...")
        data: List[Dict[str, Tensor]] = [self.featurizer(sample) for sample in tqdm(self.train)];
        feat_dim: int = data[0]["feature"].shape[0];
        N: int = 0;
        sum_x: Tensor = torch.zeros([feat_dim,], dtype=torch.float64);  # Sigma_i x_i
        sum_x2: Tensor = torch.zeros([feat_dim,], dtype=torch.float64);  # Sigma_i x_i^2

        print("Computing mean and standard deviation...")
        for d in tqdm(data):
            sum_x = sum_x + d["feature"].sum(dim=-1);
            sum_x2 = sum_x2 + d["feature"].square().sum(dim=-1);
            N += d["feature"].shape[-1];
        
        mean: Tensor = sum_x / N;
        std: Tensor = torch.sqrt(sum_x2 / N - mean.square());

        print("Finish calculating stats!")
        with open(save_path, "wb") as f:
            pickle.dump([mean.to(torch.float32), std.to(torch.float32)], f);

    def setup(self):
        """
        Initialize train, val, test samples
        """
        # setup train, val, test
        self.setup_train();
        self.setup_val();
        self.setup_test();

        # compute global stats if necessary
        if self.packer.stats_path is not None:
            if not os.path.exists(self.packer.stats_path):
                print(f"Global statistics `{self.packer.stats_path}` does not exists. Creating one...");
                self.compute_global_stats(self.packer.stats_path);

    def prepare(self, frame_size: int) -> Tuple[
        Dict[str, Union[Tensor, str]],
        Dict[str, Union[Tensor, str]],
        Dict[str, Union[Tensor, str]]
    ]:
        # prepare train
        print("Preparing Training Samples");
        train_samples: List[Dict[str, Union[Tensor, int]]] = list();
        for sample in tqdm(self.train):
            feature: Dict[str, Union[Tensor, int]] = self.featurizer(sample);
            train_samples += self.packer(feature, frame_size=frame_size);

        # prepare val
        print("Preparing Validation Samples");
        val_samples: List[Dict[str, Union[Tensor, int]]] = list();
        for sample in tqdm(self.val):
            feature: Dict[str, Union[Tensor, int]] = self.featurizer(sample);
            val_samples += self.packer(feature, frame_size=frame_size);
            
        # prepare train
        print("Preparing Testing Samples");
        test_samples: List[Dict[str, Union[Tensor, int]]] = list();
        for sample in tqdm(self.test):
            feature: Dict[str, Union[Tensor, int]] = self.featurizer(sample, test=True);
            test_samples += self.packer(feature, frame_size=frame_size);

        return train_samples, val_samples, test_samples
