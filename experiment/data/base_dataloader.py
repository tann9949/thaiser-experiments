import os
import pickle
from typing import Dict, List, Tuple, Union, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .feature.feature_packer import FeaturePacker
from .feature.featurizer import Featurizer
from ..model.base_model import BaseModel


class BaseDataLoader:
    """
    Base class for SER data loader
    """
    def __init__(self, 
        label_path: str,
        featurizer: Featurizer,
        packer: FeaturePacker,
        smoothing_param: Optional[float] = None) -> None:
        """
        Base Data Loader constructor
        """
        self.label_path: str;
        self.label: pd.DataFrame = pd.read_csv(label_path);
        self.featurizer: Featurizer = featurizer;
        self.packer: FeaturePacker = packer;
        self.smoothing_param: float = smoothing_param;

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
        
        This is optional because limitation of loading THAISER test with different mic

        Ex.
        self.train = [
            { feature: <Tensor-feature>, emotion: <emotion-array> },
            ...
        ]

        Filter data as you wish by override this data and manipulating self.label
        """
        pass

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
        if not os.path.exists(os.path.dirname(save_path)) and os.path.dirname(save_path) != "":
            os.makedirs(os.path.dirname(save_path));
        with open(save_path, "wb") as f:
            pickle.dump([mean.to(torch.float32), std.to(torch.float32)], f);

    def setup(self) -> None:
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
            else:
                # remove if exists to make it robust for different fold
                print(f"Global statistics `{self.packer.stats_path}` exists. Recomputing statistics...");
                os.remove(self.packer.stats_path);
                self.compute_global_stats(self.packer.stats_path);

    def prepare_train(self, frame_size: float, batch_size: int, model: Optional[BaseModel] = None, eta: Optional[float] = 0.) -> DataLoader:
        """
        Prepare training data

        frame_size: float
            Frame size used when extract features
        batch_size: int
            Batch size for data loader
        model: Optional[BaseModel]
            Model that will be used to rescore labels
        eta: Optional[float]
            Weight of rescored labels. The new labels distribution will be rescored according
            to the following equation:

                y_new = y * (1-eta) + y_model * (eta)

            where y_model = model(X)
        """
        # prepare train
        if model is not None and eta > 0.:
            print(f"Preparing Training Samples with rescoring; eta = {eta}")
        else:
            print("Preparing Training Samples");
        train_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(self.train):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample);
            samples: List[Dict[str, Tensor]] = self.packer(feature, frame_size=frame_size);

            # add packed features into training data
            train_samples += samples

        train_dataloader: DataLoader = DataLoader(train_samples, batch_size=batch_size, num_workers=0, shuffle=True);
            
        # rescore with provided model and factor eta
        if model is not None and eta > 0.:
            new_train_samples = [];
            with torch.no_grad():
                # move to gpu for faster computation
                if torch.cuda.is_available():
                    model.cuda();
                # iterate over batches
                for sample in train_dataloader:
                    feature: Tensor = sample["feature"].cuda() if torch.cuda.is_available() else sample["feature"];  # feature, fbank; cuda if avail
                    emotion: Tensor = sample["emotion"];  # label distribution, R^{Bxn_emotion}

                    pseudo_emotion: Tensor = F.softmax(model(feature).cpu());
                    rescored_emotion: Tensor = emotion * (1 - eta) + pseudo_emotion * (eta);  # R^{Bxn_emotion}

                    assert len(feature) == len(rescored_emotion) == len(sample["name"]);
                    for feat, label, name in zip(feature, rescored_emotion, sample["name"]):
                        new_train_samples.append({"name": name, "feature": feat, "emotion": label})
            
            model.cpu()        
            train_dataloader: DataLoader = DataLoader(new_train_samples, batch_size=batch_size, num_workers=0, shuffle=True);
        
        # smoothing if provided :TODO

            
        return train_dataloader;

    def prepare_val(self, frame_size: float) -> DataLoader:
        # prepare val
        print("Preparing Validation Samples");
        val_samples: List[Dict[str, Union[Tensor, str]]] = list();
        for sample in tqdm(self.val):
            feature: Dict[str, Union[Tensor, str]] = self.featurizer(sample);
            val_samples.append(self.packer(feature, frame_size=frame_size, test=True));

        val_dataloader: DataLoader = DataLoader(val_samples, batch_size=1, num_workers=0);

        return val_dataloader;

    def prepare(self, frame_size: float, batch_size: int, model: Optional[BaseModel] = None, eta: Optional[float] = 0.) -> Tuple[
        DataLoader,
        DataLoader
    ]:
        train_dataloader: DataLoader = self.prepare_train(frame_size=frame_size, batch_size=batch_size, model=model, eta=eta);
        val_dataloader: DataLoader = self.prepare_val(frame_size=frame_size);
            
        return train_dataloader, val_dataloader
