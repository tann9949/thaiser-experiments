from typing import List, Dict, Union, Tuple

from torch import Tensor
from tqdm import tqdm
import pandas as pd

from .feature.featurizer import Featurizer
from .feature.feature_packer import FeaturePacker

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

    def setup(self):
        """
        Initialize train, val, test samples
        """
        self.setup_train();
        self.setup_val();
        self.setup_test();

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
