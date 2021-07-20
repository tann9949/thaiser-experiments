from typing import Callable, Dict, Union, List, Optional
from functools import partial
import pickle

import torch
from torch import Tensor

from .padding import pad_dup, pad_constant


class FeaturePacker:
    """
    A function to pack a list of variable length tensor
    into a single tensor for training
    """
    def __init__(
        self,
        pad_mode: str,
        max_len: int,
        stats_path: Optional[str] = None,
        pad_constant: float = 0.) -> None:
        """
        FeaturePacker constructor
        
        Arguments
        ---------
        pad_mode: str
            Padding mode. Two available method constant|dup.
        max_len: int
            Maximum length of packed signal (in second)
        stats_path: Optional[str]
            Path to saved statistic called from self.compute_global_stats(path)
        pad_constant: float
            Padding constant, parse this if pad_mode = constant
        """
        self.pad_mode: str = pad_mode.lower().strip();
        self.max_len: int = max_len;

        if self.pad_mode == "constant":
            self.pad_fn: Callable = partial(pad_constant, constant=pad_constant);
        elif self.pad_mode == "dup":
            self.pad_fn: Callable = pad_dup;

        self.stats_path: Optional[str] = stats_path;

    def compute_global_stats(self, data: List[Dict[str, Tensor]], save_path: str = "stats.pckl") -> None:
        """
        Compute mean, and std deviation of given list of samples

        Arguments
        ---------
        data: List[Dict[str, Tensor]]
            List of sample, dictionary of feature and its label
        save_path: str
            Path to saved statistic
        """
        feat_dim: int = data[0]["feature"].shape[-1];
        N: int = 0;
        sum_x: Tensor = torch.zeros([feat_dim,]);  # Sigma_i x_i
        sum_x2: Tensor = torch.zeros([feat_dim,]);  # Sigma_i x_i^2

        for d in data:
            sum_x = sum_x + d["feature"].sum(dim=0);
            sum_x2 = sum_x2 + d["feature"].square().sum(dim=0);
            N += d["feature"].shape[0];
        
        mean: Tensor = sum_x / N;
        std: Tensor = torch.sqrt(sum_x2 / N - mean.square());

        with open(save_path, "wb") as f:
            pickle.dump([mean, std], f);

    def normalize(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Normalize feature according to given stats path attribute
        into a zero mean and unit variance.
        If stats_path is None, normalized by sample, else
        utterance is normalize per-sample

        # TODO: this normalization must be skipped if use raw audio

        Argument
        --------
        sample: Dict[str, Tensor]
            Sample to be normalized. Dictonary must be in the following format {"feature": "<path>", "emotion": "<emotion>"}
            The tensor must be in the following dimension (seq_len, feat_dim)
        
        Return
        ------
        normalized_sample: Dict[str, Tensor]
            Return the same sample dictionary with normalized feature
        """
        # unpack
        feature: Tensor = sample["feature"];
        emotion: int = sample["emotion"];

        # if global mean, std is given
        if self.stats_path is not None:
            with open(self.stats_path) as f:
                mean, std = pickle.load(f);
        # if mean, std is not given, normalize by sample
        else:
            mean: Tensor = feature.mean(dim=0);
            std: Tensor = torch.sqrt(torch.unsqueeze(feature.var(dim=0), dim=0));

        feature = feature - mean;  # center
        feature = feature / (std + 1e-8);  # scale
        return {"feature": feature, "emotion": emotion};

    def __call__(self, sample: Dict[str, Tensor], frame_size: int):
        """
        Split file according to class attribute, max_len and
        pad the remainding file

        Argument
        --------
        sample: Dict[str, Tensor]
            Sample to be splitted and pad. Dictonary must be in the following format {"feature": "<path>", "emotion": "<emotion>"}
        frame_size: int
            Size of each feature frame in second
        """
        # unpack
        x: Tensor = sample["feature"];  # feature Tensor
        y: int = sample["emotion"];  # emotion label

        # e.g.
        # 1 frame = 10 ms = 0.01 s (10ms overlapping window)
        # max_len = 3 second
        # max_frame = 3 second * (1 frame / 0.01 second) = 300 frames
        max_frame: int = int(self.max_len / frame_size);

        time_dim: int = x.shape[-1];
        x_chopped: List[Tensor] = list();
        for i in range(time_dim):
            if i % max_frame == 0 and i != 0:  # if reach max_frame
                xi: Tensor = x[:, i - max_frame:i];
                assert xi.shape[-1] == max_frame, xi.shape;
                x_chopped.append(self.normalize({"feature": xi, "emotion": y}));
        if time_dim < max_frame:  # if file length not reach max_frame
            if self.pad_fn:
                xi: Tensor = self.pad_fn(x, max_len=max_frame);
                assert xi.shape[-1] == max_frame;
            else:
                xi: Tensor = x;
            x_chopped.append(self.normalize({"feature": xi, "emotion": y}));
        else:  # if file is longer than n_frame, pad remainder
            remainder: Tensor = x[:, x.shape[-1] - x.shape[0] % max_frame:];
            if not remainder.shape[-1] <= self.len_thresh:
                if self.pad_fn:
                    xi: Tensor = self.pad_fn(remainder, max_len=max_frame);
                else:
                    xi: Tensor = x;
                x_chopped.append(self.normalize({"feature": xi, "emotion": y}));
        return x_chopped;
