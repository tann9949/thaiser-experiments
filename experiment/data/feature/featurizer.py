from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import random

import torchaudio
from torch import Tensor
from torchaudio.compliance import kaldi


class Featurizer:
    """
    Base featurizer class for each specific dataset
    """
    def __init__(
        self,
        feature_type: str,
        feature_param: Dict[str, Any],
        sampling_rate: int = 16000,
        vtlp_range: Optional[Tuple[float, float]] = None,) -> None:
        """
        Featurizer constructor

        Arguments
        ---------
        feature_type: str
            Type of feature for featurizer to extract. (raw|spectrogram|fbank|mfcc)
        feature_param: Dict[str, Any]
            Parameter of featurizer. We use torchaudio.complicance.kaldi as a backend
        sampling_rate: int
            Target sampling rate of audio signal
        vtlp_range: Optional[Tuple[float, float]]
            If parsed, fbank or mfcc feature will be vtln warped according to random number generated from provided range
        """
        self.feature_type: str = feature_type;
        self.feature_param: Dict[str, Any] = feature_param;
        self.sampling_rate: int = sampling_rate;
        self.vtlp_range: Tuple[float, float] = vtlp_range;

        if feature_type.lower().strip() == "raw":
            self.featurizer: Callable = None;
        elif feature_type.lower().strip() == "spectrogram":
            self.featurizer: Callable = kaldi.spectrogram;
        elif feature_type.lower().strip() == "fbank":
            self.featurizer: Callable = kaldi.fbank;
        elif feature_type.lower().strip() == "mfcc":
            self.featurizer: Callable = kaldi.mfcc;
        else:
            raise NameError(f"Unrecognized feature type `{feature_type}`. Only raw|spectrogram|fbank|mfcc available");

    def __call__(self, sample: Dict[str, str], test: bool = False) -> Dict[str, Tensor]:
        """
        Extract Feature according to `feature_type` attribute
        
        Argument
        --------
        sample: Dict[str, str]
            Dictionary contains wav_path and its emotion ({"feature": "<path>", "emotion": Tensor([0., 0., 1., 0.])})
        test: bool
            Specify whether to override disable vtlp augmentation

        Return
        ------
        feature: Dict[str, Tensor]
            Return a dictionary of 2 fields: "feature", a feature Tensor, 
            and "emotion" a label encoded emotion
        """
        # unpack sample
        wav_path: str = sample["feature"];
        emotion: Tensor = sample["emotion"];

        wav, sampling_rate = torchaudio.load(wav_path);  # load audio
        # if wav is not mono
        if wav.shape[0] != 1:
            wav: Tensor = wav.mean(0, keepdim=True);
        # if sampling rate mismatch
        if sampling_rate != self.sampling_rate:
            wav: Tensor = kaldi.resample_waveform(wav, sampling_rate, self.sampling_rate);

        # extracat feature
        if self.featurizer is not None:
            feat_param: Dict[str, Any] = self.feature_param;

            # add vtlp warp factor
            if not test:  # augment if not in test phase
                if (self.feature_type == "mfcc" or self.feature_type == "fbank") and self.vtlp_range is not None:
                    # add VTLN warp randomed from vtlp_range (insert VTLP)
                    warp_factor: float = random.uniform(min(self.vtlp_range), max(self.vtlp_range));
                    if "vtln_warp" in feat_param.keys():
                        feat_param["vtln_warp"] = warp_factor;
                    else:
                        feat_param = {
                            "vtln_warp": warp_factor,
                            **feat_param
                        };

            feature: Tensor = self.featurizer(wav, **feat_param);
            feature = feature.transpose(0, 1);
            return {"feature": feature, "emotion": emotion};
        else:
            return {"feature": wav, "emotion": emotion};
