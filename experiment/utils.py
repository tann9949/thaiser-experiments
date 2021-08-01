import os
from typing import Any, Dict

import yaml
from line_notify import LineNotify

ACCESS_TOKEN = open(f"{os.path.dirname(__file__)}/../LINE_TOKEN.txt").readlines()[0].strip();


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Read .yaml file and stored in dictionary

    Argument
    --------
    yaml_path: str
        Path to yaml to read
    
    Return
    ------
    data: Dict[str, Any]
        Dictionary containing yaml data
    """
    with open(yaml_path, 'r') as stream:
        return yaml.safe_load(stream);


def read_config(config_path: str) -> Dict[str, Any]:
    """
    Read config file to get hyperparameters

    Featurizer
        

    Argument
    --------
    config_path: str
        Path to config file to read in .yaml format
    
    Return
    ------
    hparams: Dict[str, Any]
    """
    config: Dict[str, Any] = load_yaml(config_path);  # load config file

    #### unpack ####

    # data
    data: Dict[str, Any] = config.get("data", {});
    test_mics: str = data.get("test_mics", ["con"]);
    test_zoom: bool = data.get("test_zoom", True);
    dataloader_param: Dict[str, Any] = data.pop("dataloader") if "dataloader" in data.keys() else {};
    include_zoom: bool = dataloader_param.get("include_zoom", False);

    # feautre
    feature: Dict[str, Any] = config.get("featurizer", {"feature_type": "fbank", "feature_param": {}});
    
    # packer
    packer: Dict[str, Any] = config.get("packer", {"max_len": 3, "pad_mode": "dup"});

    # model
    feature_type: str = feature.get("feat_type", "fbank");
    feature_param: Dict[str, Any] = feature.get("feature_param", {});
    if feature_type == "fbank":
        in_channel: int = feature_param.get("num_mel_bins", 64);
    elif feature_type == "spectrogram":
        # TODO:
        pass
    elif feature_type == "mfcc":
        # TODO:
        pass
    max_len: int = packer.get("max_len", 3);
    frame_shift: int = feature_param.get("frame_shift", 10);
    include_fru: bool = dataloader_param.get("include_fru", False);

    n_class: int = 5 if include_fru else 4;
    sequence_length: int = int(max_len * (1000 / frame_shift))

    model_hparam: Dict[str, Any] = {
        "in_channel": in_channel,
        "sequence_length": sequence_length,
        "n_classes": n_class,
        **config.get("model", {})
    }

    # train
    train: Dict[str, Any] = config.get("train", {});
    trainer_param: Dict[str, Any] = train.pop("trainer_param") if "trainer_param" in train.keys() else {};
    batch_size: int = train.get("batch_size", 64);
    n_iteration: int = train.get("n_iteration", 25);
    exp_path: str = train.get("exp_path", "log/exp");

    if test_zoom and include_zoom:
        raise ValueError(f"Cannot parse `test_zoom` and `include_zoom` at the same time")

    return {
        "featurizer": feature,
        "packer": packer,
        "dataloader": dataloader_param,
        "model_hparams": model_hparam,
        "trainer_params": trainer_param,
        "frame_size": frame_shift/1000,
        "test_mics": test_mics,
        "test_zoom": test_zoom,
        "batch_size": batch_size,
        "n_iteration": n_iteration,
        "exp_path": exp_path
    }


def notify_line(template: str) -> None:
    """
    Notify training results to LINE via token
    
    Argument
    --------
    template: str
        String template format to send to
    """
    if ACCESS_TOKEN is None or ACCESS_TOKEN == "":
        return
    notify: LineNotify = LineNotify(ACCESS_TOKEN);
    notify.send("\n\n" + template);
