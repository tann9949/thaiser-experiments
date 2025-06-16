import multiprocessing as mp
import os
from glob import glob
from typing import List, Optional, Dict, Any

import pandas as pd

from utils import preprocess_audio, remove_extension

EMOTION_MAPPING: Dict[str, str] = {
    "N": "neutral",
    "W": "angry",
    "F": "happy",
    "T": "sad",
    "L": "boredom",
    "E": "disgust",
    "A": "fear"
};


def get_full_path(wav_root: str, name: str) -> str:
    """
    Get EMODB full path from file name

    Argument
    --------
    name: str
        File name to get full path
    
    Return
    ------
    wav_root: str
        Path to `wav` directory
    f_path: str
        Path to given file name
    """
    f_path: str = f"{wav_root}/emodb/wav/{name}.wav";
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"File `{f_path}` does not exists");
    return f_path;


def generate_emodb_label(raw_path: str) -> pd.DataFrame:
    """
    Generate label for EmoDB dataset

    Argument
    --------
    raw_path: str
        Path to raw directory containing EmoDB data

    Return
    ------
    label: pd.DataFrame
        A dataframe containing EmoDB data labels
    """
    # init variables
    wav_root: str = raw_path.replace("/raw", "/wav");  # path to wav dir
    label_path: str = f"{wav_root}/emodb/labels.csv";
    columns: List[str] = ["name", "path", "emotion"] + [e + "_score" for e in EMOTION_MAPPING.values()];  # columns of labels.csv
    
    # creating dataframe
    labels: List[List[Any]] = [];
    for wav in glob(f"{wav_root}/emodb/wav/*.wav"):  # ite
        emo_score: Dict[str, int] = {e + "_score": 0 for e in EMOTION_MAPPING.values()}
        name: str = remove_extension(os.path.basename(wav));
        emo_key: str = name[-2];
        if emo_key not in EMOTION_MAPPING.keys():
            raise NameError(f"Invalid emo_key `{emo_key}` from `{name}` from file name `{wav}`");
        emotion: str = EMOTION_MAPPING[emo_key];
        emo_score[emotion+"_score"] += 1;
        f_path: str = get_full_path(wav_root, name);
        labels.append([name, f_path, emotion] + list(emo_score.values()));
    labels: pd.DataFrame = pd.DataFrame(labels, columns=columns);
    
    # save (overwrite if exists)
    if os.path.exists(label_path):
        print(f"Label file exists at `{label_path}`. Overwriting...");
        os.remove(label_path);
    labels.to_csv(label_path, index=False);
    return labels;


def preprocess_EmoDB(raw_path: str, n_workers: Optional[int] = None) -> None:
    """
    Process EmoDB directory and get soft/hard label.
    Format all audios into 16k sampling rate

    Arguments
    ---------
    raw_path: str
        Path to directory of raw data
    n_workers: Optional[int]
        Number of workers to run process. If not specified, will use all cpu avaliable
    """
    if n_workers is None:
        n_workers: int = mp.cpu_count();
    emodb_root: str = os.path.join(raw_path, "emodb");
    if not os.path.exists(emodb_root):
        raise FileNotFoundError(f"Folder `{emodb_root}` not found, make sure to download the dataset as instructed in README.md");

    audio_files: List[str] = sorted(glob(f"{emodb_root}/**/*.wav", recursive=True));
    pool = mp.Pool(processes=n_workers);
    pool.map(preprocess_audio, audio_files);  # multiprocessing convert

    generate_emodb_label(raw_path);

    # prompt whether to delete old files
    print(f"Finished processing all files. Delete {emodb_root}? [Y/n]", end=" ");
    inp: str = input();
    if inp.strip().lower() == "y":
        os.removedirs(emodb_root);
        print(f"Directory `{emodb_root}` deleted");
    else:
        print(f"Directory `{emodb_root}` kept");
