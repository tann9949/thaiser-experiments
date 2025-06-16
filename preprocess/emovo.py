import multiprocessing as mp
import os
from glob import glob
from typing import List, Optional, Dict, Any

import pandas as pd

from utils import preprocess_audio, remove_extension

EMOTION_MAPPING: Dict[str, str] = {
    "dis": "disgust",
    "gio": "happy",  # joy
    "neu": "neutral",
    "pau": "fear",
    "rab": "angry",
    "sor": "surprise",
    "tri": "sad",
}


def get_full_path(wav_root: str, name: str) -> str:
    """
    Get EMOVO full path from file name

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
    spk_id: str = name.split("-")[1];
    name: str = remove_extension(name);
    f_path: str = f"{wav_root}/EMOVO/{spk_id}/{name}.wav"

    if not os.path.exists(f_path):
        raise FileNotFoundError(f"File path does not exists at `{f_path}`");

    return f_path;


def generate_emovo_label(raw_path: str) -> pd.DataFrame:
    """
    Generate label for EMOVO dataset

    Argument
    --------
    raw_path: str
        Path to raw directory containing EMOVO data

    Return
    ------
    label: pd.DataFrame
        A dataframe containing EMOVO data labels
    """
    # init variables
    wav_root: str = raw_path.replace("/raw", "/wav");  # path to wav dir
    label_path: str = f"{wav_root}/EMOVO/labels.csv";
    columns: List[str] = ["name", "path", "emotion"] + [e+"_score" for e in EMOTION_MAPPING.values()];  # columns of labels.csv
    
    # creating dataframe
    labels: List[List[Any]] = [];
    for wav in glob(f"{wav_root}/EMOVO/**/*.wav", recursive=True):  # ite
        score: Dict[str, int] = { e+"_score": 0 for e in EMOTION_MAPPING.values() };
        name: str = remove_extension(os.path.basename(wav));
        f_path: str = get_full_path(wav_root, name);
        emo_key: str = name.split("-")[0];
        if emo_key not in EMOTION_MAPPING.keys():
            raise NameError(f"Invalid emo_key `{emo_key}` from `{name}` from file name `{wav}`");
        emotion: str = EMOTION_MAPPING[emo_key];
        score[emotion+"_score"] += 1;
        labels.append([name, f_path, emotion] + list(score.values()));
    labels: pd.DataFrame = pd.DataFrame(labels, columns=columns);
    
    # save (overwrite if exists)
    if os.path.exists(label_path):
        print(f"Label file exists at `{label_path}`. Overwriting...");
        os.remove(label_path);
    labels.to_csv(label_path, index=False);
    return labels;


def preprocess_EMOVO(raw_path: str, n_workers: Optional[int] = None) -> None:
    """
    Process EMOVO directory and get soft/hard label.
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
    emovo_root: str = os.path.join(raw_path, "EMOVO");
    if not os.path.exists(emovo_root):
        raise FileNotFoundError(f"Folder `{emovo_root}` not found, make sure to download the dataset as instructed in README.md");

    audio_files: List[str] = sorted(glob(f"{emovo_root}/**/*.wav", recursive=True));
    pool = mp.Pool(processes=n_workers);
    pool.map(preprocess_audio, audio_files);  # multiprocessing convert

    generate_emovo_label(raw_path);

    # prompt whether to delete old files
    print(f"Finished processing all files. Delete {emovo_root}? [Y/n]", end=" ");
    inp: str = input();
    if inp.strip().lower() == "y":
        os.removedirs(emovo_root);
        print(f"Directory `{emovo_root}` deleted");
    else:
        print(f"Directory `{emovo_root}` kept");
