import multiprocessing as mp
import os
from glob import glob
from typing import Optional, List, Dict, Any

import pandas as pd

from utils import preprocess_audio, remove_extension

EMOTION_MAPPING: Dict[str, str] = {
    "ang": "anger", 
    "hap": "happiness", 
    "exc": "excited", 
    "sad": "sadness", 
    "fru": "frustration", 
    "fea": "fear", 
    "sur": "surprise", 
    "oth": "other", 
    "neu": "neutral", 
    "dis": "disgust"
};


def get_full_iemocap_path(wav_root: str, name: str) -> str:
    """
    Get IEMOCAP full path from file name

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
    # format full path
    name = remove_extension(name);
    splitted_name: List[str] = name.split("_");
    session: int = int(splitted_name[0][:-1].replace("Ses", ""));
    assert session in range(1, 6), "Invalid session"
    turn_name: str = "_".join(name.split("_")[:-1]);
    f_path: str = f"{wav_root}/IEMOCAP_full_release/Session{session}/sentences/wav/{turn_name}/{name}.wav";
    
    # sanity check if file exists
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"File {f_path} not found");
    return f_path;


def read_iemocap_text(wav_root: str, text: str) -> List[Any]:
    """
    Read IEMOCAP label file

    Argument
    --------
    text: str
        Path to text file of IEMOCAP label

    Return
    ------
    Data: List[Any]
        A list containing data to be put in labels dataframe
    """
    # read lines from file
    with open(text, "r") as f:
        lines: List[str] = [l.strip() for l in f.readlines()];

    # initialize variables
    f_name: str = None;
    dominant_emotion: str = None;
    score: Dict[str, int] = {
        emotion: 0
        for emotion in EMOTION_MAPPING.values()
    };
    sample: List[Any] = None;
    data: List[Any] = [];

    # iterate over line
    for line in lines:
        if len(line) <= 0:  # ignore if line is empty
            continue;
        if line[0] == "[":  # start file
            f_name = line.split("\t")[1].strip();
            dominant_emotion: str = line.split("\t")[2].strip().lower();
            if dominant_emotion in EMOTION_MAPPING.keys():
                dominant_emotion = EMOTION_MAPPING[dominant_emotion];
        elif line[:3] == "C-E":  # get categorical emotion, ignore self-eval
            emotions: str = [e for e in line.split("\t")[1].split(";") if e != ""];  # get emotion chunk
            for emotion in emotions:
                score[emotion.lower().strip()] += 1 / len(emotions);
            path: str = get_full_iemocap_path(wav_root, f_name);
            sample: List[Any] = [f_name, path, dominant_emotion] + list(score.values());
            data.append(sample);
        elif line[0] == "A":
            # reset, we ignore continuous emotion anyway
            f_name = None;
            dominant_emotion = None;
            sample = None;
            score = {
                emotion: 0
                for emotion in EMOTION_MAPPING.values()
            };
        else:
            continue;
    return data;
            

def generate_iemocap_label(raw_path: str) -> pd.DataFrame:
    """
    Generate label for IEMOCAP dataset

    Argument
    --------
    raw_path: str
        Path to raw directory containing IEMOCAP data

    Return
    ------
    label: pd.DataFrame
        A dataframe containing IEMOCAP data labels
    """
    # init variables
    wav_root: str = raw_path.replace("/raw", "/wav");  # path to wav dir
    label_path: str = f"{wav_root}/IEMOCAP_full_release/labels.csv";
    columns: List[str] = [
        "name", "path", "dominant_emotion", "angry", "happy", 
        "excited", "sad", "frustrated", "fear", "surprise", 
        "other", "neutral", "disgust"];  # columns of labels.csv
    
    # creating dataframe
    labels: List[List[Any]] = [];
    for session in range(1, 6):  # iterate over session 1-5
        eval_dir: str = f"{raw_path}/IEMOCAP_full_release/Session{session}/dialog/EmoEvaluation";
        for txt in glob(f"{eval_dir}/*.txt"):
            result = read_iemocap_text(wav_root, txt);
            labels += result;
    labels: pd.DataFrame = pd.DataFrame(labels, columns=columns);
    
    # save (overwrite if exists)
    if os.path.exists(label_path):
        print(f"Label file exists at `{label_path}`. Overwriting...");
        os.remove(label_path);
    labels.to_csv(label_path, index=False);
    return labels;


def preprocess_IEMOCAP(raw_path: str, n_workers: Optional[int] = None) -> None:
    """
    Process IEMOCAP directory and get soft/hard label.
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
    iemocap_root: str = os.path.join(raw_path, "IEMOCAP_full_release");
    if not os.path.exists(iemocap_root):
        raise FileNotFoundError(f"Folder `{iemocap_root}` not found, make sure to download the dataset as instructed in README.md");

    audio_files: List[str] = sorted(glob(f"{iemocap_root}/**/*.wav", recursive=True));
    pool: mp.Pool = mp.Pool(processes=n_workers);

    pool.map(preprocess_audio, audio_files);  # multiprocessing convert
    labels: pd.DataFrame = generate_iemocap_label(raw_path);

    # prompt whether to delete old files
    print(f"Finished processing all files. Delete {iemocap_root}? [Y/n]", end=" ");
    inp: str = input();
    if inp.strip().lower() == "y":
        os.removedirs(iemocap_root);
        print(f"Directory `{iemocap_root}` deleted");
    else:
        print(f"Directory `{iemocap_root}` kept");
