import json
import multiprocessing as mp
import os
from glob import glob
from shutil import copy
from typing import List, Optional, Dict, Any, Union

import pandas as pd

from utils import format_name, preprocess_audio, remove_extension


def get_metadata_from_name(name: str) -> Dict[str, Union[str, int]]:
    """
    Get metadata from file name. Metadata include
    - studio id -> ID of the recording studio
    - mic type -> Type of recorded microphone
    - actor id -> actor ID
    - room id -> room recording ID: A = studio environment, B = normal room environment, Zoom = zoom
    - turn type -> either script or impro

    Argument
    --------
    name: str
        File name to extract metadata

    Return
    ------
    metadata: Dict[str, Union[str, int]]
        Metadata of given file name
    """
    # unpack name
    name: str = os.path.basename(name);
    splitted: List[str] = name.split("_");
    studio: str = splitted[0];
    mic_type: str = splitted[1];
    actor_id: str = splitted[2];
    turn_type: str = splitted[3];

    # process some variables
    if "script" in turn_type:
        turn_type: str = "script";
    elif "impro" in turn_type:
        turn_type: str = "impro";
    else:
        raise NameError(f"Wrong name format: `{name}`")
    actor_id: str = int(actor_id.replace("actor", ""));
    if studio[0] != "z":
        room_id: str = "A" if int(studio[1:]) in range(1, 19) else "B";
    else:
        room_id: str = "zoom";

    # pack to dict
    data: Dict[str, Union[str, int]] = {
        "studio_id": studio,
        "mic": mic_type,
        "actor_id": f"{actor_id:03d}",
        "room_id": room_id,
        "turn_type": turn_type,
    };
    return data


def get_full_path(wav_root: str, f_name: str) -> str:
    """
    Get file's absolute path

    Argument
    --------
    wav_root: str
        Relative path to `wav` directory
    f_name: str
        File name to get absolute path

    Return
    ------
    full_path: str
        Full path of provided `f_name`
    """
    f_name: str = remove_extension(f_name);
    studio_id: str = f_name.split("_")[0];
    mic_id:  str = f_name.split("_")[1];
    
    if studio_id[0] in "s":
        max_range: int = 8;
        prefix: str = "studio";
    elif studio_id[0] in "z":
        max_range: int = 2;
        prefix: str = "zoom";
    else:
        raise NameError(f"Name format unregcognized: `{f_name}`")

    for i in range(max_range):
        studio_range: range = range(10) if i == 0 else range(int(f"{i}1"), int(f"{i+1}0") + 1);
        if int(studio_id[1:]) not in studio_range:
            continue;
        studio_dir: str = f"{prefix}1-10" if i == 0 else f"{prefix}{i}1-{i+1}0";
        wav_path: str = f"{studio_dir}/{prefix}{int(studio_id[1:]):03d}/{mic_id}/{f_name}";
        if wav_path[:-4] != ".wav":
            wav_path += ".wav";
        wav_path = os.path.join(wav_root, wav_path);

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Path `{wav_path}` does not exists")
        return wav_path;


def get_score_from_label(annotated: List[List[str]]) -> Dict[str, float]:
    """
    Calculate unnormalized agreement from annotated label list.
    Annotation score is calculated as follow

    Let 
        a^{(n)}_{i, c} be 1 if annotator i vote for emotion c for nth sample
        N_i be number of emotion annotator i vote for nth sample
    
    Score of emotion c for nth sentence can be calculated as

    score_n(c) = \Sigma_{i \n I} \frac{1}{N_i} \Sigma_{c \in C} a_{i, c}

    Argument
    --------
    annotated: List[List[str]]
        List of annotation from each annotators
    
    Return
    ------
    scores: Dict[str, float]
        A dictionary mapping emotion to its correcponding score
    """
    score: Dict[str, float] = {
        "neutral": 0.,
        "angry": 0.,
        "happy": 0.,
        "sad": 0.,
        "frustrated": 0.,
    };
    for annotator in annotated:
        tmp_score: Dict[str, float] = {
            "neutral": 0.,
            "angry": 0.,
            "happy": 0.,
            "sad": 0.,
            "frustrated": 0.,
        };

        for emotion in annotator:
            if emotion.lower() not in tmp_score.keys():
                continue;  # ignore other
            tmp_score[emotion.lower().strip()] = tmp_score[emotion.lower().strip()] + 1.;
        
        # normalize tmp_score
        normalized_score: float = sum(list(tmp_score.values()));

        # update score
        if normalized_score != 0:
            for e, s in tmp_score.items():
                score[e] = score[e] + s / normalized_score;
    return score;


def preprocess_THAISER(raw_path: str, n_workers: Optional[int] = None) -> None:
    """
    Preprocess THAISER directory. Get emotion as soft label and
    format all audios into 16k sampling rate and format labels
    """
    if n_workers is None:
        n_workers: int = mp.cpu_count();
    thaiser_root: str = os.path.join(raw_path, "THAISER");
    if not os.path.exists(thaiser_root):
        raise FileNotFoundError(f"Folder `{thaiser_root}` not found, make sure to download the dataset as instructed in README.md");

    audio_files: List[str] = sorted(glob(f"{thaiser_root}/**/*.flac", recursive=True));
    pool: mp.Pool = mp.Pool(processes=n_workers);

    pool.map(preprocess_audio, audio_files);  # multiprocessing convert

    # copy labels and metadata
    wav_root: str = format_name(thaiser_root);
    if not os.path.exists(f"{wav_root}/actor_demography.json"):
        copy(f"{thaiser_root}/actor_demography.json", f"{wav_root}/actor_demography.json");
    if not os.path.exists(f"{wav_root}/emotion_label.json"):
        copy(f"{thaiser_root}/emotion_label.json", f"{wav_root}/emotion_label.json");
    
    # specify whether to delete old files
    print(f"Finished processing all files. Delete {thaiser_root}? [Y/n]");
    inp: str = input();
    if inp.strip().lower() == "y":
        os.removedirs(thaiser_root);
        print(f"Directory `{thaiser_root}` deleted")
    else:
        print(f"Directory `{thaiser_root}` kept")
    
    # generate dataset label
    label_path: str = f"{wav_root}/labels.csv"
    if not os.path.exists(label_path):
        # read label, demography file
        with open(f"{wav_root}/emotion_label.json") as f:
            label: Dict[str, Any] = json.load(f);
        with open(f"{wav_root}/actor_demography.json") as f:
            demography: Dict[str, Any] = json.load(f);
        demography: pd.DataFrame = pd.DataFrame(demography["data"]).set_index("Actor's ID");

        # format label and demography
        data: List[Any] = [
            [
                "filename", 
                "path", 
                "studio_id", 
                "mic", 
                "actor_id", 
                "actor_gender",
                "actor_age",
                "room_id", 
                "turn_type", 
                "assigned_emotion",
                "agreement",
                "neutral_score",
                "angry_score",
                "happy_score",
                "sad_score",
                "frustrated_score",
            ]
        ];
        for f_name, label in label.items():
            metadata = get_metadata_from_name(f_name);
            score: Dict[str, float] = get_score_from_label(label[0]["annotated"]);

            data.append([
                remove_extension(f_name),
                get_full_path(wav_root, f_name),
                metadata["studio_id"],
                metadata["mic"],
                metadata["actor_id"],
                demography.loc[metadata["actor_id"]]["Sex"],
                demography.loc[metadata["actor_id"]]["Age"],
                metadata["room_id"],
                metadata["turn_type"],
                label[0]["assigned_emo"],
                label[0]["agreement"],
                score["neutral"],
                score["angry"],
                score["happy"],
                score["sad"],
                score["frustrated"],
            ]);

        data: pd.DataFrame = pd.DataFrame(data);
        data.to_csv(label_path, index=False);
    