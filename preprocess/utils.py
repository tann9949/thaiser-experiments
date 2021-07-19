from typing import List
import os
import subprocess


def preprocess_audio(wav_path: str) -> None:
    """
    Run ffmpeg on provided wav_path and save on out_path 
    to make audio 16k sampling rate mono channel. The following 
    bash command is called on background

    ffmpeg -i {wav_path} -ac 1 -ar 16000 {out_path}

    Arguments
    ---------
    wav_path: str
        Path to wav file to preprocess
    out_path: str
        Output wav path
    """
    # get out_path
    out_path: str = format_name(wav_path);
    if not os.path.exists(out_path):
        # mkdir directory is not exists
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path));
        # run ffmpeg command to preprocess
        cmd: List[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", wav_path, "-ac", "1", "-ar", "16000", out_path];
        result: subprocess.CompletedProcess = subprocess.run(cmd);
        if result.returncode == 1:
            raise BrokenPipeError(f"Error running command {' '.join(cmd)}");


def remove_extension(f_name: str) -> str:
    """
    Remove File Extension from name

    Argument
    --------
    f_name: str
        Name of file to remove extension

    Return
    ------
    formatted_name: str
        File name without extension
    """
    splitted: List[str] = f_name.split(".");
    return ".".join(splitted[:-1]) if len(splitted) > 1 else f_name;

    
def format_name(wav_path: str) -> str:
    """
    Format wav_path and get their output. Split absolute path
    swap `/raw/` into `/wav/`.

    Argument
    --------
    wav_path: str
        Absolute path of wav file

    Return
    ------
    formatted_path: str
        Expected output path of preprocessed wav file
    """
    return wav_path.replace("/raw/", "/wav/").replace(".flac", ".wav");
