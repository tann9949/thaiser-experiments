import multiprocessing as mp
import os
from glob import glob
from typing import Optional, List

from utils import format_name, preprocess_audio, remove_extension


def generate_iemocap_label(raw_path: str) -> None:
    """
    Generate label for IEMOCAP dataset

    Argument
    --------
    raw_path: str
        Path to raw directory containing IEMOCAP data
    """
    pass


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
    generate_iemocap_label(raw_path);
