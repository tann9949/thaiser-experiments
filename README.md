# ICASSP 2022 Experiments

## Installation
It is recommend to use linux based environment or MacOS.

To install package dependencies, run:
```bash
$ apt install ffmpeg  # brew install ffmpeg for MacOS
```

## Dataset Preparation
Before you can run any experiment, it is important to download dataset first. There are four datasets used in this experiments:
1. [THAI SER](https://github.com/vistec-AI/dataset-releases/releases/tag/v1) - Thai Speech Emotion Recognition Dataset form VISTEC.
2. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) - The Interactive Emotional Dyadic Motion Capture (IEMOCAP), an English Speech Emotion Recognition Dataset.
3. [Emo-DB](http://www.emodb.bilderbar.info/download/) - Berlin Database of Emotional Speech, a German Speech Emotion Recognition Dataset.
4. [EMOVO]()

Click on the dataset for a download link. For IEMOCAP dataset, user need to submit his/her email to request for dataset permission.

Once all dataset are downloaded, place the directory as follows:
```
|- dataset
|   |- raw
|       |- THAISER
|       |   |- studio1-10
|       |   |- ...
|       |   |- zoom11-20
|       |   |- actor_demography.json
|       |   |- emotion_label.json
|       |- IEMOCAP_full_release
|       |   |- ...
|       |- emodb
|       |   |- ...
|       |- EMOVO
|           |- ...
```

Then, to preprocess all data and generate label csv file, run the following command:
```bash
$ python preprocess/preprocess_dataset.py --raw-path dataset/raw
```

## Usage
*TBD*

## Authors
Chompakorn Chaksangchaichot
Ekapol Chungsuwanich