# THAI-SER Experiments

## Installation
It is recommend to use linux based environment or MacOS with Python 3.10+.

### Install `ffmpeg`
To install package dependencies, run:
```bash
$ apt install ffmpeg  # brew install ffmpeg for MacOS
```

### Create python environment and activate (optional)
```bash
pip install virtualenv
python -m venv venv
source venv/bin/activate
```

### Install python dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation
Before you can run any experiment, it is important to download dataset first. There are four datasets used in this experiments:
1. [THAI SER](https://github.com/vistec-AI/dataset-releases/releases/tag/v1) - Thai Speech Emotion Recognition Dataset form VISTEC.
2. [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) - The Interactive Emotional Dyadic Motion Capture (IEMOCAP), an English Speech Emotion Recognition Dataset.
3. [Emo-DB](http://www.emodb.bilderbar.info/download/) - Berlin Database of Emotional Speech, a German Speech Emotion Recognition Dataset.
4. [EMOVO](http://voice.fub.it/activities/corpora/emovo/index.html) - EMOVO Corpus: an Italian Emotional Speech Database.

Click on the dataset for a download link. For IEMOCAP dataset, user need to submit his/her email to request for dataset permission. You can run `bash prepare_thaiser.sh` to download THAI-SER dataset and EmoDB dataset automatically.

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
|       |   |- lablaut
|       |   |- labsilb
|       |   |- ... <unzipped-emodb-directory>
|       |- EMOVO
|           |- f1
|           |- f2
|           |- ...<unzipped-emovo-directory>
```

Then, to preprocess all data and generate label csv file, run the following command:
```bash
$ python preprocess/preprocess_dataset.py --raw-path dataset/raw
```

## Reproducing Experiments

### Reproducing Baseline (All Emotions)
To reproduce baseline with ALL emotions covered (including frustrated):

```bash
python run_thaiser.py --config-path config/sample-normalize/baseline/baseline.yaml
```

### Reproducing Baseline (4 Basic Emotions)
Reproduce baseline with 4 basic emotions:

```bash
python run_thaiser.py --config-path config/sample-normalize/baseline/four_emotions.yaml
```

### Run hard/soft target with different agreements

```bash
# change soft-label_agreements to hard-label_agreements to use hard agreement
for config in $(find config/sample-normalize/soft-label_agreements -name "*.yaml"); do
  python run_thaiser.py --config-path $config
done
```

### Training SER model curriculum learning

```bash
# curriculum-baseline.yaml: Normal curriculum learning
# curriculum-hard.yaml: Curriculum learning using hard label with scheduling bootstrapping
# curriculum-soft.yaml: Curriculum learning using soft label with scheduling bootstrapping
# curriculum-smoothing.yaml: Curriculum learning using hard label + label smoothing with scheduling bootstrapping
python run_curriculum.py --config-path config/curriculum/curriculum-baseline.yaml
```

### Running Cross corpus Experiment

```bash
for config in $(find config/iemocap-crosscorpus -name "*.yaml"); do
  filename=$(basename "$config")
  if [[ $filename == thaiser* ]]; then
    script="run_thaiser.py"
  elif [[ $filename == iemocap* ]]; then
    script="run_iemocap.py"
  else
    echo "Unrecognized config: $filename"
    exit 1
  fi
  python "$script" --config-path "$config"
done
```

## Authors
Chompakorn Chaksangchaichot
Ekapol Chungsuwanich
