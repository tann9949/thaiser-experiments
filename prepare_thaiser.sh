#!/bin/bash

# THAISER
root=dataset/raw/THAISER
mkdir -p $root

for split in "studio1-10" "studio11-20" "studio21-30" "studio31-40" "studio41-50" "studio51-60" "studio61-70" "studio71-80" "zoom1-10" "zoom11-20"; do
    if [[ ! -d "$root/$split" ]]; then
        if [[ ! -f "$root/$split.zip" ]]; then
            wget "https://github.com/vistec-AI/dataset-releases/releases/download/v1/$split.zip" -O "$root/$split.zip"
        fi
        unzip "$root/$split.zip" -d $root
        rm "$root/$split.zip"
    fi
done

if [[ ! -f "emotion_label.json" ]]; then
    wget https://github.com/vistec-AI/dataset-releases/releases/download/v1/emotion_label.json -O "$root/emotion_label.json"
fi

if [[ ! -f "actor_demography.json" ]]; then
    wget https://github.com/vistec-AI/dataset-releases/releases/download/v1/actor_demography.json -O "$root/actor_demography.json"
fi

###

# EmoDB
root=dataset/raw/emodb
mkdir -p $root

if [[ ! -f "$root/download.zip" ]]; then
    wget http://emodb.bilderbar.info/download/download.zip -O "$root/download.zip"
fi

unzip "$root/download.zip" -d $root

###

# python preprocess/preprocess_dataset.py --raw-path dataset/raw