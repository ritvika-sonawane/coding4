#!/bin/bash
# Training script for Bilingual model (Checkpoints 3 & 4)

python train.py \
    --config conf/bili_optimized.yaml \
    --tag train_bili \
    --mode multilingual \
    --train_json dump/raw/train_bilingual/data.json \
    --valid_json dump/raw/dev_bilingual/data.json \
    --seed 42

