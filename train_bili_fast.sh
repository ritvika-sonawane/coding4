#!/bin/bash
# Fast training script for Bilingual model (6-8 hours)

python train.py \
    --config conf/bili_fast.yaml \
    --tag train_bili_fast \
    --mode multilingual \
    --train_json dump/raw/train_bilingual/data.json \
    --valid_json dump/raw/dev_bilingual/data.json \
    --seed 42

