#!/bin/bash
# SIMPLE bilingual training - Guaranteed to work

python train.py \
    --config conf/bili_simple.yaml \
    --tag train_bili_simple \
    --mode multilingual \
    --train_json dump/raw/train_bilingual/data.json \
    --valid_json dump/raw/dev_bilingual/data.json \
    --seed 42

