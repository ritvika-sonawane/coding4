#!/bin/bash
# Training script for Monolingual model (Checkpoint 1)

python train.py \
    --config conf/mono_optimized.yaml \
    --tag train_mono \
    --mode monolingual \
    --train_json dump/raw/train_monolingual/data.json \
    --valid_json dump/raw/dev_monolingual/data.json \
    --seed 42

