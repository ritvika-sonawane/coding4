#!/bin/bash
# Fast training script for Monolingual model (3-4 hours)

python train.py \
    --config conf/mono_fast.yaml \
    --tag train_mono_fast \
    --mode monolingual \
    --train_json dump/raw/train_monolingual/data.json \
    --valid_json dump/raw/dev_monolingual/data.json \
    --seed 42

