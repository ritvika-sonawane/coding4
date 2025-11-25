#!/bin/bash
# SIMPLE training - Guaranteed to work

python train.py \
    --config conf/mono_simple.yaml \
    --tag train_mono_simple \
    --mode monolingual \
    --train_json dump/raw/train_monolingual/data.json \
    --valid_json dump/raw/dev_monolingual/data.json \
    --seed 42

