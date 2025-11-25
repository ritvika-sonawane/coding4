#!/bin/bash
# Optimized monolingual training - Version 2
# Expected: ~50% faster than v1, better convergence

python train.py \
    --config conf/mono_optimized_v2.yaml \
    --train_json dump/raw/train_monolingual/data.json \
    --valid_json dump/raw/dev_monolingual/data.json \
    --tag train_mono_v2 \
    --mode monolingual

