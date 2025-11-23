#!/bin/bash
# Decoding script for Monolingual model on test set

python decode.py \
    --exp_dir exp/train_train_mono \
    --ckpt_name epoch59.pth \
    --decode_tag test \
    --recog_json dump/raw/test_monolingual/data.json \
    --mode monolingual

