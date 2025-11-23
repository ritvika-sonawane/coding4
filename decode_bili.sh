#!/bin/bash
# Decoding script for Bilingual model on test set

python decode.py \
    --exp_dir exp/train_train_bili \
    --ckpt_name best_model.pth \
    --decode_tag test \
    --recog_json dump/raw/test_bilingual/data.json \
    --mode multilingual

