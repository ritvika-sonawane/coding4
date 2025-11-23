# Training Guide for Speech Recognition Assignment

## Overview
This guide explains how to train and evaluate ASR models for the different checkpoints.

## Checkpoint Requirements

### Checkpoint 1: Monolingual (English Only)
- **Target**: WER < 60%
- **Dataset**: `train_monolingual`, `test_monolingual`
- **Config**: `conf/mono_optimized.yaml`

### Checkpoint 2: Language ID Implementation
- **Requirement**: Correctly prepend [ENG]/[ITA] tags
- **Already implemented** in `loader.py`

### Checkpoint 3: Bilingual WER
- **Target**: WER < 30% for 2 points
- **Dataset**: `train_bilingual`, `test_bilingual`  
- **Config**: `conf/bili_optimized.yaml`

### Checkpoint 4: Language Identification Accuracy
- **Target**: ACC > 90% for 1 point
- **Evaluated on**: Bilingual test set
- **Config**: `conf/bili_optimized.yaml`

## Training Instructions

### 1. Train Monolingual Model (Checkpoint 1)

```bash
# Start training
bash train_mono.sh

# Or with custom settings
python train.py \
    --config conf/mono_optimized.yaml \
    --tag train_mono \
    --mode monolingual \
    --train_json dump/raw/train_monolingual/data.json \
    --valid_json dump/raw/dev_monolingual/data.json
```

**Expected timeline**: ~6-12 hours on a single GPU

### 2. Train Bilingual Model (Checkpoints 3 & 4)

```bash
# Start training
bash train_bili.sh

# Or with custom settings
python train.py \
    --config conf/bili_optimized.yaml \
    --tag train_bili \
    --mode multilingual \
    --train_json dump/raw/train_bilingual/data.json \
    --valid_json dump/raw/dev_bilingual/data.json
```

**Expected timeline**: ~12-24 hours on a single GPU (larger model)

## Decoding Instructions

### Decode Monolingual Model

```bash
# Find best checkpoint
ls exp/train_train_mono/ckpts/

# Decode (update epoch number as needed)
python decode.py \
    --exp_dir exp/train_train_mono \
    --ckpt_name epoch59.pth \
    --decode_tag test \
    --recog_json dump/raw/test_monolingual/data.json \
    --mode monolingual

# Check results
cat exp/train_train_mono/decode_test_epoch59.pth/decoded_hyp.txt
```

### Decode Bilingual Model

```bash
# Find best checkpoint
ls exp/train_train_bili/ckpts/

# Decode
python decode.py \
    --exp_dir exp/train_train_bili \
    --ckpt_name best_model.pth \
    --decode_tag test \
    --recog_json dump/raw/test_bilingual/data.json \
    --mode multilingual

# Check results
cat exp/train_train_bili/decode_test_best_model.pth/decoded_hyp.txt
```

## Hyperparameter Tuning Tips

### Key Hyperparameters for Better Performance

1. **Model Size**
   - `hidden_dim`: 256 (mono) → 512 (bili)
   - `attention_heads`: 4 (mono) → 8 (bili)
   - `eblocks`: 12 (mono) → 16 (bili)

2. **Conformer Features**
   - `econformer_kernel_size`: 31 (enables convolutions)
   - `eposition_embedding_type`: "relative" (better than absolute)

3. **Optimization**
   - `lr`: 1.5e-3 to 2e-3
   - `warmup_steps`: 10000-20000
   - `label_smoothing`: 0.1

4. **Training**
   - `nepochs`: 60-80
   - `batch_bins`: Adjust based on GPU memory
   - `accum_grad`: 2-4 for stable training

### If Performance is Not Meeting Targets

**For Monolingual (WER > 60%)**:
- Increase `nepochs` to 80
- Try `lr: 2.5e-3` with `warmup_steps: 8000`
- Increase `linear_units` to 3072

**For Bilingual (WER > 30%)**:
- Increase `hidden_dim` to 768
- Increase `eblocks` to 20
- Use `accum_grad: 8` for more stable gradients
- Train for 100 epochs
- Try `lr: 1e-3` with longer `warmup_steps: 25000`

**For Language ID (ACC < 90%)**:
- Usually improves with better WER
- Ensure language tags are correctly prepended
- Consider increasing `dblocks` to 8

## Monitoring Training

Training logs are saved in `exp/train_<tag>/logs/train.log`

```bash
# Monitor training progress
tail -f exp/train_train_mono/logs/train.log

# Check tensorboard
tensorboard --logdir exp/train_train_mono/tensorboard
```

Look for:
- Decreasing loss
- Decreasing WER on validation set
- Model should converge around epoch 40-60

## Submission

After training and decoding:

```bash
# Generate submission files
bash prepare_submission.sh

# This will create:
# - decoded_hyp_monolingual.txt (Checkpoint 1)
# - decoded_hyp_bilingual.txt (Checkpoints 3 & 4)
# - All code files in a ZIP
```

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_bins` (try 2000000)
- Reduce `hidden_dim` or `eblocks`
- Enable gradient checkpointing (requires code modification)

### Poor Convergence
- Check learning rate (might be too high/low)
- Verify data loading (check a few samples)
- Ensure warmup is appropriate

### NaN Loss
- Reduce learning rate
- Check for data issues
- Enable gradient clipping (already set to 1.0)

## Advanced Tips

1. **Ensemble**: Train multiple models with different seeds and average predictions
2. **Learning Rate Schedule**: Try cosine annealing after warmup
3. **Data Augmentation**: Already enabled via audio variations
4. **Model Selection**: Use validation WER to select best checkpoint, not just last epoch

