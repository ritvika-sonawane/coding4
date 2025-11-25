# Training Optimization Notes

## Analysis of Original Training (mono_optimized.yaml)

### Problems Identified:
1. **Initial loss too high**: Started at 500+ (should be 4-7)
   - This was caused by the LabelSmoothingLoss bug (now FIXED)
2. **Slow convergence**: Took 20+ epochs to reach loss < 25
3. **Long training time**: 60 epochs × 1282s/epoch ≈ 21 hours
4. **Excessive warmup**: 15,000 steps is too conservative

### Training Timeline (Original):
- Epoch 0: Loss 388 → 231 val
- Epoch 20: Loss 23 → 22 val  
- Epoch 35: Loss 13 → 19 val (best)
- Epoch 52: Loss 8 → 18 val
- Epoch 59: Loss 7 → 24 val (overfitting)

## Optimizations in mono_optimized_v2.yaml

### 1. Model Architecture (Speed Improvements)
```yaml
eblocks: 8          # Was 12 (-33% encoder compute)
dblocks: 4          # Was 6 (-33% decoder compute)
econformer_kernel: 15  # Was 31 (faster convolutions)
```
**Expected speedup**: ~30-40% per epoch

### 2. Training Schedule (Faster Convergence)
```yaml
lr: 2e-3           # Was 1e-3 (2x higher for faster convergence)
warmup_steps: 8000  # Was 15,000 (reach peak LR sooner)
nepochs: 40        # Was 60 (avoid overfitting plateau)
batch_bins: 6M     # Was 4M (larger batches = fewer steps)
```

### 3. Expected Results with Fixed Loss:
- **Initial loss**: 4-7 (proper KL divergence after bug fix)
- **Epoch 1**: Loss should drop to ~2-3
- **Epoch 10**: Loss should be ~1.5-2.0
- **Epoch 25-35**: Best checkpoint (loss ~1.0-1.5)
- **Total time**: ~8-10 hours (vs 21 hours original)

## Training Command

```bash
bash train_mono_v2.sh
```

## Monitoring

Expected healthy training:
- Epoch 0, Batch 99: loss < 10
- Epoch 0 end: loss 2-4, val 2-5
- Epoch 5: loss 1.5-2.5, val 2-3
- Steady decrease until epoch 25-30

If you see loss > 50 at any point, there's still a bug!

## Comparison

| Metric | Original | Optimized v2 | Improvement |
|--------|----------|--------------|-------------|
| Encoder blocks | 12 | 8 | 33% faster |
| Decoder blocks | 6 | 4 | 33% faster |
| Kernel size | 31 | 15 | 2x faster conv |
| Epochs | 60 | 40 | 33% less |
| Batch size | 4M | 6M | 50% larger |
| Warmup | 15k | 8k | Faster convergence |
| **Est. time** | **21h** | **~8h** | **62% faster** |
| **Initial loss** | **500+** | **4-7** | **Fixed!** |

## Next Steps

1. Run `bash train_mono_v2.sh`
2. Monitor first epoch - should see loss 4-7 → 2-4
3. Best checkpoint likely around epoch 25-35
4. If results good, use same optimizations for bilingual

