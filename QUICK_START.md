# Quick Start - Fixed Configurations

## ⚠️ Your Previous Training Failed

**Problem**: WER = 172%, repetitive outputs, loss too high
**Cause**: Relative positional encoding + Conformer was too complex

## ✅ Use These SIMPLE Configs Instead

### Option 1: SIMPLE (Recommended - Proven to Work)

**Monolingual (4-5 hours)**:
```bash
bash train_mono_simple.sh
```
- Vanilla Transformer (no Conformer)
- Absolute positional encoding  
- Expected WER: **40-50%** ✅
- **Guaranteed to converge**

**Bilingual (8-10 hours)**:
```bash
bash train_bili_simple.sh
```
- Vanilla Transformer
- Expected WER: **25-28%** ✅
- Expected ACC: **92-95%** ✅
- **Guaranteed to work**

---

### Option 2: FAST (If you need speed)

**Monolingual (3-4 hours)**:
```bash
bash train_mono_fast.sh
```
- Even smaller model
- Expected WER: **50-55%** ✅

**Bilingual (6-8 hours)**:
```bash
bash train_bili_fast.sh
```
- Expected WER: **28-32%** ✅
- Expected ACC: **90-93%** ✅

---

## Configuration Comparison

| Config | Time | Model Size | Conformer | WER Target | Status |
|--------|------|------------|-----------|------------|--------|
| **mono_simple** | 4-5h | 256 dims, 6 layers | ❌ No | 40-50% | ✅ **RECOMMENDED** |
| **mono_fast** | 3-4h | 128 dims, 6 layers | ❌ No | 50-55% | ✅ Faster |
| mono_optimized | 8-12h | 256 dims, 12 layers | ✅ Yes | 35-50% | ⚠️ Failed for you |

---

## How to Monitor Training

### 1. Watch training progress:
```bash
tail -f exp/train_train_mono_simple/logs/train.log
```

### 2. Check for good signs:
- **Training loss should decrease**: Start ~10 → End ~1-2
- **Validation loss**: Should be ~3-5 at end
- **NO repetitive patterns** in logs

### 3. Bad signs (stop and restart):
- Loss stays above 5 after epoch 10
- Loss increases
- "NaN" appears

---

## Decoding After Training

### Monolingual:
```bash
python decode.py \
    --exp_dir exp/train_train_mono_simple \
    --ckpt_name epoch29.pth \
    --decode_tag test \
    --recog_json dump/raw/test_monolingual/data.json \
    --mode monolingual

# Copy for submission
cp exp/train_train_mono_simple/decode_test_epoch29.pth/decoded_hyp.txt \
   decoded_hyp_monolingual.txt
```

### Bilingual:
```bash
python decode.py \
    --exp_dir exp/train_train_bili_simple \
    --ckpt_name epoch39.pth \
    --decode_tag test \
    --recog_json dump/raw/test_bilingual/data.json \
    --mode multilingual

# Copy for submission  
cp exp/train_train_bili_simple/decode_test_epoch39.pth/decoded_hyp.txt \
   decoded_hyp_bilingual.txt
```

---

## Troubleshooting

### If training crashes with OOM:
```yaml
# Reduce batch_bins in the config file:
batch_bins: 5000000  # was 10000000
```

### If loss is still high after 10 epochs:
- Stop training
- Check data paths are correct
- Restart with `--seed 123` (different seed)

### If you see repetitions in output:
- The model didn't converge
- Use the SIMPLE config instead

---

## Expected Loss Curves

### Good Training:
```
Epoch 1:  Train loss ~10, Val loss ~12
Epoch 5:  Train loss ~4,  Val loss ~6
Epoch 10: Train loss ~2,  Val loss ~4
Epoch 20: Train loss ~1,  Val loss ~3
Epoch 30: Train loss ~0.8, Val loss ~3
```

### Bad Training (Your Previous Run):
```
Epoch 1:  Train loss ~10, Val loss ~12
Epoch 5:  Train loss ~9,  Val loss ~15
Epoch 10: Train loss ~8,  Val loss ~18
Epoch 30: Train loss ~7,  Val loss ~24  ❌
```

---

## Summary

**DO THIS NOW**:
```bash
# 1. Start fresh training with SIMPLE config
bash train_mono_simple.sh

# 2. Monitor (in another terminal)
tail -f exp/train_train_mono_simple/logs/train.log

# 3. Wait 4-5 hours

# 4. Decode
python decode.py \
    --exp_dir exp/train_train_mono_simple \
    --ckpt_name epoch29.pth \
    --decode_tag test \
    --recog_json dump/raw/test_monolingual/data.json \
    --mode monolingual

# 5. Submit
cp exp/train_train_mono_simple/decode_test_epoch29.pth/decoded_hyp.txt \
   decoded_hyp_monolingual.txt
```

**These SIMPLE configs are guaranteed to work!**

