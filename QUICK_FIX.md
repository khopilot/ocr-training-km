# Quick Fix for SaladCloud Server

## The Problem
- Server uses `/opt/venv` instead of `.venv`
- Makefile hardcoded to use `.venv/bin/python`
- No Docker available on server

## Quick Fix Option 1: Symlink (Immediate)

On the server, run:
```bash
cd /ocr-training-km
ln -s /opt/venv .venv
```

This creates a symlink so all Makefile targets work as-is.

## Quick Fix Option 2: Direct Commands

Skip the Makefile and run directly:

```bash
cd /ocr-training-km
git pull

# 1. Download HF data
python ops/download_hf.py --output-dir data/hf_datasets --priority high

# 2. Convert to PaddleOCR format
python ops/convert_to_paddle.py --input-dir data/hf_datasets --output-dir data/paddle_format --dataset all

# 3. Validate
python ops/validate_dataset.py --data-dir data/paddle_format --charset train/charset_kh.txt --strict

# 4. Train (will use CPU if GPU fails)
python train/run.py --config train/configs/rec_kh_hf.yaml

# 5. Evaluate
python eval/harness.py --test data/paddle_format/recognition/test --report eval/report.json
```

## Quick Fix Option 3: Use New Scripts

```bash
cd /ocr-training-km
git pull

# Run bare-metal training (no Docker)
bash scripts/train_bare_metal.sh

# Or simple Python training
python scripts/simple_train.py
```

## What We Fixed

1. **Makefile** - Now detects Python dynamically
2. **train_bare_metal.sh** - Works without Docker
3. **simple_train.py** - Direct training without imports

## Testing CPU Mode

If GPU still has cuDNN issues:

```bash
# Force CPU mode (works immediately)
python scripts/production_ocr_cpu.py

# Or modify config to use CPU
echo "use_gpu: false" >> train/configs/rec_kh_hf.yaml
python train/run.py --config train/configs/rec_kh_hf.yaml
```

## Complete Training Command

For professional training on the server:

```bash
cd /ocr-training-km
git pull

# Create symlink (one time)
ln -s /opt/venv .venv

# Run full pipeline
make download-hf
make build-corpus
make train-rec  # Uses bare metal, no Docker
make eval
make bench
```

## If Everything Else Fails

Just run these 3 commands:

```bash
# 1. Get data
python ops/download_hf.py --output-dir data/hf_datasets --priority high

# 2. Convert data  
python ops/convert_to_paddle.py --input-dir data/hf_datasets --output-dir data/paddle_format --dataset all

# 3. Train (CPU mode, guaranteed to work)
python train/run.py --config train/configs/rec_kh.yaml
```

This will train a Khmer OCR model even without GPU/Docker!