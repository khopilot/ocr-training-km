# Quick Fix Commands for SaladCloud Server

## On your server terminal, run these commands:

```bash
cd /ocr-training-km
git pull

# 1. IMMEDIATE FIX (Use CPU - works now!)
python scripts/production_ocr_cpu.py

# 2. AUTO-DETECT (Tries GPU, falls back to CPU)
python scripts/production_ocr_auto.py

# 3. DIAGNOSE GPU/cuDNN issues
python scripts/test_paddle_gpu.py

# 4. FIX cuDNN (for GPU acceleration)
bash scripts/fix_cudnn_server.sh
source ~/.bashrc
python scripts/production_ocr.py

# 5. If still having issues after fix
pip uninstall paddlepaddle paddlepaddle-gpu -y
pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/cuda120/stable.html
```

## What each script does:

- **production_ocr_cpu.py** - Forces CPU mode (no cuDNN needed!)
- **production_ocr_auto.py** - Smart detection, uses best available
- **test_paddle_gpu.py** - Shows exactly what's wrong
- **fix_cudnn_server.sh** - Installs missing cuDNN libraries

## Expected Results:

1. CPU mode will work immediately (slower but functional)
2. After running fix_cudnn_server.sh, GPU should work
3. GPU is ~10x faster than CPU for OCR

## Test with:
```bash
# After fixing, test with
python scripts/production_ocr.py

# Or use auto-detection
python scripts/production_ocr_auto.py
```