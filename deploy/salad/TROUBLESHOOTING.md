# SaladCloud Deployment Troubleshooting Guide

## 🔧 Quick Fixes for Common Issues

### Issue 1: Missing ~/.bashrc_salad
**Error:** `bash: /root/.bashrc_salad: No such file or directory`

**Solution:** Use the fixed scripts that create this file automatically:
```bash
bash deploy/salad/salad-ubuntu-setup-fixed.sh
# or
bash deploy/salad/training-entrypoint-fixed.sh
```

### Issue 2: Python 3.12 Compatibility Issues
**Error:** `scipy==1.10.1` not found, or `module 'pkgutil' has no attribute 'ImpImporter'`

**Solution:** Python 3.12 requires different package versions:
```bash
# The setup script auto-detects Python version and uses appropriate requirements
bash deploy/salad/salad-ubuntu-setup-fixed.sh

# Or manually for Python 3.12:
pip install numpy==1.26.4
pip install -r deploy/salad/requirements-py312.txt
pip install --no-deps paddlepaddle-gpu==2.6.1
pip install --no-deps paddleocr==2.7.0
```

**Note:** Key version differences for Python 3.12:
- scipy needs >=1.11.1 (not 1.10.1)
- pandas needs >=2.1.0 (not 2.0.3)
- scikit-image needs >=0.22.0 (not 0.21.0)

### Issue 3: KenLM Directory Already Exists
**Error:** `fatal: destination path 'kenlm' already exists`

**Solution:** The fixed scripts check if KenLM exists before cloning:
```bash
# Manual fix if needed:
cd $HOME
if [ -d "kenlm/build" ]; then
    echo "KenLM already built"
else
    cd kenlm && mkdir -p build && cd build
    cmake .. && make -j$(nproc)
fi
```

### Issue 4: Python 3.12 Compatibility
**Error:** PyMuPDF or other packages failing on Python 3.12

**Solution:** Skip incompatible packages or use alternatives:
```bash
# Skip PyMuPDF if it fails
pip install PyMuPDF==1.23.8 || echo "Skipping PyMuPDF"

# Use compatible versions from requirements-salad.txt
pip install -r deploy/salad/requirements-salad.txt
```

## 📝 Step-by-Step Deployment Instructions

### 1. Fresh Installation
```bash
# Clone repository
git clone https://github.com/khopilot/ocr-training-km.git /ocr-training-km
cd /ocr-training-km

# Run fixed setup script
bash deploy/salad/salad-ubuntu-setup-fixed.sh

# Start training
bash deploy/salad/training-entrypoint-fixed.sh
```

### 2. Fixing Existing Installation
```bash
cd /ocr-training-km

# Pull latest changes
git pull

# Activate virtual environment
source /opt/venv/bin/activate

# Fix numpy conflicts
pip uninstall -y numpy paddlepaddle-gpu paddleocr
pip install numpy==1.23.5
pip install -r deploy/salad/requirements-salad.txt
pip install --no-deps paddlepaddle-gpu==2.6.1
pip install --no-deps paddleocr==2.7.0

# Install missing dependencies
pip install httpx==0.24.1 decorator==5.1.1 astor==0.8.1 opt-einsum==3.3.0 protobuf==3.20.3

# Reinstall project
pip install -e . --no-deps
```

### 3. Verification Commands
```bash
# Check environment
source ~/.bashrc_salad || source /opt/venv/bin/activate

# Verify installations
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"

# Check GPU
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

## 🐛 Debugging Tips

### Check Logs
```bash
# View setup logs
cat /tmp/salad-setup.log

# Monitor training
tail -f logs/training/train.log

# Check system resources
htop
nvidia-smi -l 1  # Monitor GPU usage
```

### Environment Variables
```bash
# Required for SaladCloud
export SALAD_MACHINE_ID=$(hostname)
export SALAD_CONTAINER_GROUP_ID=local
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Optional
export SALAD_MODE=training  # or 'api'
export SALAD_DEV_MODE=true  # Keep container alive after training
```

### Manual Dependency Resolution
```bash
# If automatic installation fails, install manually in order:
pip install numpy==1.23.5
pip install opencv-python==4.6.0.66
pip install Pillow==10.4.0
pip install --no-deps paddlepaddle-gpu==2.6.1
pip install httpx decorator astor opt-einsum protobuf
pip install --no-deps paddleocr==2.7.0
pip install -r deploy/salad/requirements-salad.txt --no-deps
```

## 🚀 Quick Start Commands

### Development Mode
```bash
# Keep container alive for debugging
export SALAD_DEV_MODE=true
bash /ocr-training-km/start.sh
```

### Training Mode
```bash
export SALAD_MODE=training
export TRAINING_MODE=recognition  # or 'detection'
export BATCH_SIZE=16
export EPOCHS=100
bash /ocr-training-km/start.sh
```

### API Server Mode
```bash
export SALAD_MODE=api
bash /ocr-training-km/start.sh
```

## 📊 Monitoring Training Progress

### TensorBoard
```bash
# In a separate terminal or tmux session
tensorboard --logdir=logs/training --host=0.0.0.0 --port=6006
```

### Training Metrics
```bash
# View latest metrics
tail -n 50 logs/training/metrics.json | jq '.'

# Check model checkpoints
ls -lah models/recognition/khmer/
```

## 🆘 Still Having Issues?

1. **Check GitHub Issues:** https://github.com/khopilot/ocr-training-km/issues
2. **Review Logs:** All errors are logged to `/tmp/salad-*.log`
3. **Test Locally First:** Use Docker to replicate SaladCloud environment
4. **Contact Support:** Include full error logs and environment details

## 🐳 Local Testing with Docker

```bash
# Build test container
docker build -f deploy/salad/Dockerfile -t khmer-ocr-salad .

# Run locally
docker run --gpus all -it \
    -e SALAD_MACHINE_ID=local \
    -e SALAD_DEV_MODE=true \
    -v $(pwd)/data:/ocr-training-km/data \
    khmer-ocr-salad
```

## 📋 Checklist Before Deployment

- [ ] Tested setup script locally
- [ ] Verified numpy version compatibility
- [ ] KenLM builds successfully
- [ ] GPU is detected and accessible
- [ ] All Python imports work
- [ ] Training data is prepared
- [ ] Model checkpoints directory exists
- [ ] Environment variables are set
- [ ] Virtual environment is activated
- [ ] Project is installed in editable mode