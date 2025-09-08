#!/bin/bash
set -euo pipefail

# Khmer OCR Training Setup for SaladCloud Ubuntu
# Run this script in your SaladCloud Ubuntu instance JupyterLab terminal

echo "🥗 SaladCloud Ubuntu Setup for Khmer OCR Training"
echo "=================================================="

# Check if we're running in SaladCloud Ubuntu environment
if [[ -f "/etc/lsb-release" ]]; then
    echo "✅ Ubuntu environment detected"
    cat /etc/lsb-release
else
    echo "⚠️  Non-Ubuntu environment detected, proceeding anyway..."
fi

# Check GPU availability
echo ""
echo "🔍 Checking GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers available"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "❌ NVIDIA drivers not found - this might be an issue"
fi

# Install system dependencies (skip Python - using existing venv)
echo ""
echo "📦 Installing system dependencies..."
echo "💡 Using existing Python environment: $(python --version 2>/dev/null || echo 'Python not found, will use python3')"

if [[ $EUID -eq 0 ]]; then
    # Running as root, no sudo needed
    apt-get update
    apt-get install -y \
        python3-dev \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        libgomp1 \
        libopenblas-dev \
        libboost-all-dev \
        libeigen3-dev \
        swig \
        pkg-config \
        htop \
        tree

    echo "✅ System dependencies installed"
else
    # Running as non-root, use sudo
    sudo apt-get update
    sudo apt-get install -y \
        python3-dev \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        libgomp1 \
        libopenblas-dev \
        libboost-all-dev \
        libeigen3-dev \
        swig \
        pkg-config \
        htop \
        tree

    echo "✅ System dependencies installed"
fi

# Repository should already be cloned and we're in it
echo ""
echo "📥 Khmer OCR repository setup..."
echo "✅ Repository already available in current directory"

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
pip install paddlepaddle-gpu==2.6.1
pip install paddleocr==2.7.0
pip install onnxruntime-gpu==1.18.1
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install huggingface_hub>=0.20.0
pip install sentencepiece==0.1.99
pip install pyyaml requests tqdm

# Install project
pip install -e .

# Build KenLM
echo ""
echo "🔧 Building KenLM..."
if [[ ! -f "/usr/local/bin/lmplz" ]]; then
    cd /tmp
    git clone https://github.com/kpu/kenlm.git
    cd kenlm
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    if [[ $EUID -eq 0 ]]; then
        make install
        ldconfig
    else
        sudo make install
        sudo ldconfig
    fi
    cd /ocr-training-km
    echo "✅ KenLM installed"
else
    echo "✅ KenLM already installed"
fi

# Pre-cache HuggingFace tokenizer
echo ""
echo "🤗 Pre-caching HuggingFace tokenizer..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('khopilot/km-tokenizer-khmer')
    tokenizer.save_pretrained('lang/tokenizer/khopilot')
    print('✅ Tokenizer cached successfully')
except Exception as e:
    print(f'⚠️  Tokenizer caching failed: {e}')
" || echo "⚠️  Tokenizer caching failed, will download during training"

# Pre-download PaddleOCR models
echo ""
echo "📥 Pre-downloading PaddleOCR models..."
python -c "
from paddleocr import PaddleOCR
try:
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
    print('✅ PaddleOCR models cached')
except Exception as e:
    print(f'⚠️  PaddleOCR caching failed: {e}')
" || echo "⚠️  PaddleOCR model caching failed, will download during training"

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p {models,data,logs,lang/{kenlm,tokenizer}}

# Set environment variables
echo ""
echo "🔧 Setting up environment..."
cat > ~/.bashrc_salad << 'EOF'
# Khmer OCR Training Environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVIDIA_VISIBLE_DEVICES=all
export SERVICE_VARIANT=paddle
export PRODUCTION_MODE=prod
export USE_GPU=true
export NUM_GPUS=8
export BATCH_SIZE_PER_GPU=16
export TOTAL_BATCH_SIZE=128
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export WEBHOOK_SECRET_KEY="mJdnzFePyJ68JrW5XY73IVASOdZ9OZguxiCJc/I3NOzL5XgiwbHgyGzhO47dkWgg7J405IdxyNiUB7NNWvx3vA=="
export MODEL_OUTPUT_URL="https://webhook.site/test"
EOF

echo "source ~/.bashrc_salad" >> ~/.bashrc

# Make scripts executable
chmod +x deploy/salad/*.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Source environment: source ~/.bashrc_salad"
echo "   2. Check GPUs: nvidia-smi"
echo "   3. Start training: bash deploy/salad/training-entrypoint.sh"
echo ""
echo "💰 Estimated training cost: ~\$15 for 6-hour training"
echo "📊 Monitor progress with: tail -f logs/*.log"
echo "🔗 Access URL: https://olive-panzanella-akgrsolz439ub3xz.salad.cloud"