#!/bin/bash
set -euo pipefail

# Khmer OCR Multi-GPU Training Entrypoint for SaladCloud
# This script orchestrates distributed training across 8x L40S GPUs

echo "🚀 Starting Khmer OCR Training on SaladCloud"
echo "📊 GPU Setup: $NUM_GPUS GPUs, Batch Size: $TOTAL_BATCH_SIZE"
echo "🔧 Mode: $PRODUCTION_MODE, Backend: $SERVICE_VARIANT"

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export NVIDIA_VISIBLE_DEVICES=all

# Create necessary directories
mkdir -p /app/{models,data,logs,lang/{kenlm,tokenizer}}

# Function to send webhook notifications
send_webhook() {
    local status="$1"
    local message="$2"
    local data="$3"
    
    if [[ -n "${WEBHOOK_SECRET_KEY:-}" ]]; then
        curl -X POST "${MODEL_OUTPUT_URL:-https://webhook.site/test}" \
            -H "Content-Type: application/json" \
            -H "X-Webhook-Secret: $WEBHOOK_SECRET_KEY" \
            -d "{
                \"status\": \"$status\",
                \"message\": \"$message\",
                \"timestamp\": \"$(date -Iseconds)\",
                \"data\": $data
            }" || echo "⚠️  Webhook notification failed"
    fi
}

# Function to monitor GPU usage
monitor_gpus() {
    while true; do
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> /app/logs/gpu_usage.log
        sleep 60
    done
}

# Trap for cleanup on exit
cleanup() {
    echo "🧹 Cleaning up training process..."
    pkill -f python || true
    pkill -f nvidia-smi || true
    
    # Upload final logs and models
    send_webhook "cleanup" "Training process terminated" '{"final": true}'
    
    # Sync filesystem
    sync
    echo "✅ Cleanup completed"
}
trap cleanup EXIT

echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
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
    htop \
    nvtop

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

echo "🐍 Setting up Python environment..."
cd /app

# Clone the repository (assuming it's public or use token)
if [[ ! -d "/app/.git" ]]; then
    git clone https://github.com/khopilot/khmer-ocr-v1.git /tmp/repo
    cp -r /tmp/repo/* /app/
    rm -rf /tmp/repo
fi

# Install Python dependencies
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

echo "🔧 Building KenLM..."
if [[ ! -f "/usr/local/bin/lmplz" ]]; then
    git clone https://github.com/kpu/kenlm.git /tmp/kenlm
    cd /tmp/kenlm
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    make install
    ldconfig
    cd /app
    rm -rf /tmp/kenlm
fi

echo "🤗 Pre-loading HuggingFace tokenizer..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('khopilot/km-tokenizer-khmer')
    tokenizer.save_pretrained('/app/lang/tokenizer/khopilot')
    print('✅ Tokenizer cached successfully')
except Exception as e:
    print(f'⚠️  Tokenizer caching failed: {e}')
"

echo "📊 GPU Information:"
nvidia-smi

# Start GPU monitoring in background
monitor_gpus &
MONITOR_PID=$!

echo "📥 Downloading and preparing datasets..."
send_webhook "started" "Beginning data preparation" '{"stage": "data_prep"}'

# Download HuggingFace datasets
python -c "
import os
from datasets import load_dataset
from pathlib import Path

print('📥 Downloading HuggingFace datasets...')
datasets_to_download = [
    'seanghay/khmer-dictionary-44k',
    'seanghay/SynthKhmer-10k'
]

for dataset_name in datasets_to_download:
    try:
        dataset = load_dataset(dataset_name)
        print(f'✅ Downloaded {dataset_name}')
    except Exception as e:
        print(f'⚠️  Failed to download {dataset_name}: {e}')
"

echo "🔤 Building corpus and charset..."
python ops/build_corpus.py \
    --input-dirs data/hf_datasets \
    --output data/corpus/khmer_training_corpus.txt \
    --min-length 2 --max-length 200

python ops/charset.py \
    --corpus data/corpus/khmer_training_corpus.txt \
    --output train/charset_kh.txt

echo "🏗️  Starting distributed training..."
send_webhook "training" "Starting model training" '{"stage": "training", "gpus": 8}'

# Phase 1: Train DBNet Detection Model (GPUs 0-3)
echo "🎯 Phase 1: Training DBNet Detection Model (4 GPUs)"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    train/run.py \
    --config train/configs/dbnet_hf.yaml \
    --output models/dbnet_kh \
    --amp \
    --save_epoch 5 \
    2>&1 | tee /app/logs/dbnet_training.log &

DBNET_PID=$!

# Phase 2: Train Recognition Model (GPUs 4-7) 
echo "🔤 Phase 2: Training Recognition Model (4 GPUs)"
sleep 60  # Stagger start to avoid resource conflicts
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m paddle.distributed.launch \
    --gpus="4,5,6,7" \
    train/run.py \
    --config train/configs/rec_kh_hf.yaml \
    --output models/rec_kh \
    --charset train/charset_kh.txt \
    --amp \
    --save_epoch 5 \
    2>&1 | tee /app/logs/rec_training.log &

REC_PID=$!

# Phase 3: Train KenLM Language Model (CPU parallel)
echo "🧠 Phase 3: Training KenLM Language Model"
sleep 120  # Let GPU training stabilize first
python lang/train_lm.py \
    --corpus data/corpus/khmer_training_corpus.txt \
    --order 5 \
    --output lang/kenlm/khmer_5gram \
    --use-hf-tokenizer \
    --hf-tokenizer khopilot/km-tokenizer-khmer \
    --prune 0 0 1 \
    2>&1 | tee /app/logs/lm_training.log &

LM_PID=$!

# Wait for training processes to complete
echo "⏳ Waiting for training processes..."

# Monitor training progress
while kill -0 $DBNET_PID 2>/dev/null || kill -0 $REC_PID 2>/dev/null || kill -0 $LM_PID 2>/dev/null; do
    echo "📊 Training progress check $(date)"
    
    # Send periodic status updates
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 8 | tr '\n' ',' | sed 's/,$//')
    send_webhook "progress" "Training in progress" "{\"gpu_usage\": \"$gpu_usage\"}"
    
    sleep 300  # Check every 5 minutes
done

wait $DBNET_PID && echo "✅ DBNet training completed" || echo "❌ DBNet training failed"
wait $REC_PID && echo "✅ Recognition training completed" || echo "❌ Recognition training failed"  
wait $LM_PID && echo "✅ Language model training completed" || echo "❌ Language model training failed"

echo "🧪 Running evaluation..."
send_webhook "evaluation" "Starting model evaluation" '{"stage": "evaluation"}'

# Run evaluation on trained models
python eval/harness.py \
    --test data/test \
    --model-dir models \
    --report eval/report.json \
    --gpu --strict \
    2>&1 | tee /app/logs/evaluation.log

echo "📊 Generating manifest..."
python ops/manifests.py --update

echo "📤 Uploading results..."
send_webhook "completed" "Training completed successfully" "{
    \"models\": [\"models/dbnet_kh\", \"models/rec_kh\", \"lang/kenlm/khmer_5gram.arpa\"],
    \"logs\": [\"logs/dbnet_training.log\", \"logs/rec_training.log\", \"logs/lm_training.log\", \"logs/evaluation.log\"],
    \"evaluation\": \"eval/report.json\",
    \"manifest\": \"governance/manifest.json\"
}"

# Archive results for download
echo "📦 Creating results archive..."
tar -czf /app/khmer_ocr_trained_models.tar.gz \
    models/ \
    lang/kenlm/ \
    eval/report.json \
    governance/manifest.json \
    logs/

echo "🎉 Training completed successfully!"
echo "📊 Final GPU utilization:"
nvidia-smi

# Keep container alive for result download
echo "🔄 Keeping container alive for result retrieval..."
python -m http.server 8080 --directory /app &
HTTP_PID=$!

# Create simple health endpoints
cat > /app/health_server.py << 'EOF'
#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "training": "completed"}).encode())
        elif self.path == '/ready':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ready": True}).encode())
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                "training_completed": True,
                "models_available": os.path.exists('/app/models'),
                "results_archive": os.path.exists('/app/khmer_ocr_trained_models.tar.gz'),
                "archive_size_mb": round(os.path.getsize('/app/khmer_ocr_trained_models.tar.gz') / 1024 / 1024, 2) if os.path.exists('/app/khmer_ocr_trained_models.tar.gz') else 0
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
    server.serve_forever()
EOF

python /app/health_server.py