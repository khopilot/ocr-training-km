#!/bin/bash
# Fixed Training Entrypoint for SaladCloud Khmer OCR
# Handles missing environment files and dependency conflicts

set -e  # Exit on error

echo "🚀 Khmer OCR Training Entrypoint (Fixed)"
echo "========================================"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"

# Create environment file if missing
if [ ! -f ~/.bashrc_salad ]; then
    echo "📝 Creating missing environment configuration..."
    cat > ~/.bashrc_salad << 'EOF'
# SaladCloud Environment Variables
export SALAD_MACHINE_ID=${SALAD_MACHINE_ID:-$(hostname)}
export SALAD_CONTAINER_GROUP_ID=${SALAD_CONTAINER_GROUP_ID:-local}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=8
export PATH=$HOME/.local/bin:$PATH
export KENLM_ROOT=$HOME/kenlm
export PATH=$KENLM_ROOT/build/bin:$PATH
EOF
fi

# Source environment
source ~/.bashrc_salad

# Activate virtual environment
VENV_PATH="/opt/venv"
if [ -d "$VENV_PATH" ]; then
    echo "✅ Activating virtual environment at $VENV_PATH"
    source $VENV_PATH/bin/activate
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run salad-ubuntu-setup-fixed.sh first"
    exit 1
fi

# Change to project directory
cd /ocr-training-km

# Environment info
echo -e "\n📊 Environment Information:"
echo "Machine ID: ${SALAD_MACHINE_ID}"
echo "Container Group: ${SALAD_CONTAINER_GROUP_ID}"
echo "Python: $(python --version)"
echo "Working Directory: $(pwd)"
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"

# Check GPU availability
echo -e "\n🔍 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
else
    echo "⚠️  No GPU detected. Training will use CPU (slow)."
fi

# Verify dependencies
echo -e "\n✅ Verifying dependencies:"
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')" || echo "❌ NumPy import failed"
python -c "import paddle; print(f'  PaddlePaddle: {paddle.__version__}')" || echo "❌ PaddlePaddle import failed"
python -c "from paddleocr import PaddleOCR; print('  PaddleOCR: OK')" || echo "❌ PaddleOCR import failed"

# Create necessary directories
echo -e "\n📁 Setting up directories..."
mkdir -p data/raw data/processed data/synth
mkdir -p models/detection models/recognition models/kenlm
mkdir -p logs/training logs/evaluation
mkdir -p outputs/predictions outputs/metrics

# Download pre-trained models if not present
echo -e "\n📥 Checking for pre-trained models..."
if [ ! -f "models/detection/ch_PP-OCRv4_det_infer.tar" ]; then
    echo "Downloading detection model..."
    wget -P models/detection/ https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
    cd models/detection && tar -xf ch_PP-OCRv4_det_infer.tar && cd ../..
fi

if [ ! -f "models/recognition/ch_PP-OCRv4_rec_infer.tar" ]; then
    echo "Downloading recognition model..."
    wget -P models/recognition/ https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
    cd models/recognition && tar -xf ch_PP-OCRv4_rec_infer.tar && cd ../..
fi

# Training configuration
echo -e "\n⚙️  Training Configuration:"
export TRAINING_MODE=${TRAINING_MODE:-"recognition"}  # detection or recognition
export BATCH_SIZE=${BATCH_SIZE:-16}
export LEARNING_RATE=${LEARNING_RATE:-0.001}
export EPOCHS=${EPOCHS:-100}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}
export EVAL_INTERVAL=${EVAL_INTERVAL:-5}

echo "  Mode: ${TRAINING_MODE}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Epochs: ${EPOCHS}"
echo "  Checkpoint Interval: ${CHECKPOINT_INTERVAL}"
echo "  Eval Interval: ${EVAL_INTERVAL}"

# Data preparation
echo -e "\n📊 Data Preparation:"
if [ ! -f "data/processed/train_list.txt" ]; then
    echo "Generating training data list..."
    python scripts/prepare_data.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --split_ratio 0.8
fi

# Start training based on mode
echo -e "\n🏋️ Starting Training..."
if [ "$TRAINING_MODE" == "detection" ]; then
    echo "Training detection model..."
    python -m paddleocr.tools.train \
        -c configs/det/ch_PP-OCRv4_det_student.yml \
        -o Global.pretrained_model=models/detection/ch_PP-OCRv4_det_infer/inference \
        -o Train.dataset.data_dir=data/processed \
        -o Train.dataset.label_file_list=["data/processed/train_list.txt"] \
        -o Eval.dataset.data_dir=data/processed \
        -o Eval.dataset.label_file_list=["data/processed/val_list.txt"] \
        -o Train.loader.batch_size_per_card=${BATCH_SIZE} \
        -o Optimizer.lr.learning_rate=${LEARNING_RATE} \
        -o Global.epoch_num=${EPOCHS} \
        -o Global.save_model_dir=models/detection/khmer \
        -o Global.save_epoch_step=${CHECKPOINT_INTERVAL} \
        -o Global.eval_batch_step=[0, ${EVAL_INTERVAL}] \
        -o Global.use_gpu=true \
        -o Global.device=gpu
        
elif [ "$TRAINING_MODE" == "recognition" ]; then
    echo "Training recognition model..."
    
    # Generate Khmer character dictionary if not present
    if [ ! -f "configs/rec/khmer_dict.txt" ]; then
        echo "Generating Khmer character dictionary..."
        python scripts/generate_khmer_dict.py > configs/rec/khmer_dict.txt
    fi
    
    python -m paddleocr.tools.train \
        -c configs/rec/PP-OCRv4_rec_distillation.yml \
        -o Global.pretrained_model=models/recognition/ch_PP-OCRv4_rec_infer/inference \
        -o Global.character_dict_path=configs/rec/khmer_dict.txt \
        -o Train.dataset.data_dir=data/processed \
        -o Train.dataset.label_file_list=["data/processed/train_list.txt"] \
        -o Eval.dataset.data_dir=data/processed \
        -o Eval.dataset.label_file_list=["data/processed/val_list.txt"] \
        -o Train.loader.batch_size_per_card=${BATCH_SIZE} \
        -o Optimizer.lr.learning_rate=${LEARNING_RATE} \
        -o Global.epoch_num=${EPOCHS} \
        -o Global.save_model_dir=models/recognition/khmer \
        -o Global.save_epoch_step=${CHECKPOINT_INTERVAL} \
        -o Global.eval_batch_step=[0, ${EVAL_INTERVAL}] \
        -o Global.use_gpu=true \
        -o Global.device=gpu
        
else
    echo "❌ Invalid TRAINING_MODE: ${TRAINING_MODE}"
    echo "Please set TRAINING_MODE to 'detection' or 'recognition'"
    exit 1
fi

# Post-training tasks
echo -e "\n📊 Post-training tasks..."

# Export model for inference
echo "Exporting model for inference..."
python -m paddle.distributed.launch \
    --gpus '0' \
    tools/export_model.py \
    -c configs/${TRAINING_MODE}/config.yml \
    -o Global.pretrained_model=models/${TRAINING_MODE}/khmer/best_accuracy \
    -o Global.save_inference_dir=models/${TRAINING_MODE}/khmer_infer/

# Run evaluation
echo "Running evaluation on test set..."
python scripts/evaluate.py \
    --model_dir models/${TRAINING_MODE}/khmer_infer \
    --test_dir data/processed/test \
    --output_dir outputs/metrics \
    --mode ${TRAINING_MODE}

# Upload results to cloud storage (if configured)
if [ ! -z "$SALAD_RESULTS_BUCKET" ]; then
    echo "Uploading results to cloud storage..."
    python scripts/upload_results.py \
        --model_dir models/${TRAINING_MODE}/khmer_infer \
        --metrics_dir outputs/metrics \
        --bucket $SALAD_RESULTS_BUCKET \
        --prefix "khmer-ocr/${SALAD_MACHINE_ID}/$(date +%Y%m%d_%H%M%S)"
fi

echo -e "\n✅ Training completed successfully!"
echo "Model saved to: models/${TRAINING_MODE}/khmer_infer/"
echo "Metrics saved to: outputs/metrics/"

# Keep container alive if in development mode
if [ "$SALAD_DEV_MODE" == "true" ]; then
    echo -e "\n🔧 Development mode - keeping container alive..."
    tail -f /dev/null
fi