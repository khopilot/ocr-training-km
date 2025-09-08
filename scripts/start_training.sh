#!/bin/bash
# Start Khmer OCR training on SaladCloud

echo "🚀 Starting Khmer OCR Training"
echo "================================"

# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Training configuration
TRAINING_MODE=${1:-"recognition"}  # "recognition" or "detection"
BATCH_SIZE=${2:-8}  # Start small, can increase based on GPU memory
EPOCHS=${3:-50}
LEARNING_RATE=${4:-0.001}

echo "Configuration:"
echo "  Mode: $TRAINING_MODE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Check GPU
echo "🔍 Checking GPU status..."
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv

# Navigate to project root
cd /ocr-training-km

# Step 1: Verify setup
echo -e "\n📋 Step 1: Verifying setup..."
python scripts/verify_setup.py

# Step 2: Prepare data lists
echo -e "\n📋 Step 2: Preparing data lists..."
python scripts/prepare_data_lists.py

# Step 3: Download pre-trained models (if not already done)
echo -e "\n📋 Step 3: Checking pre-trained models..."
if [ ! -d "models/pretrained" ]; then
    bash scripts/download_pretrained.sh
else
    echo "✅ Pre-trained models already downloaded"
fi

# Step 4: Start training
echo -e "\n📋 Step 4: Starting training..."

if [ "$TRAINING_MODE" == "recognition" ]; then
    echo "Training recognition model for Khmer text..."
    
    # Use the simple demo script for initial testing
    python train/train_demo.py \
        --config train/configs/rec_kh.yaml \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --data_list data/all_train_list.txt \
        --val_list data/all_val_list.txt \
        --save_dir models/khmer_rec \
        --log_dir logs/rec_training \
        --pretrained_model models/pretrained/ch_PP-OCRv4_rec_train/best_accuracy \
        --use_gpu true
        
elif [ "$TRAINING_MODE" == "detection" ]; then
    echo "Training detection model for Khmer text..."
    
    python train/train_demo.py \
        --config train/configs/dbnet.yaml \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --data_list data/all_train_list.txt \
        --val_list data/all_val_list.txt \
        --save_dir models/khmer_det \
        --log_dir logs/det_training \
        --pretrained_model models/pretrained/ch_PP-OCRv4_det_train/best_accuracy \
        --use_gpu true
        
else
    echo "❌ Invalid training mode: $TRAINING_MODE"
    echo "Please use 'recognition' or 'detection'"
    exit 1
fi

echo -e "\n✅ Training completed!"
echo "Check logs in: logs/${TRAINING_MODE}_training/"
echo "Model saved in: models/khmer_${TRAINING_MODE}/"

# Optional: Run evaluation
echo -e "\n📊 Running evaluation..."
if [ "$TRAINING_MODE" == "recognition" ]; then
    python eval/evaluate.py \
        --model_dir models/khmer_rec \
        --test_list data/val_list.txt \
        --mode recognition
else
    python eval/evaluate.py \
        --model_dir models/khmer_det \
        --test_list data/val_list.txt \
        --mode detection
fi