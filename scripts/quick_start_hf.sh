#!/bin/bash
# Quick start training with HuggingFace setup

echo "🚀 Quick Start: Khmer OCR Training with HuggingFace"
echo "===================================================="

# Set environment
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Configuration
MODE=${1:-"recognition"}
BATCH_SIZE=${2:-4}
EPOCHS=${3:-10}
NUM_SAMPLES=${4:-500}

echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Synthetic Samples: $NUM_SAMPLES"
echo ""

# Navigate to project root
cd /ocr-training-km

# Step 1: Install Khmer fonts (needed for synthetic data)
echo "📦 Installing Khmer fonts..."
if [ "$EUID" -eq 0 ]; then
    apt-get update && apt-get install -y fonts-noto fonts-noto-cjk fonts-khmeros fonts-khmeros-core
else
    sudo apt-get update && sudo apt-get install -y fonts-noto fonts-noto-cjk fonts-khmeros fonts-khmeros-core
fi

# Step 2: Generate synthetic data
echo -e "\n🎨 Generating synthetic Khmer training data..."
python scripts/generate_khmer_synthetic.py $NUM_SAMPLES

# Step 3: Check data
echo -e "\n📊 Data summary:"
if [ -f "data/train_list.txt" ]; then
    TRAIN_COUNT=$(wc -l < data/train_list.txt)
    VAL_COUNT=$(wc -l < data/val_list.txt)
    echo "  Training samples: $TRAIN_COUNT"
    echo "  Validation samples: $VAL_COUNT"
else
    echo "  No data found!"
    exit 1
fi

# Step 4: Start training with HuggingFace config
echo -e "\n🤗 Starting training with HuggingFace configuration..."
python scripts/train_with_hf.py \
    --mode $MODE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --num_samples $NUM_SAMPLES

echo -e "\n✅ Training pipeline complete!"
echo ""
echo "Your model is saved in: models/khmer_${MODE}_hf/"
echo ""
echo "To test your model:"
echo "  python infer/predict.py --model models/khmer_${MODE}_hf --image <test_image.png>"
echo ""
echo "To start the API server:"
echo "  python service/app.py --model models/khmer_${MODE}_hf"