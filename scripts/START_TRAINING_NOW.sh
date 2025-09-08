#!/bin/bash
# ONE COMMAND TO FIX EVERYTHING AND START TRAINING

echo "🚀 KHMER OCR - FIX EVERYTHING AND START TRAINING"
echo "================================================="
echo ""

cd /ocr-training-km
git pull

# Run the complete setup
echo "Step 1: Fixing all issues..."
bash scripts/setup_everything.sh

echo ""
echo "Step 2: Starting real training..."
python scripts/train_real_hf.py

echo ""
echo "🎉 DONE! Check your model in models/hf_trained/"