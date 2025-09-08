#!/bin/bash
# Download pre-trained PaddleOCR models for fine-tuning

echo "📥 Downloading pre-trained PaddleOCR models..."
echo "=" * 50

# Create model directories
mkdir -p models/detection
mkdir -p models/recognition
mkdir -p models/pretrained

cd models/pretrained

# Download PaddleOCR v4 models (latest and best)
echo "Downloading PP-OCRv4 detection model..."
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
tar -xf ch_PP-OCRv4_det_train.tar

echo "Downloading PP-OCRv4 recognition model..."
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
tar -xf ch_PP-OCRv4_rec_train.tar

# Download multilingual models (might work better for Khmer)
echo "Downloading multilingual recognition model..."
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_train.tar
tar -xf latin_ppocr_mobile_v2.0_rec_train.tar

# Download lightweight mobile models (faster training)
echo "Downloading lightweight mobile models..."
wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
tar -xf ch_ppocr_mobile_v2.0_det_train.tar

wget -nc https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar
tar -xf ch_ppocr_mobile_v2.0_rec_train.tar

cd ../..

echo "✅ Pre-trained models downloaded to models/pretrained/"
echo ""
echo "Available models for fine-tuning:"
echo "  - ch_PP-OCRv4_det_train: Latest detection model"
echo "  - ch_PP-OCRv4_rec_train: Latest recognition model"
echo "  - latin_ppocr_mobile_v2.0_rec_train: Multilingual recognition"
echo "  - ch_ppocr_mobile_v2.0_det_train: Lightweight detection"
echo "  - ch_ppocr_mobile_v2.0_rec_train: Lightweight recognition"
echo ""
echo "You can now start fine-tuning with these models!"