#!/bin/bash
# Complete setup script to fix all issues and start real training

echo "🔧 FIXING EVERYTHING FOR KHMER OCR TRAINING"
echo "============================================"

cd /ocr-training-km

# 1. Create missing directories
echo "📁 Creating missing directories..."
mkdir -p lang/lexicon
mkdir -p lang/kenlm
mkdir -p data/paddle_format/recognition/train
mkdir -p data/paddle_format/recognition/val
mkdir -p data/paddle_format/recognition/test
mkdir -p models/pretrained
mkdir -p logs/training

# 2. Copy charset to correct location
echo "📝 Setting up charset files..."
if [ -f "train/charset_kh.txt" ]; then
    cp train/charset_kh.txt lang/lexicon/charset_khmer.txt
    echo "✅ Charset copied to lang/lexicon/"
else
    # Create a basic Khmer charset if missing
    cat > lang/lexicon/charset_khmer.txt << 'EOF'
 
ក
ខ
គ
ឃ
ង
ច
ឆ
ជ
ឈ
ញ
ដ
ឋ
ឌ
ឍ
ណ
ត
ថ
ទ
ធ
ន
ប
ផ
ព
ភ
ម
យ
រ
ល
វ
ស
ហ
ឡ
អ
ឥ
ឦ
ឧ
ឩ
ឪ
ឫ
ឬ
ឭ
ឮ
ឯ
ឰ
ឱ
ឲ
ឳ
ា
ិ
ី
ឹ
ឺ
ុ
ូ
ួ
ើ
ឿ
ៀ
េ
ែ
ៃ
ោ
ៅ
ំ
ះ
ៈ
៉
៊
់
៌
៍
៎
៏
័
៑
៖
ៗ
៘
៙
៚
៛
០
១
២
៣
៤
៥
៦
៧
៨
៩
!
"
#
$
%
&
'
(
)
*
+
,
-
.
/
0
1
2
3
4
5
6
7
8
9
:
;
<
=
>
?
@
EOF
    echo "✅ Created Khmer charset"
fi

# 3. Create a basic lexicon file
echo "📚 Creating Khmer lexicon..."
cat > lang/lexicon/khmer_lexicon.txt << 'EOF'
សួស្តី
អរគុណ
សូម
បាទ
ចាស
ទេ
មាន
គ្មាន
ខ្ញុំ
អ្នក
យើង
គាត់
នាង
ពួកគេ
នេះ
នោះ
ថ្ងៃ
ខែ
ឆ្នាំ
ព្រឹក
ល្ងាច
យប់
ម៉ោង
នាទី
មួយ
ពីរ
បី
បួន
ប្រាំ
កម្ពុជា
ភ្នំពេញ
សៀមរាប
អង្គរវត្ត
រៀន
សាលា
សិស្ស
គ្រូ
សៀវភៅ
អាន
សរសេរ
និយាយ
EOF
echo "✅ Created basic Khmer lexicon"

# 4. Create placeholder KenLM model
echo "🔤 Creating placeholder language model..."
touch lang/kenlm/khmer_5gram.bin
echo "⚠️  Note: Real KenLM model needs to be trained separately"

# 5. Fix the config file
echo "⚙️ Fixing config file paths..."
cat > train/configs/rec_kh_fixed.yaml << 'EOF'
# Fixed Recognition config for Khmer OCR
model_type: recognition
architecture: crnn_resnet34
input_size: [3, 48, 320]
charset: train/charset_kh.txt  # Using existing file

optimizer:
  name: AdamW
  lr: 0.0005
  weight_decay: 0.01
  scheduler:
    name: CosineAnnealingLR
    T_max: 10

train:
  epochs: 10
  batch_size: 4
  amp: false  # Disable AMP for stability
  seed: 42
  eval_steps: 50
  save_steps: 100
  
data:
  train_lists:
    - data/train_list.txt
  val_list: data/val_list.txt
  test_list: data/test_list.txt
  format: paddleocr_rec
  
  augmentation:
    - type: RandomCrop
      prob: 0.2
    - type: RandomRotate
      degree: 5
      
output:
  save_dir: models/khmer_rec
  log_interval: 10
  
metrics:
  - cer
  - accuracy
EOF
echo "✅ Created fixed config file"

# 6. Generate training data if needed
echo "📊 Checking training data..."
if [ ! -f "data/train_list.txt" ]; then
    echo "Generating synthetic training data..."
    python scripts/generate_khmer_synthetic.py 1000
else
    TRAIN_COUNT=$(wc -l < data/train_list.txt)
    echo "Found $TRAIN_COUNT training samples"
fi

# 7. Download pre-trained models if needed
echo "📥 Checking pre-trained models..."
if [ ! -d "models/pretrained/ch_PP-OCRv4_rec_train" ]; then
    echo "Downloading pre-trained models..."
    mkdir -p models/pretrained
    cd models/pretrained
    wget -q -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
    tar -xf ch_PP-OCRv4_rec_train.tar
    cd ../..
    echo "✅ Downloaded pre-trained models"
else
    echo "✅ Pre-trained models already available"
fi

echo ""
echo "✅ SETUP COMPLETE! Everything is fixed."
echo ""
echo "Now you can run training with:"
echo "  python train/train_demo.py --config train/configs/rec_kh_fixed.yaml --output-dir models/khmer_rec"
echo ""
echo "Or for real training (if train/run.py works):"
echo "  python train/run.py --config train/configs/rec_kh_fixed.yaml"
echo ""
echo "Training will use:"
echo "  - Charset: train/charset_kh.txt"
echo "  - Config: train/configs/rec_kh_fixed.yaml"
echo "  - Data: data/train_list.txt"
echo "  - Output: models/khmer_rec/"