#!/bin/bash
# Bare-metal Khmer OCR training (no Docker required)
# Works on servers without Docker, uses whatever Python is available

set -e  # Exit on error

echo "🚀 KHMER OCR BARE-METAL TRAINING"
echo "================================="
echo "Running directly on host (no Docker)"
echo ""

# Detect Python - use whatever is available
PY=$(command -v python3 || command -v python || echo "python")
echo "Using Python: $PY"
$PY --version

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Download HuggingFace data
print_status "Step 1: Downloading HuggingFace data..."
if ! make download-hf 2>/dev/null; then
    print_warning "Make target failed, trying direct Python..."
    
    $PY ops/download_hf.py --output-dir data/hf_datasets --priority high || {
        print_error "Download failed - SynthKhmer-10k is required"
        print_error "Check that ground_truth field is being extracted correctly"
        exit 1
    }
fi

# Step 2: Convert to PaddleOCR format
print_status "Step 2: Converting to PaddleOCR format..."
if [ ! -d "data/paddle_format" ] || [ -z "$(ls -A data/paddle_format 2>/dev/null)" ]; then
    $PY ops/convert_to_paddle.py \
        --input-dir data/hf_datasets \
        --output-dir data/paddle_format \
        --dataset all || {
        print_error "Conversion failed - no data to convert"
        exit 1
    }
else
    print_status "PaddleOCR format data already exists"
fi

# Step 3: Validate dataset
print_status "Step 3: Validating dataset..."

# Check that label files exist and are non-empty
for split in train val test; do
    label_file="data/paddle_format/recognition/$split/label.txt"
    if [ ! -f "$label_file" ]; then
        print_error "Missing $label_file - conversion failed"
        exit 1
    fi
    
    line_count=$(wc -l < "$label_file" 2>/dev/null || echo "0")
    if [ "$line_count" -eq "0" ]; then
        print_error "$label_file is empty - no samples converted"
        exit 1
    fi
    print_status "  $split: $line_count samples"
done

$PY ops/validate_dataset.py \
    --data-dir data/paddle_format \
    --charset train/charset_kh.txt \
    --strict || print_warning "Validation warnings - continuing"

# Step 4: Build corpus (optional for LM)
print_status "Step 4: Building text corpus for language model..."
if [ ! -f "data/corpus/khmer_training_corpus.txt" ]; then
    $PY ops/build_corpus.py \
        --input-dirs data/paddle_format/recognition \
        --output data/corpus/khmer_training_corpus.txt || {
        print_warning "Corpus building failed - continuing without LM"
    }
else
    print_status "Corpus already exists"
fi

# Step 5: Train language model (optional)
print_status "Step 5: Training KenLM language model..."
if command -v lmplz &> /dev/null; then
    if [ ! -f "lang/kenlm/khmer_5gram.bin" ]; then
        $PY lang/train_lm.py \
            --use-hf-tokenizer \
            --hf-tokenizer khopilot/km-tokenizer-khmer \
            --corpus data/corpus/khmer_training_corpus.txt \
            --output lang/kenlm/khmer_5gram || {
            print_warning "LM training failed - continuing without LM"
        }
    else
        print_status "Language model already exists"
    fi
else
    print_warning "KenLM not installed - skipping LM training"
fi

# Step 6: Check if Docker is available
if command -v docker &> /dev/null; then
    print_status "Docker is available - using containerized training"
    
    # Build Docker image
    docker build -f docker/Dockerfile.gpu -t khmer-ocr-gpu:latest . || {
        print_warning "Docker build failed - falling back to bare metal"
        DOCKER_AVAILABLE=false
    }
    DOCKER_AVAILABLE=true
else
    print_warning "Docker not available - using bare-metal training"
    DOCKER_AVAILABLE=false
fi

# Step 7: Train recognition model
print_status "Step 6: Training recognition model..."
if [ "$DOCKER_AVAILABLE" = true ]; then
    make docker-train-rec || {
        print_error "Docker training failed"
        echo "Falling back to bare-metal..."
        DOCKER_AVAILABLE=false
    }
fi

if [ "$DOCKER_AVAILABLE" = false ]; then
    # Check GPU availability
    if $PY -c "import paddle; print(paddle.is_compiled_with_cuda())" 2>/dev/null | grep -q "True"; then
        echo "GPU detected - using GPU training"
        CONFIG="train/configs/rec_kh_hf_pro.yaml"
    else
        echo "No GPU or cuDNN issues - using CPU training"
        CONFIG="train/configs/rec_kh_hf.yaml"
        
        # Modify config for CPU (smaller batch size)
        if [ -f "$CONFIG" ]; then
            cp $CONFIG ${CONFIG}.bak
            sed -i 's/batch_size: [0-9]*/batch_size: 4/' $CONFIG
        fi
    fi
    
    # Run training directly
    $PY train/run.py --config $CONFIG || {
        print_error "Training failed"
        echo ""
        echo "Troubleshooting:"
        echo "1. Check GPU memory: nvidia-smi"
        echo "2. Reduce batch_size in $CONFIG"
        echo "3. Set use_gpu: false for CPU training"
        echo "4. Check data exists: ls data/paddle_format/"
        exit 1
    }
fi

# Step 8: Evaluate model
print_status "Step 7: Evaluating model..."
if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
    $PY eval/harness.py \
        --test data/paddle_format/recognition/test \
        --report eval/report.json || print_warning "Evaluation failed"
    
    # Benchmark
    $PY eval/benchmark.py \
        --output eval/benchmark.json || print_warning "Benchmark failed"
else
    print_warning "No models found to evaluate"
fi

# Step 9: Rescoring sweep (optional)
print_status "Step 8: Optimizing rescoring parameters..."
if [ -f "eval/harness.py" ]; then
    $PY eval/rescoring_sweep.py \
        --test data/paddle_format/recognition/test \
        --model-dir models \
        --output eval/rescoring_results.json || print_warning "Rescoring failed"
fi

# Step 10: Export to ONNX
print_status "Step 9: Exporting to ONNX..."
if [ -d "models" ]; then
    $PY ops/export_onnx.py \
        --model-dir models \
        --output-dir models/onnx || print_warning "ONNX export failed"
fi

# Summary
echo ""
echo "================================="
print_status "TRAINING COMPLETE!"
echo "================================="
echo ""
echo "📊 Results:"
if [ -f "eval/report.json" ]; then
    echo "  ✅ Evaluation: eval/report.json"
fi
if [ -f "eval/benchmark.json" ]; then
    echo "  ✅ Benchmark: eval/benchmark.json"
fi
if [ -d "models" ]; then
    echo "  ✅ Models: models/"
fi
if [ -d "models/onnx" ]; then
    echo "  ✅ ONNX: models/onnx/"
fi

echo ""
echo "🚀 Next steps:"
echo "  1. Review evaluation metrics"
echo "  2. Serve the model:"
echo "     SERVICE_VARIANT=paddle PRODUCTION_MODE=prod $PY -m uvicorn service.app:app --host 0.0.0.0 --port 8080"
echo "  3. Test API:"
echo "     curl -F 'file=@test.png' http://localhost:8080/ocr"
echo ""
echo "💡 If training failed:"
echo "  - Check Python packages: $PY -m pip list | grep paddle"
echo "  - For CPU training: set use_gpu: false in config"
echo "  - Reduce batch_size if OOM"
echo ""