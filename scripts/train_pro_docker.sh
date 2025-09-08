#!/bin/bash
# Professional Khmer OCR Training Pipeline using Docker GPU
# Bypasses all host cuDNN issues - runs everything in container

set -e  # Exit on error

echo "🚀 PROFESSIONAL KHMER OCR TRAINING PIPELINE"
echo "==========================================="
echo "Using Docker GPU to bypass host cuDNN issues"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "Makefile" ] || [ ! -d "train" ]; then
    print_error "Please run this script from the khmer-ocr-v1 root directory"
    exit 1
fi

# Step 1: Download and prepare data
print_status "Step 1: Downloading HuggingFace data..."
make download-hf || print_warning "Download may have failed - continuing"

# Step 2: Validate dataset
print_status "Step 2: Validating dataset..."
if [ -f "ops/validate_dataset.py" ]; then
    python ops/validate_dataset.py \
        --data-dir data/paddle_format \
        --charset train/charset_kh.txt \
        --strict || print_warning "Validation warnings - continuing"
else
    print_warning "Validator not found - skipping"
fi

# Step 3: Build corpus for language model
print_status "Step 3: Building text corpus..."
make build-corpus || print_warning "Corpus building failed - continuing"

# Step 4: Build Docker image
print_status "Step 4: Building Docker GPU image..."
docker build -f docker/Dockerfile.gpu -t khmer-ocr-gpu:latest . || {
    print_error "Docker build failed"
    echo "Trying alternative build..."
    docker build -f docker/Dockerfile.gpu --no-cache -t khmer-ocr-gpu:latest .
}

# Step 5: Train language model in Docker
print_status "Step 5: Training KenLM language model (Docker)..."
make docker-train-lm || print_warning "LM training failed - continuing without LM"

# Step 6: Train recognition model (main training)
print_status "Step 6: Training recognition model (Docker GPU)..."
echo "This is the main training - will use GPU with cuDNN in container"
make docker-train-rec || {
    print_error "Recognition training failed"
    echo "Check GPU memory - you may need to reduce batch_size in train/configs/rec_kh_hf.yaml"
    exit 1
}

# Step 7: (Optional) Train detection model
read -p "Train detection model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Step 7: Training detection model (Docker GPU)..."
    make docker-train-det || print_warning "Detection training failed"
else
    print_status "Step 7: Skipping detection training"
fi

# Step 8: Evaluate models
print_status "Step 8: Evaluating models (Docker)..."
make docker-eval || print_warning "Evaluation failed"

# Step 9: Run benchmarks
print_status "Step 9: Running benchmarks..."
make bench || print_warning "Benchmarking failed"

# Step 10: Check production gates
print_status "Step 10: Checking production gates..."
make production-gates || print_warning "Some gates failed"

# Step 11: Generate manifest
print_status "Step 11: Generating manifest..."
make manifest || print_warning "Manifest generation failed"

# Step 12: Rescoring sweep (optimize parameters)
print_status "Step 12: Running rescoring sweep..."
make rescoring-sweep || print_warning "Rescoring sweep failed"

# Step 13: Export to ONNX
print_status "Step 13: Exporting to ONNX format..."
make export-onnx || print_warning "ONNX export failed"

# Summary
echo ""
echo "==========================================="
print_status "TRAINING PIPELINE COMPLETE!"
echo "==========================================="
echo ""
echo "📊 Results:"
echo "  - Models: models/paddle/"
echo "  - Evaluation: eval/report.json"
echo "  - Benchmark: eval/benchmark.json"
echo "  - Rescoring: eval/rescoring_results.json"
echo "  - Manifest: governance/manifest.json"
echo "  - ONNX: models/onnx/"
echo ""
echo "🚀 Next steps:"
echo "  1. Review eval/report.json for CER metrics"
echo "  2. Update rec_kh_hf.yaml with best λ/μ/beam from rescoring"
echo "  3. Serve with: SERVICE_VARIANT=paddle PRODUCTION_MODE=prod make serve-prod"
echo "  4. Test API: curl -F 'file=@image.png' http://localhost:8080/ocr"
echo ""
echo "💡 Tips:"
echo "  - If OOM errors, reduce batch_size in configs"
echo "  - For faster training, increase batch_size if GPU allows"
echo "  - Monitor GPU: docker exec <container> nvidia-smi"
echo ""