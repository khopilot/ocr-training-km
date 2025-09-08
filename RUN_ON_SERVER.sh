#!/bin/bash
# Quick commands to run on SaladCloud server
# This bypasses all cuDNN issues by using Docker

echo "🚀 KHMER OCR PROFESSIONAL TRAINING"
echo "=================================="
echo ""

# Pull latest code
cd /ocr-training-km
git pull

# Option 1: Quick test with CPU (works immediately)
echo "Option 1: Quick CPU test (works now):"
echo "  python scripts/production_ocr_cpu.py"
echo ""

# Option 2: Full professional training in Docker
echo "Option 2: Professional GPU training (Docker):"
echo "Run these commands in sequence:"
echo ""
echo "# 1. Download HuggingFace data"
echo "make download-hf"
echo ""
echo "# 2. Validate dataset"
echo "python ops/validate_dataset.py --data-dir data/paddle_format --charset train/charset_kh.txt --strict"
echo ""
echo "# 3. Build corpus for language model"
echo "make build-corpus"
echo ""
echo "# 4. Train language model in Docker"
echo "make docker-train-lm"
echo ""
echo "# 5. Train recognition model in Docker (main training)"
echo "make docker-train-rec"
echo ""
echo "# 6. (Optional) Train detection model"
echo "make docker-train-det"
echo ""
echo "# 7. Evaluate models"
echo "make docker-eval"
echo ""
echo "# 8. Run benchmarks"
echo "make bench"
echo ""
echo "# 9. Check production gates"
echo "make production-gates"
echo ""
echo "# 10. Generate manifest"
echo "make manifest"
echo ""
echo "# 11. Optimize rescoring parameters"
echo "make rescoring-sweep"
echo ""
echo "# 12. Export to ONNX"
echo "make export-onnx"
echo ""

# Option 3: One-command script
echo "Option 3: Run everything with one script:"
echo "  bash scripts/train_pro_docker.sh"
echo ""

echo "=================================="
echo "MONITORING:"
echo "  # Watch GPU usage"
echo "  watch nvidia-smi"
echo ""
echo "  # Check Docker containers"
echo "  docker ps"
echo ""
echo "  # View training logs"
echo "  docker logs -f <container_id>"
echo ""

echo "=================================="
echo "AFTER TRAINING:"
echo "  # Serve the API with trained models"
echo "  SERVICE_VARIANT=paddle PRODUCTION_MODE=prod make serve-prod"
echo ""
echo "  # Test the API"
echo "  curl -F 'file=@test.png' http://localhost:8080/ocr"
echo ""