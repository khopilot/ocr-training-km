#!/bin/bash
# Working commands for SaladCloud server - all fixes applied

echo "🚀 KHMER OCR TRAINING - SERVER READY COMMANDS"
echo "=============================================="
echo ""
echo "Run these commands on your SaladCloud server:"
echo ""

echo "# 1. Setup (one time)"
echo "cd /ocr-training-km"
echo "git pull"
echo "ln -s /opt/venv .venv  # Create symlink for compatibility"
echo ""

echo "# 2. Download HuggingFace data (fixed downloader)"
echo "python ops/download_hf_fixed.py --output-dir data/hf_datasets --priority high"
echo ""

echo "# 3. Convert to PaddleOCR format (with fail-fast)"
echo "python ops/convert_to_paddle.py --input-dir data/hf_datasets --output-dir data/paddle_format --dataset all"
echo ""

echo "# 4. Validate dataset"
echo "python ops/validate_dataset.py --data-dir data/paddle_format --charset train/charset_kh.txt --strict"
echo ""

echo "# 5. Train recognition model (no Docker needed)"
echo "python train/run.py --config train/configs/rec_kh_hf.yaml"
echo "# Or use Makefile target:"
echo "# make train-rec"
echo ""

echo "# 6. Evaluate model"
echo "python -m eval.harness --test data/paddle_format/recognition/test --report eval/report.json"
echo "# Or use Makefile target:"
echo "# make eval"
echo ""

echo "# 7. Optional: Benchmark"
echo "python eval/benchmark.py --output eval/benchmark.json"
echo ""

echo "# 8. Optional: Export to ONNX"
echo "python ops/export_onnx.py --model-dir models --output-dir models/onnx"
echo ""

echo "=============================================="
echo "ALTERNATIVE: Run everything with one script"
echo "=============================================="
echo ""
echo "bash scripts/train_bare_metal.sh"
echo ""

echo "=============================================="
echo "IF STILL HAVING ISSUES:"
echo "=============================================="
echo ""
echo "# Force CPU mode if GPU/cuDNN fails:"
echo "sed -i 's/use_gpu: true/use_gpu: false/' train/configs/rec_kh_hf.yaml"
echo "python train/run.py --config train/configs/rec_kh_hf.yaml"
echo ""
echo "# Reduce batch size if OOM:"
echo "sed -i 's/batch_size: [0-9]*/batch_size: 8/' train/configs/rec_kh_hf.yaml"
echo ""

echo "=============================================="
echo "EXPECTED OUTPUT:"
echo "=============================================="
echo "- Models: models/paddle/"
echo "- Evaluation: eval/report.json"
echo "- Benchmark: eval/benchmark.json"
echo "- ONNX: models/onnx/"
echo ""