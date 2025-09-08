# Khmer OCR v1 Makefile
# Platform detection and configuration

SHELL := /bin/bash

# Dynamic Python detection - no hardcoded .venv
PYTHON_BIN ?= $(shell command -v python3 || command -v python)
PIP ?= $(shell command -v pip3 || command -v pip)

# Optional venv support (if .venv exists, use it)
ifneq ($(wildcard .venv/bin/python),)
    PYTHON_BIN := .venv/bin/python
    PIP := .venv/bin/pip
endif

# Legacy variables for compatibility
PYTHON := $(PYTHON_BIN)
VENV := .venv

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    EXTRAS := macos,dev,eval
else
    PLATFORM := linux
    EXTRAS := gpu,dev,train,eval
endif

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

.PHONY: help setup download train infer serve eval bench test clean monitor vendor-tokenizer docker-build docker-run docker-train-det docker-train-rec docker-train-lm build-corpus export-onnx rescoring-sweep

help: ## Show this help message
	@echo "Khmer OCR v1 - Build System"
	@echo "Platform: $(PLATFORM)"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

setup: ## Set up development environment (full - Linux production)
	@echo "$(YELLOW)Setting up Khmer OCR environment ($(PLATFORM))...$(NC)"
	@echo "Using Python: $(PYTHON_BIN)"
	@if [ ! -d "$(VENV)" ] && [ "$(PYTHON_BIN)" != ".venv/bin/python" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(NC)"; \
		$(PYTHON_BIN) -m venv $(VENV); \
		$(VENV)/bin/pip install --upgrade pip setuptools wheel; \
		$(VENV)/bin/pip install -e ".[$(EXTRAS)]"; \
	else \
		echo "$(YELLOW)Using existing Python environment...$(NC)"; \
		$(PIP) install --upgrade pip setuptools wheel; \
		$(PIP) install -e ".[$(EXTRAS)]"; \
	fi
	@if [ "$(PLATFORM)" = "linux" ]; then \
		echo "$(YELLOW)Installing KenLM for Linux...$(NC)"; \
		$(PIP) install https://github.com/kpu/kenlm/archive/master.zip; \
	else \
		echo "$(YELLOW)Note: KenLM installation may require additional steps on macOS$(NC)"; \
		echo "Run: brew install cmake boost eigen && pip install https://github.com/kpu/kenlm/archive/master.zip"; \
	fi
	@echo "$(GREEN)✓ Setup complete!$(NC)"

setup-macos-demo: ## Set up macOS demo environment (ONNX + Tesseract only)
	@echo "$(YELLOW)Setting up macOS demo environment...$(NC)"
	@if ! command -v $(PYTHON) &> /dev/null; then \
		echo "$(RED)Error: Python 3.11 not found. Please install it first.$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON) -m venv .venv-demo
	@.venv-demo/bin/pip install --upgrade pip
	@.venv-demo/bin/pip install -r requirements-demo.txt
	@echo "$(YELLOW)Installing Tesseract with Khmer support...$(NC)"
	@if ! command -v tesseract &> /dev/null; then \
		echo "$(YELLOW)Installing Tesseract via Homebrew...$(NC)"; \
		brew install tesseract tesseract-lang; \
	else \
		echo "$(GREEN)✓ Tesseract already installed$(NC)"; \
	fi
	@echo "$(GREEN)✓ macOS demo environment ready!$(NC)"
	@echo "$(YELLOW)Note: This is DEMO MODE with limited Khmer recognition$(NC)"

download: ## Download and prepare datasets
	@echo "$(YELLOW)Downloading datasets...$(NC)"
	@$(PYTHON_BIN) ops/download.py --out data/raw
	@echo "$(GREEN)✓ Download complete!$(NC)"

download-hf: ## Download HuggingFace datasets from seanghay
	@echo "$(YELLOW)Downloading HuggingFace datasets from seanghay...$(NC)"
	@$(PYTHON_BIN) ops/download_hf.py --output-dir data/hf_datasets --priority high || exit 1
	@echo "$(GREEN)✓ HF datasets downloaded!$(NC)"
	@echo "$(YELLOW)Converting to PaddleOCR format...$(NC)"
	@$(PYTHON_BIN) ops/convert_to_paddle.py --input-dir data/hf_datasets --output-dir data/paddle_format || exit 1
	@echo "$(GREEN)✓ Conversion complete!$(NC)"
	@echo "$(YELLOW)Building unified lexicon...$(NC)"
	@$(PYTHON_BIN) ops/build_lexicon.py --input-dir data/hf_datasets --output-dir lang/lexicon
	@echo "$(GREEN)✓ Lexicon built!$(NC)"

charset: ## Generate Khmer character set
	@echo "$(YELLOW)Generating Khmer charset...$(NC)"
	@$(PYTHON_BIN) ops/generate_charset.py --output train/charset_kh.txt
	@echo "$(GREEN)✓ Charset generated!$(NC)"

synth: ## Generate synthetic training data
	@echo "$(YELLOW)Generating synthetic data...$(NC)"
	@$(PYTHON_BIN) ops/synth.py --samples 10000 --output data/synth
	@echo "$(GREEN)✓ Synthetic data generated!$(NC)"

train: ## Train OCR models (DBNet + Recognition)
	@echo "$(YELLOW)Starting training pipeline...$(NC)"
	@if [ ! -f train/charset_kh.txt ]; then \
		echo "$(RED)Error: charset_kh.txt not found. Run 'make charset' first.$(NC)"; \
		exit 1; \
	fi
	@$(PYTHON_BIN) train/run.py --dbnet train/configs/dbnet.yaml --rec train/configs/rec_kh.yaml
	@echo "$(GREEN)✓ Training complete!$(NC)"

build-corpus: ## Extract corpus from converted HF datasets
	@echo "$(YELLOW)Building training corpus...$(NC)"
	@mkdir -p data/corpus
	@$(PYTHON_BIN) ops/build_corpus.py \
		--input-dirs data/paddle_format \
		--output data/corpus/khmer_training_corpus.txt \
		--min-length 2 --max-length 200
	@echo "$(GREEN)✓ Corpus built: data/corpus/khmer_training_corpus.txt$(NC)"

train-lm: build-corpus ## Train KenLM language model with HF tokenizer
	@echo "$(YELLOW)Training language model with HF tokenizer...$(NC)"
	@mkdir -p lang/kenlm
	@$(PYTHON_BIN) lang/train_lm.py \
		--corpus data/corpus/khmer_training_corpus.txt \
		--order 5 \
		--output lang/kenlm/khmer_5gram \
		--use-hf-tokenizer \
		--hf-tokenizer khopilot/km-tokenizer-khmer \
		--update-manifest
	@echo "$(GREEN)✓ Language model trained with khopilot tokenizer!$(NC)"

export-onnx: ## Export trained models to ONNX format
	@echo "$(YELLOW)Exporting models to ONNX...$(NC)"
	@mkdir -p models/onnx
	@$(PYTHON_BIN) ops/export_onnx.py \
		--model-dir models \
		--output-dir models/onnx \
		--models recognizer detector
	@echo "$(GREEN)✓ Models exported to ONNX format!$(NC)"

rescoring-sweep: ## Run rescoring parameter sweep
	@echo "$(YELLOW)Running rescoring parameter sweep...$(NC)"
	@mkdir -p eval
	@$(PYTHON_BIN) eval/rescoring_sweep.py \
		--test data/test \
		--model-dir models \
		--output eval/rescoring_results.json
	@echo "$(GREEN)✓ Rescoring sweep complete!$(NC)"

infer: ## Run inference on sample images (auto-detect backend)
	@echo "$(YELLOW)Running inference...$(NC)"
	@$(PYTHON_BIN) infer/engine.py --images samples/ --output out/
	@echo "$(GREEN)✓ Inference complete!$(NC)"

infer-demo: ## Run demo inference (macOS with Tesseract + ONNX)
	@echo "$(YELLOW)Running DEMO inference (macOS)...$(NC)"
	@.venv-demo/bin/python infer/engine.py --images data/synth/ --output out/ --backend tesseract
	@echo "$(GREEN)✓ Demo inference complete!$(NC)"
	@echo "$(YELLOW)Note: Demo mode with limited Khmer recognition quality$(NC)"

serve: ## Start FastAPI OCR service
	@echo "$(YELLOW)Starting OCR service...$(NC)"
	@$(PYTHON_BIN) -m uvicorn service.app:app --host 0.0.0.0 --port 8080 --reload

serve-prod: ## Start production OCR service
	@echo "$(YELLOW)Starting production OCR service...$(NC)"
	@$(PYTHON_BIN) -m uvicorn service.app:app --host 0.0.0.0 --port 8080 --workers 4

train-rec: ## Train recognition model (bare-metal, no Docker)
	@echo "$(YELLOW)Training recognition model...$(NC)"
	@$(PYTHON_BIN) train/run.py --config train/configs/rec_kh_hf.yaml
	@echo "$(GREEN)✓ Recognition training complete!$(NC)"

eval: ## Run evaluation harness (auto-detect backend)
	@echo "$(YELLOW)Running evaluation...$(NC)"
	@$(PYTHON_BIN) -m eval.harness --test data/paddle_format/recognition/test --report eval/report.json
	@echo "$(GREEN)✓ Evaluation complete! Report: eval/report.json$(NC)"

eval-demo: ## Run demo evaluation (macOS with clear labels)
	@echo "$(YELLOW)Running DEMO evaluation...$(NC)"
	@.venv-demo/bin/python eval/harness.py --test data/test --report eval/demo-report.json --label "DEMO/CPU/macOS"
	@echo "$(GREEN)✓ Demo evaluation complete! Report: eval/demo-report.json$(NC)"
	@echo "$(YELLOW)Note: Results labeled as DEVELOPMENT/DEMO quality$(NC)"

train-demo: ## Run lightweight demo training (macOS CPU)
	@echo "$(YELLOW)Running DEMO training...$(NC)"
	@.venv-demo/bin/python train/train_demo.py --config train/configs/rec_kh.yaml --output-dir models/demo
	@echo "$(GREEN)✓ Demo training complete!$(NC)"
	@echo "$(YELLOW)Note: This is DEMO training for development validation$(NC)"

train-demo-quick: ## Quick demo training (1 epoch, small batch)
	@echo "$(YELLOW)Running QUICK demo training...$(NC)"
	@mkdir -p models/demo
	@.venv-demo/bin/python -c "import yaml; config = yaml.safe_load(open('train/configs/rec_kh.yaml')); config['train']['epochs'] = 1; config['train']['batch_size'] = 16; yaml.dump(config, open('models/demo/quick_config.yaml', 'w'))"
	@.venv-demo/bin/python train/train_demo.py --config models/demo/quick_config.yaml --output-dir models/demo
	@echo "$(GREEN)✓ Quick demo training complete!$(NC)"

demo-infer: infer-demo ## Alias for infer-demo (macOS Tesseract + ONNX)

demo-eval: eval-demo ## Alias for eval-demo (macOS with clear demo labels)

docker-train: ## Run full training in Docker (Linux GPU)
	@echo "$(YELLOW)Running training in Docker GPU container...$(NC)"
	@docker build -f docker/Dockerfile.gpu -t khmer-ocr:gpu-train .
	@docker run --gpus all -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data khmer-ocr:gpu-train make train
	@echo "$(GREEN)✓ Docker training complete!$(NC)"

docker-train-det: ## Train DBNet detection in Docker (Linux GPU)
	@echo "$(YELLOW)Training DBNet detection in Docker...$(NC)"
	@docker build -f docker/Dockerfile.gpu -t khmer-ocr:gpu-train .
	@docker run --gpus all -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data -v $(PWD)/train:/app/train khmer-ocr:gpu-train \
		$(PYTHON_BIN) train/run.py --config train/configs/dbnet_hf.yaml --output models/dbnet
	@echo "$(GREEN)✓ DBNet training complete!$(NC)"

docker-train-rec: ## Train recognition model in Docker (Linux GPU)
	@echo "$(YELLOW)Training recognition model in Docker...$(NC)"
	@docker build -f docker/Dockerfile.gpu -t khmer-ocr:gpu-train .
	@docker run --gpus all -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data -v $(PWD)/train:/app/train khmer-ocr:gpu-train \
		$(PYTHON_BIN) train/run.py --config train/configs/rec_kh_hf.yaml --output models/rec_kh --charset train/charset_kh.txt
	@echo "$(GREEN)✓ Recognition training complete!$(NC)"

docker-train-lm: ## Train KenLM language model in Docker
	@echo "$(YELLOW)Training KenLM language model in Docker...$(NC)"
	@docker build -f docker/Dockerfile.gpu -t khmer-ocr:gpu-train .
	@docker run --gpus all -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data -v $(PWD)/lang:/app/lang khmer-ocr:gpu-train \
		$(PYTHON_BIN) lang/train_lm.py --corpus data/corpus/khmer_training_corpus.txt --order 5 --output lang/kenlm/khmer_5gram --use-hf-tokenizer --update-manifest
	@echo "$(GREEN)✓ KenLM training complete!$(NC)"

docker-eval: ## Run production evaluation in Docker
	@echo "$(YELLOW)Running production evaluation in Docker...$(NC)"
	@docker run --gpus all -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data -v $(PWD)/eval:/app/eval khmer-ocr:gpu-train make eval
	@echo "$(GREEN)✓ Docker evaluation complete!$(NC)"

bench: ## Run performance benchmark
	@echo "$(YELLOW)Running benchmarks...$(NC)"
	@$(PYTHON_BIN) eval/benchmark.py --output eval/benchmark.json
	@echo "$(GREEN)✓ Benchmark complete! Report: eval/benchmark.json$(NC)"

test: ## Run unit tests
	@echo "$(YELLOW)Running tests...$(NC)"
	@$(PYTHON_BIN) -m pytest tests/ -v --tb=short --cov=kh_ocr --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete!$(NC)"

lint: ## Run code linting
	@echo "$(YELLOW)Running linter...$(NC)"
	@$(PYTHON_BIN) -m ruff check .
	@$(PYTHON_BIN) -m ruff format --check .
	@echo "$(GREEN)✓ Linting complete!$(NC)"

format: ## Format code
	@echo "$(YELLOW)Formatting code...$(NC)"
	@$(PYTHON_BIN) -m ruff check --fix .
	@$(PYTHON_BIN) -m ruff format .
	@echo "$(GREEN)✓ Formatting complete!$(NC)"

monitor: ## Monitor pipeline execution
	@echo "$(YELLOW)Starting pipeline monitor...$(NC)"
	@$(PYTHON_BIN) .claude/scripts/monitor-pipeline.py report

monitor-train: ## Monitor training run
	@echo "$(YELLOW)Monitoring training...$(NC)"
	@$(PYTHON_BIN) .claude/scripts/monitor-pipeline.py train $(RUN_ID)

orchestrate: ## Run YAML pipeline orchestrator
	@echo "$(YELLOW)Running pipeline from template...$(NC)"
	@$(PYTHON_BIN) ops/pipeline.py --template .claude/templates/agent-ml-pipeline.yaml --continue-on-error

manifest: ## Generate manifest.json
	@echo "$(YELLOW)Generating manifest...$(NC)"
	@$(PYTHON_BIN) ops/manifests.py --update
	@echo "$(GREEN)✓ Manifest generated!$(NC)"

production-gates: ## Validate production deployment requirements
	@echo "$(YELLOW)Validating production gates...$(NC)"
	@$(PYTHON_BIN) ops/production_gates.py --project-root . || \
		(echo "$(RED)❌ Production gates failed!$(NC)" && exit 1)
	@echo "$(GREEN)✅ Production gates passed!$(NC)"



clean: ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning...$(NC)"
	@rm -rf $(VENV) build/ dist/ *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.log" -delete
	@rm -rf .pytest_cache/ .coverage htmlcov/
	@rm -rf out/ tmp/ temp/
	@echo "$(GREEN)✓ Clean complete!$(NC)"

clean-models: ## Clean model files (careful!)
	@echo "$(RED)Warning: This will delete all trained models!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/*.pdparams models/*.pdmodel models/*.onnx; \
		echo "$(GREEN)✓ Models cleaned!$(NC)"; \
	fi

# Development shortcuts
.PHONY: dev
dev: setup charset synth ## Quick development setup

.PHONY: pipeline
pipeline: charset build-corpus train-lm train eval rescoring-sweep ## Run complete training pipeline

.PHONY: all
all: setup download pipeline serve ## Complete setup and run

# Default target
.DEFAULT_GOAL := help

# Additional production targets
.PHONY: vendor-tokenizer
vendor-tokenizer: ## Download tokenizer for offline use
	@echo "$(YELLOW)Vendoring khopilot/km-tokenizer-khmer...$(NC)"
	@mkdir -p lang/tokenizer/khopilot
	@$(PIP) show transformers >/dev/null 2>&1 || $(PIP) install transformers>=4.30.0
	@$(PYTHON_BIN) -c "from transformers import AutoTokenizer; \
		tokenizer = AutoTokenizer.from_pretrained('khopilot/km-tokenizer-khmer'); \
		tokenizer.save_pretrained('lang/tokenizer/khopilot'); \
		print('✓ Tokenizer vendored to lang/tokenizer/khopilot')" || \
		echo "$(RED)Failed to vendor tokenizer$(NC)"
	@echo "$(GREEN)✓ Tokenizer available offline at lang/tokenizer/khopilot$(NC)"
	@echo "$(YELLOW)Use --hf-tokenizer lang/tokenizer/khopilot for offline training$(NC)"

.PHONY: docker-build
docker-build: ## Build Docker images
	@echo "$(YELLOW)Building Docker images...$(NC)"
	@docker build -f docker/Dockerfile.cpu -t khmer-ocr:cpu .
	@if [ "$(PLATFORM)" = "linux" ]; then \
		docker build -f docker/Dockerfile.gpu -t khmer-ocr:gpu .; \
	fi
	@echo "$(GREEN)Docker images built$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container
	@echo "$(YELLOW)Starting Docker container...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)Service running at http://localhost:8080$(NC)"

.PHONY: docker-stop
docker-stop: ## Stop Docker container
	@docker-compose down

.PHONY: helm-install
helm-install: ## Install Helm chart
	@echo "$(YELLOW)Installing Helm chart...$(NC)"
	@helm install khmer-ocr deploy/helm/khmer-ocr
	@echo "$(GREEN)Helm chart installed$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "$(YELLOW)Running unit tests...$(NC)"
	@$(PYTHON_BIN) -m pytest tests/unit -v --cov=. --cov-report=term

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(YELLOW)Running integration tests...$(NC)"
	@$(PYTHON_BIN) -m pytest tests/integration -v
