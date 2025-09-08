# Makefile

```make
SHELL := /bin/bash
PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
PORT ?= 8000

.DEFAULT_GOAL := help

## help: Show targets
help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS := ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sed 's/"$$//'

venv:  ## Create venv
	$(PYTHON) -m venv $(VENV); \
	. $(VENV)/bin/activate; $(PIP) install -U pip wheel setuptools

setup: venv  ## Install deps for macOS M3
	$(PIP) install -r requirements-mac.txt
	$(PY) -m pip install pre-commit && $(VENV)/bin/pre-commit install

download:  ## Download sample data
	$(PY) ops/download.py --out data/raw || true

train:  ## Placeholder training (CPU). Heavy on Mac; use cloud for real training
	$(PY) train/run.py --dbnet train/configs/dbnet.yaml --rec train/configs/rec_kh.yaml || true

infer:  ## Run local inference on samples
	$(PY) infer/engine.py --images eval/samples --out out

serve:  ## Start API (FastAPI + Uvicorn)
	$(UVICORN) service.app:app --host 0.0.0.0 --port $(PORT)

eval:  ## Evaluate CER/WER on test set
	$(PY) eval/harness.py --test data/test --report eval/report.json

bench: ## Latency benchmark
	$(PY) eval/bench.py --corpus eval/samples --batch 4

format: ## Format and lint
	$(VENV)/bin/ruff check --fix || true
	$(VENV)/bin/black .

clean: ## Remove caches and outputs
	rm -rf out .pytest_cache __pycache__

docker-linux: ## Build Linux/amd64 container from Mac (buildx)
	docker buildx build --platform linux/amd64 -t kh-ocr:v1 .
```

---

# requirements-mac.txt

```txt
# Core
fastapi==0.112.0
uvicorn[standard]==0.30.5
numpy==1.26.4
pandas==2.2.2
pillow==10.4.0
opencv-python-headless==4.10.0.84
sentencepiece==0.1.99
prometheus-client==0.20.0
pydantic==2.8.2
python-multipart==0.0.9

# Khmer LM rescoring
git+https://github.com/kpu/kenlm.git#subdirectory=python

# ONNX Runtime for Apple Silicon
onnxruntime-silicon>=1.18.0

# Dev
black==24.8.0
ruff==0.6.8
pre-commit==3.7.1

# Optional OCR stacks (comment out if failing on macOS arm64)
# paddlepaddle==2.6.1
# paddleocr==2.7.0
# tesseract is installed via Homebrew: brew install tesseract
```

---

## Quick start (Mac M3)

```bash
# System deps
brew install tesseract cmake pkg-config protobuf

# Setup
make setup

# Run API
make serve PORT=8000

# Inference example
make infer
```

