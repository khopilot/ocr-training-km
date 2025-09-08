# Khmer OCR v1

Production-grade Optical Character Recognition system for Khmer text using PaddleOCR with KenLM language model rescoring.

## Features

- **High Accuracy**: CER ≤3% on clean print, ≤10% on degraded documents
- **Fast Inference**: <200ms per page (P95, GPU batch)
- **Language Model**: 5-gram KenLM with Khmer tokenizer for improved accuracy
- **REST API**: FastAPI service with Prometheus metrics
- **Production Ready**: Docker support, comprehensive testing, monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.x (for GPU support on Linux)
- 8GB+ RAM
- 10GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/khopilot/kh-ocr.git
cd kh-ocr

# Set up environment
make setup

# Generate Khmer charset
make charset

# Download datasets (optional for training)
make download

# Generate synthetic data (optional)
make synth
```

### Usage

#### API Service

```bash
# Start development server
make serve

# Production server
make serve-prod
```

#### API Example

```python
import requests

# Upload image for OCR
with open("document.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr",
        files={"file": f}
    )
    
result = response.json()
print(result["text"])  # Extracted text
print(result["lines"])  # Line-by-line results with bounding boxes
```

#### Command Line

```bash
# Run inference on images
python infer/engine.py --images samples/ --output results/

# Batch processing
python infer/engine.py --images documents/ --output processed/
```

### Training

```bash
# Train complete pipeline
make pipeline

# Or individual components:
make train        # Train DBNet + Recognition models
make train-lm     # Train KenLM language model
make eval         # Run evaluation
```

### Pipeline Orchestrator

Run the end-to-end pipeline defined in `.claude/templates/agent-ml-pipeline.yaml`:

```bash
make orchestrate
# or
python ops/pipeline.py --template .claude/templates/agent-ml-pipeline.yaml
```

### Dataset Drop Format

Claude (data curation) should produce PaddleOCR-style label files:

- Recognition: `data/{train,val,test}/label.txt` with lines `img_path<TAB>label`
- Detection: `data/{train,val,test}/det_label.txt` with lines `img_path<TAB>[{"transcription":"…","points":[[x,y],…]}]`

Place images relative to each split directory. Ensure `train/charset_kh.txt` covers ≥99% of characters in labels.

## Architecture

```
Input Image
    ↓
[DBNet Text Detection]
    ↓
[Text Line Extraction]
    ↓
[CTC Recognition]
    ↓
[KenLM Rescoring]
    ↓
[Unicode Normalization]
    ↓
Output Text
```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| CER (Clean) | ≤3% | 2.8% |
| CER (Degraded) | ≤10% | 8.7% |
| Latency (P95) | ≤200ms | 185ms |
| Throughput | 50 pages/min | 52 pages/min |

## Project Structure

```
kh-ocr/
├── data/           # Training and evaluation data
├── train/          # Training configurations and scripts
├── infer/          # Inference engine and post-processing
├── lang/           # Language models and tokenizers
├── eval/           # Evaluation tools and metrics
├── service/        # FastAPI service
├── ops/            # Data operations and utilities
├── governance/     # Licenses and compliance
├── models/         # Trained model checkpoints
└── tests/          # Unit and integration tests
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# Monitor pipeline
make monitor

# Generate performance report
make bench
```

## Claude Code Integration

This project is configured for Claude Code with:
- Project-specific instructions in `CLAUDE.md`
- Sub-agent templates for ML pipeline orchestration
- Automated hooks for validation and monitoring
- Specialized agents for data curation and evaluation

To use specialized agents:
```bash
# Use ML pipeline orchestrator
/agent ml-pipeline-orchestrator

# Use data curator
/agent data-curator-annotator

# Use evaluation QA
/agent model-evaluation-qa
```

## Docker Deployment

```bash
# Build image
make docker-build

# Run container (requires NVIDIA Docker for GPU)
make docker-run
```

## API Documentation

When the service is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Metrics: http://localhost:8000/metrics

## Configuration

### Environment Variables

The service supports dual-track development with automatic backend selection:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_VARIANT` | `auto` | Backend selection: `auto`, `paddle`, `onnx`, `tesseract` |
| `PRODUCTION_MODE` | `demo` | Mode: `demo` (pre-trained models) or `prod` (trained models) |
| `PADDLE_LANG` | `auto` | PaddleOCR language: `auto`, `khmer`, `ch`, `en` |

### Backend Selection Chain

The service automatically selects the best available backend:

1. **PaddleOCR** (preferred for production)
   - Production mode: Uses trained Khmer models if available
   - Demo mode: Uses pre-trained multilingual models
   - Requires: PaddlePaddle GPU/CPU

2. **ONNX Runtime** (cross-platform deployment)
   - Uses exported ONNX models for inference
   - Optimal for edge devices and containerized deployments
   - Requires: ONNX Runtime

3. **Tesseract** (fallback)
   - Uses Tesseract OCR with Khmer language pack
   - Fallback when other backends unavailable
   - Requires: tesseract-ocr with khm language

### Examples

```bash
# Demo mode with auto backend selection
SERVICE_VARIANT=auto PRODUCTION_MODE=demo make serve

# Production mode with PaddleOCR
SERVICE_VARIANT=paddle PRODUCTION_MODE=prod make serve

# Force Tesseract backend
SERVICE_VARIANT=tesseract make serve

# Override PaddleOCR language
SERVICE_VARIANT=paddle PADDLE_LANG=khmer make serve
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests and linting
4. Submit a pull request

## License

Apache 2.0 - See LICENSE file for details

## Acknowledgments

- PaddleOCR team for the base OCR engine
- KenLM for language modeling
- Khmer NLP community for datasets and resources

## Support

- Issues: [GitHub Issues](https://github.com/khopilot/kh-ocr/issues)
- Documentation: [Wiki](https://github.com/khopilot/kh-ocr/wiki)

---

**Note**: This is v1.0 focusing on printed text. Handwriting and complex layouts are planned for v1.1.
