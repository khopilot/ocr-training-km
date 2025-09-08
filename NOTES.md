# Khmer OCR - Technical Notes

## Known Limitations

### Platform-Specific

#### macOS (Apple Silicon)
- **GPU Support**: PaddleOCR GPU inference not available on macOS
- **Performance**: CPU-only inference, expect 300-500ms latency
- **Workaround**: Use ONNX runtime for better CPU performance
- **Acceptance**: CER thresholds apply, latency thresholds relaxed

#### Linux (CUDA)
- **Requirements**: CUDA 12.x, minimum 4GB VRAM
- **Performance**: P95 latency ≤ 200ms for batch 8
- **Optimization**: Use FP16 quantization for memory efficiency

### Model Limitations
- **Languages**: Optimized for Khmer script only
- **Text Types**: Best performance on printed text
- **Image Quality**: Minimum 150 DPI recommended
- **Orientation**: Assumes horizontal text (0° rotation)

### API Limitations
- **File Size**: Maximum 50MB per image
- **Timeout**: 30 seconds per request
- **Batch Size**: Maximum 32 images per batch
- **Rate Limiting**: 100 requests/minute (configurable)

## ONNX Usage

### Enabling ONNX Runtime
```bash
# Set environment variable
export SERVICE_VARIANT=onnx

# Or in docker-compose
environment:
  - SERVICE_VARIANT=onnx
```

### Performance Comparison
| Backend | P50 (ms) | P95 (ms) | Memory (GB) |
|---------|----------|----------|-------------|
| PaddleOCR CPU | 150 | 400 | 1.5 |
| PaddleOCR GPU | 30 | 80 | 2.0 |
| ONNX CPU | 100 | 250 | 1.0 |
| ONNX GPU | 20 | 50 | 1.8 |

### Quantization Options
```python
# FP16 (recommended for GPU)
python ops/export_onnx.py --quantize fp16

# INT8 (recommended for CPU)
python ops/export_onnx.py --quantize int8
```

## Acceptance Thresholds

### CER (Character Error Rate)
- **Clean Dataset**: ≤ 3.0%
- **Degraded Dataset**: ≤ 10.0%
- **Measurement**: Levenshtein distance / reference length

### Latency (P95)
- **GPU (batch 8)**: ≤ 200ms (hard requirement)
- **CPU**: ≤ 500ms (soft target, reported only)
- **Measurement**: End-to-end including preprocessing

### Detection Metrics
- **IoU Threshold**: 0.5
- **Precision Target**: ≥ 0.90
- **Recall Target**: ≥ 0.85

## Tokenizer Configuration

### Using Offline Tokenizer
```bash
# Vendor tokenizer for offline use
make vendor-tokenizer

# Set environment variable
export TOKENIZER_PATH=lang/tokenizer/khopilot
```

### Online Fallback
If offline tokenizer not available:
```python
# Automatically downloads from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("khopilot/km-tokenizer-khmer")
```

## Language Model Tuning

### Optimal Parameters (from sweep)
- **λ (LM weight)**: 0.3 for clean text, 0.4 for degraded
- **μ (Lexicon weight)**: 0.1
- **Beam width**: 10 for quality, 5 for speed

### Score Scale Alignment
```python
# KenLM uses log10, CTC uses ln
lm_score_ln = lm_score_log10 * 2.3026  # ln(10)
combined = 0.7 * ctc_score + 0.3 * lm_score_ln
```

## Monitoring and Debugging

### Key Metrics
```bash
# Check service health
curl http://localhost:8080/health

# Get Prometheus metrics
curl http://localhost:8080/metrics | grep ocr_

# Important metrics:
# - ocr_latency_p95_ms{operation="ocr"}
# - ocr_cer_pass_total{dataset_type="clean"}
# - ocr_active_requests
```

### Request Tracing
All requests include correlation ID:
```json
{
  "metrics": {
    "request_id": "uuid-here",
    "latency_ms": 120.5
  }
}
```

### Debug Mode
```bash
# Enable debug info in response
curl -X POST http://localhost:8080/ocr \
  -F "file=@image.jpg" \
  -F "debug=true"
```

## Troubleshooting

### High CER
1. Check image quality (minimum 150 DPI)
2. Verify charset coverage
3. Increase LM weight (λ)
4. Check for diacritic issues

### High Latency
1. Enable ONNX runtime
2. Reduce beam width
3. Use quantization (FP16/INT8)
4. Check GPU utilization

### Memory Issues
1. Reduce batch size
2. Enable model quantization
3. Use CPU inference for large batches
4. Monitor with `nvidia-smi` or `htop`

### Dataset Validation Failures
```bash
# Check charset coverage
python ops/validate_dataset.py --data-dir data --min-coverage 99

# Common issues:
# - Missing diacritics in charset
# - Incorrect label encoding (must be UTF-8)
# - Image/label mismatch
```

## Development Tips

### Fast Iteration
```bash
# Skip slow steps during development
make dev  # Quick setup without full training

# Test with smoke data
python eval/harness.py --test data/test --max-samples 10
```

### Local Testing
```bash
# CPU-only Docker for development
docker-compose up

# Test API
python tests/integration/test_api.py
```

### Performance Profiling
```bash
# Benchmark specific configurations
python eval/benchmark.py --batch-sizes 1 8 32 --gpu

# Profile with cProfile
python -m cProfile -o profile.stats service/app.py
```