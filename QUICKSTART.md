# Khmer OCR - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.11+
- 8GB RAM minimum
- (Optional) NVIDIA GPU with CUDA 12.x

### Installation

#### Option 1: Docker (Recommended)
```bash
# Pull and run the CPU version
docker run -p 8080:8080 ghcr.io/khopilot/khmer-ocr-cpu:latest

# Or with GPU support
docker run --gpus all -p 8080:8080 ghcr.io/khopilot/khmer-ocr-gpu:latest
```

#### Option 2: Local Installation
```bash
# Clone repository
git clone https://github.com/khopilot/khmer-ocr.git
cd khmer-ocr

# Install dependencies
make setup

# Start service
make serve
```

### First OCR Request
```bash
# Test with sample image
curl -X POST http://localhost:8080/ocr \
  -F "file=@samples/khmer_text.jpg" \
  -F "enable_lm=true"
```

Expected response:
```json
{
  "text": "សួស្តី ពិភពលោក",
  "lines": [
    {"text": "សួស្តី", "confidence": 0.95},
    {"text": "ពិភពលោក", "confidence": 0.93}
  ],
  "metrics": {
    "latency_ms": 120.5,
    "cer": 2.8
  }
}
```

## 📊 Expected Performance Metrics

### Character Error Rate (CER)
| Dataset Type | Target | Typical Range |
|-------------|--------|---------------|
| Clean (printed) | ≤ 3% | 1.5 - 3.0% |
| Degraded (scanned) | ≤ 10% | 5.0 - 10.0% |
| Handwritten | ≤ 20% | 15.0 - 25.0% |

### Latency (P95)
| Platform | Configuration | Expected P95 |
|----------|--------------|--------------|
| GPU (NVIDIA T4) | Batch 1 | 30-50ms |
| GPU (NVIDIA T4) | Batch 8 | 150-200ms |
| CPU (8 cores) | Batch 1 | 200-300ms |
| CPU (8 cores) | Batch 8 | 800-1200ms |
| Apple Silicon | Batch 1 | 300-500ms |

### Throughput
| Platform | Images/Second |
|----------|--------------|
| GPU (single) | 20-40 |
| CPU (8 cores) | 3-5 |
| Apple Silicon | 2-3 |

## 🔧 Basic Configuration

### Environment Variables
```bash
# Model variant (paddle or onnx)
export SERVICE_VARIANT=onnx

# Enable GPU
export USE_GPU=true

# Number of workers
export WORKERS=4

# Logging level
export LOG_LEVEL=info
```

### Language Model Parameters
```python
# Optimal settings for different scenarios

# Clean printed text
lm_weight = 0.3
beam_width = 10

# Degraded/noisy text
lm_weight = 0.4
beam_width = 20

# Speed-optimized
lm_weight = 0.2
beam_width = 5
```

## 📝 Common Use Cases

### 1. Document Digitization
```python
import requests

def ocr_document(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'enable_lm': True,
            'lm_weight': 0.3,
            'return_lines': True
        }
        response = requests.post(
            'http://localhost:8080/ocr',
            files=files,
            data=data
        )
    return response.json()

result = ocr_document('document.jpg')
print(f"Extracted text: {result['text']}")
print(f"CER: {result['metrics']['cer']}%")
```

### 2. Batch Processing
```bash
#!/bin/bash
# Process multiple images
for img in images/*.jpg; do
  curl -X POST http://localhost:8080/ocr \
    -F "file=@$img" \
    -F "enable_lm=true" \
    -o "results/$(basename $img .jpg).json"
done
```

### 3. Real-time OCR with Monitoring
```python
import requests
import time

def ocr_with_monitoring(image_path):
    # Add request ID for tracking
    headers = {'X-Request-ID': f'req-{int(time.time())}'}
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'debug': True}
        
        start = time.time()
        response = requests.post(
            'http://localhost:8080/ocr',
            files=files,
            data=data,
            headers=headers
        )
        total_time = time.time() - start
    
    result = response.json()
    print(f"Request ID: {result['metrics']['request_id']}")
    print(f"Server latency: {result['metrics']['latency_ms']}ms")
    print(f"Total time: {total_time*1000:.2f}ms")
    
    return result
```

## 🏥 Health Checks

### Service Health
```bash
# Check if service is ready
curl http://localhost:8080/health

# Expected response
{
  "status": "healthy",
  "models_loaded": true,
  "version": "0.1.0",
  "uptime_seconds": 3600
}
```

### Metrics Monitoring
```bash
# Get Prometheus metrics
curl http://localhost:8080/metrics | grep ocr_

# Key metrics to watch:
ocr_requests_total
ocr_latency_p95_ms
ocr_cer_pass_total
ocr_active_requests
```

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check logs
docker logs khmer-ocr

# Common issues:
# - Port 8080 already in use
# - Insufficient memory (need 4GB minimum)
# - Models not found (run: make download)
```

### Poor OCR Quality
1. Check image quality (minimum 150 DPI)
2. Enable language model: `enable_lm=true`
3. Increase LM weight: `lm_weight=0.4`
4. Verify image orientation (should be horizontal)

### High Latency
1. Enable ONNX runtime: `SERVICE_VARIANT=onnx`
2. Reduce beam width: `beam_width=5`
3. Check GPU availability: `nvidia-smi`
4. Monitor CPU usage: `htop`

## 📈 Performance Tuning

### For Best Quality
```json
{
  "enable_lm": true,
  "lm_weight": 0.4,
  "beam_width": 20,
  "SERVICE_VARIANT": "paddle"
}
```
Expected: CER < 2%, Latency ~300ms

### For Best Speed
```json
{
  "enable_lm": false,
  "beam_width": 1,
  "SERVICE_VARIANT": "onnx"
}
```
Expected: CER < 5%, Latency ~50ms

### Balanced
```json
{
  "enable_lm": true,
  "lm_weight": 0.3,
  "beam_width": 10,
  "SERVICE_VARIANT": "onnx"
}
```
Expected: CER < 3%, Latency ~150ms

## 📚 Next Steps

1. **Production Deployment**: See [DEPLOYMENT.md](DEPLOYMENT.md)
2. **API Documentation**: http://localhost:8080/docs
3. **Custom Training**: See [train/README.md](train/README.md)
4. **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🆘 Getting Help

- **Documentation**: https://docs.khopilot.ai/khmer-ocr
- **GitHub Issues**: https://github.com/khopilot/khmer-ocr/issues
- **Community**: https://discord.gg/khmer-ocr

## 📊 Sample Results

### Clean Printed Text
- Input: High-quality scan of printed document
- CER: 1.8%
- Latency: 85ms (GPU)
- Confidence: 0.96

### Degraded Scan
- Input: Old photocopied document
- CER: 7.2%
- Latency: 120ms (GPU)
- Confidence: 0.88

### Mobile Photo
- Input: Smartphone photo of text
- CER: 4.5%
- Latency: 95ms (GPU)
- Confidence: 0.92

---

**Ready to process Khmer text? You're all set! 🎉**