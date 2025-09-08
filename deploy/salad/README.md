# SaladCloud GPU Training Deployment

This directory contains scripts and configurations for deploying Khmer OCR training to SaladCloud with 8x L40S GPUs.

## 🚀 Quick Start

1. **Set environment variables:**
```bash
export SALAD_API_KEY="salad_cloud_user_qi2nkBgslQU4AuJvakI9lmq0CfMEWOaLa2p7fdXgaQU3mXgdM"
export WEBHOOK_SECRET_KEY="mJdnzFePyJ68JrW5XY73IVASOdZ9OZguxiCJc/I3NOzL5XgiwbHgyGzhO47dkWgg7J405IdxyNiUB7NNWvx3vA=="
export MODEL_OUTPUT_URL="https://webhook.site/your-webhook-id"
```

2. **Deploy to SaladCloud:**
```bash
cd deploy/salad
python salad-deploy.py
```

3. **Monitor training:**
   - Check SaladCloud portal: https://portal.salad.com
   - Monitor webhook endpoint for progress updates
   - Training typically takes 5-6 hours (~$13-15 total cost)

## 📁 Files Overview

### Core Deployment Files

- **`.env.salad`** - Environment configuration with API keys
- **`salad-config.yaml`** - Container group specification for SaladCloud
- **`salad-deploy.py`** - Python script for automated deployment
- **`training-entrypoint.sh`** - Main training orchestrator script
- **`distributed_train.py`** - Multi-GPU distributed training implementation

### Training Pipeline

The deployment executes the following pipeline:

1. **Environment Setup** (15 mins)
   - Install CUDA 12.1, Python 3.11, PaddleOCR
   - Download HuggingFace datasets
   - Cache khopilot/km-tokenizer-khmer tokenizer

2. **Data Preparation** (30 mins) 
   - Build corpus from khmer-dictionary-44k, SynthKhmer-10k
   - Generate Khmer character set
   - Prepare PaddleOCR format datasets

3. **Model Training** (4-5 hours)
   - **DBNet Detection**: GPUs 0-3 (2 hours)
   - **CTC Recognition**: GPUs 4-7 (3 hours)
   - **KenLM Language Model**: CPU parallel (1 hour)

4. **Evaluation & Export** (30 mins)
   - Run CER evaluation on test set
   - Update model manifest with hashes
   - Package results for download

## 🔧 Configuration Options

### Resource Allocation

- **8x L40S GPUs** (384GB total VRAM)
- **128 vCPUs** for data preprocessing
- **512 GB RAM** for large batch processing
- **500 GB storage** for models and data

### Training Parameters

```bash
# Batch sizes optimized for L40S memory
BATCH_SIZE_PER_GPU=16
TOTAL_EFFECTIVE_BATCH_SIZE=128

# Training configuration
EPOCHS_DETECTION=20
EPOCHS_RECOGNITION=25
KENLM_ORDER=5
```

### Webhook Integration

The training sends progress updates to your webhook:

```json
{
  "status": "progress",
  "message": "Training DBNet epoch 10/20",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "gpu_usage": "85,87,84,89,0,0,0,0",
    "stage": "detection_training",
    "progress_percent": 50
  }
}
```

## 💰 Cost Estimation

- **Hourly rate**: $2.56/hour
- **Typical training time**: 5-6 hours  
- **Total cost**: $13-15 for complete pipeline
- **Cost breakdown**:
  - Data prep: $1.28 (30 mins)
  - Model training: $10-13 (4-5 hours)
  - Evaluation: $1.28 (30 mins)

## 🔍 Monitoring & Debugging

### Check Training Status

```bash
# Via SaladCloud API
python -c "
from salad_deploy import SaladCloudDeployer
deployer = SaladCloudDeployer('$SALAD_API_KEY')
status = deployer.get_container_group_status('khmer-ocr-gpu-training', 'CONTAINER_GROUP_ID')
print(status)
"
```

### Access Training Logs

```bash
# Container logs available at:
# https://portal.salad.com/organizations/khmer-ocr/projects/khmer-ocr-gpu-training/container-groups/YOUR_ID/logs
```

### Health Endpoints

Once training starts, the container exposes:
- `GET /health` - Health check
- `GET /ready` - Readiness probe  
- `GET /status` - Training progress
- `GET /khmer_ocr_trained_models.tar.gz` - Download results

## 🎯 Expected Results

After successful training, you'll receive:

### Models
- `models/dbnet_kh/` - Trained DBNet detection model
- `models/rec_kh/` - Trained Khmer recognition model  
- `lang/kenlm/khmer_5gram.arpa` - KenLM language model

### Evaluation Report
```json
{
  "cer_clean": 0.028,
  "cer_degraded": 0.087, 
  "latency_p95_ms": 185,
  "acceptance_criteria": {
    "all_pass": true,
    "cer_clean_pass": true,
    "cer_degraded_pass": true,
    "latency_pass": true
  }
}
```

### Manifest
- SHA256 hashes for all models
- Training metadata and provenance
- Model version tracking

## 🚨 Troubleshooting

### Common Issues

1. **GPU memory errors**
   - Reduce `BATCH_SIZE_PER_GPU` from 16 to 8
   - Enable gradient checkpointing

2. **Dataset download failures**
   - Check HuggingFace Hub connectivity
   - Verify dataset names and availability

3. **Training stalls**
   - Monitor GPU utilization via webhook
   - Check for CUDA OOM errors in logs

4. **Webhook delivery failures**
   - Verify webhook URL is accessible
   - Check webhook secret key format

### Manual Cleanup

If deployment fails:

```bash
# Stop container group
python -c "
from salad_deploy import SaladCloudDeployer
deployer = SaladCloudDeployer('$SALAD_API_KEY')
deployer.stop_container_group('khmer-ocr-gpu-training', 'CONTAINER_GROUP_ID')
deployer.delete_container_group('khmer-ocr-gpu-training', 'CONTAINER_GROUP_ID')
"
```

## 📞 Support

- **SaladCloud Documentation**: https://docs.salad.com
- **Khmer OCR Issues**: https://github.com/khopilot/khmer-ocr-v1/issues
- **GPU Performance**: Monitor via nvidia-smi in container logs