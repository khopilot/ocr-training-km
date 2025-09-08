# Khmer OCR - Deployment Guide

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or RHEL 8+
- **CPU**: 8+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 50GB for models and data

### Software Requirements
- Docker 20.10+
- Kubernetes 1.24+ (for K8s deployment)
- Helm 3.0+ (for Helm deployment)
- Python 3.11 (for local development)
- CUDA 12.x (for GPU support)

## Deployment Options

### 1. Docker Compose (Recommended for Single Node)

#### Quick Start
```bash
# Clone repository
git clone https://github.com/khopilot/khmer-ocr.git
cd khmer-ocr

# Build images
make docker-build

# Start services
docker-compose up -d

# Verify deployment
curl http://localhost:8080/health
```

#### Production Configuration
```bash
# Use production overrides
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With GPU support
export COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml
docker-compose up -d
```

#### Environment Variables
```bash
# Create .env file
cat > .env << EOF
SERVICE_VARIANT=onnx
USE_GPU=true
WORKERS=4
LOG_LEVEL=info
MAX_REQUEST_SIZE=50M
REQUEST_TIMEOUT=30
TOKENIZER_PATH=/app/lang/tokenizer/khopilot
MODEL_DIR=/app/models
EOF
```

### 2. Kubernetes with Helm

#### Install Helm Chart
```bash
# Add repository (if published)
helm repo add khmer-ocr https://charts.khopilot.ai
helm repo update

# Install with default values
helm install khmer-ocr khmer-ocr/khmer-ocr

# Install with custom values
helm install khmer-ocr khmer-ocr/khmer-ocr \
  --set image.tag=v0.1.0 \
  --set replicaCount=3 \
  --set resources.limits.memory=8Gi \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=ocr.example.com
```

#### Custom Values
```yaml
# values-prod.yaml
replicaCount: 3

image:
  repository: ghcr.io/khopilot/khmer-ocr-gpu
  tag: v0.1.0

resources:
  limits:
    memory: 8Gi
    nvidia.com/gpu: 1
  requests:
    cpu: 4
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  SERVICE_VARIANT: "onnx"
  USE_GPU: "true"
  WORKERS: "4"

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 20Gi
```

#### Deploy with Custom Values
```bash
helm install khmer-ocr ./deploy/helm/khmer-ocr \
  -f values-prod.yaml \
  --namespace ocr \
  --create-namespace
```

### 3. Kubernetes (Raw Manifests)

```bash
# Create namespace
kubectl create namespace khmer-ocr

# Apply configurations
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/ingress.yaml

# Verify deployment
kubectl get pods -n khmer-ocr
kubectl get svc -n khmer-ocr
```

### 4. Cloud Deployments

#### AWS EKS
```bash
# Create EKS cluster
eksctl create cluster --name khmer-ocr \
  --version 1.27 \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type g4dn.xlarge \
  --nodes 2

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy application
helm install khmer-ocr ./deploy/helm/khmer-ocr
```

#### Google GKE
```bash
# Create GKE cluster with GPU
gcloud container clusters create khmer-ocr \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 2

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy
helm install khmer-ocr ./deploy/helm/khmer-ocr
```

## Model Management

### Download Pre-trained Models
```bash
# Download from release
wget https://github.com/khopilot/khmer-ocr/releases/download/v0.1.0/models.tar.gz
tar -xzf models.tar.gz -C models/

# Or use DVC
dvc pull models
```

### Mount Models in Kubernetes
```yaml
# Create ConfigMap from models
kubectl create configmap ocr-models \
  --from-file=models/ \
  -n khmer-ocr

# Or use PersistentVolume
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ocr-models-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadOnlyMany
  nfs:
    server: nfs-server.example.com
    path: /exports/ocr-models
EOF
```

## Monitoring Setup

### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'khmer-ocr'
    static_configs:
      - targets: ['khmer-ocr:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard
Import dashboard from `docker/grafana-dashboard.json` or ID: 15897

Key metrics to monitor:
- Request rate and latency
- CER pass rate
- GPU utilization
- Memory usage
- Error rate

### Alerting Rules
```yaml
# alerts.yml
groups:
  - name: ocr
    rules:
      - alert: HighErrorRate
        expr: rate(ocr_errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High OCR error rate"
      
      - alert: HighLatency
        expr: ocr_latency_p95_ms{operation="ocr"} > 500
        for: 10m
        annotations:
          summary: "OCR P95 latency exceeds threshold"
      
      - alert: LowCERPassRate
        expr: rate(ocr_cer_pass_total[1h]) < 0.9
        for: 30m
        annotations:
          summary: "CER pass rate below 90%"
```

## Security Configuration

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ocr-network-policy
spec:
  podSelector:
    matchLabels:
      app: khmer-ocr
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: nginx-ingress
      ports:
        - protocol: TCP
          port: 8080
```

### TLS Configuration
```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt \
  -subj "/CN=ocr.example.com"

# Create secret
kubectl create secret tls ocr-tls \
  --cert=tls.crt \
  --key=tls.key \
  -n khmer-ocr
```

### Rate Limiting
```yaml
# nginx ingress annotation
metadata:
  annotations:
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/limit-connections: "5"
```

## Backup and Recovery

### Database Backup (if using persistent state)
```bash
# Backup PVC
kubectl exec -n khmer-ocr deployment/khmer-ocr -- \
  tar czf - /app/data | gzip > backup-$(date +%Y%m%d).tar.gz

# Restore
kubectl exec -n khmer-ocr -i deployment/khmer-ocr -- \
  tar xzf - -C / < backup-20240101.tar.gz
```

### Model Versioning
```bash
# Tag models with version
git tag -a models-v0.1.0 -m "Release models v0.1.0"
git push origin models-v0.1.0

# Store in S3
aws s3 cp models/ s3://khmer-ocr-models/v0.1.0/ --recursive
```

## Scaling Strategies

### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ocr-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: khmer-ocr
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Vertical Pod Autoscaling
```bash
# Install VPA
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/latest/download/vertical-pod-autoscaler.yaml

# Create VPA
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ocr-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: khmer-ocr
  updatePolicy:
    updateMode: "Auto"
EOF
```

## Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n khmer-ocr

# Check logs
kubectl logs <pod-name> -n khmer-ocr

# Common causes:
# - Insufficient resources
# - GPU not available
# - Model files not mounted
```

#### High Memory Usage
```bash
# Check memory usage
kubectl top pods -n khmer-ocr

# Solutions:
# - Reduce batch size
# - Enable quantization
# - Increase memory limits
```

#### Slow Response Times
```bash
# Check metrics
curl http://service:8080/metrics | grep latency

# Solutions:
# - Enable ONNX runtime
# - Scale horizontally
# - Check GPU utilization
```

## Maintenance

### Rolling Updates
```bash
# Update image
kubectl set image deployment/khmer-ocr \
  ocr=ghcr.io/khopilot/khmer-ocr-gpu:v0.2.0 \
  -n khmer-ocr

# Monitor rollout
kubectl rollout status deployment/khmer-ocr -n khmer-ocr
```

### Health Checks
```bash
# Liveness probe
curl http://service:8080/health

# Readiness probe
curl http://service:8080/health

# Metrics check
curl http://service:8080/metrics
```

### Log Collection
```bash
# View logs
kubectl logs -f deployment/khmer-ocr -n khmer-ocr

# Stream to logging system
kubectl logs -f deployment/khmer-ocr -n khmer-ocr | \
  fluentd -c /etc/fluentd/fluent.conf
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/khopilot/khmer-ocr/issues
- Documentation: https://docs.khopilot.ai/khmer-ocr
- Email: support@khopilot.ai