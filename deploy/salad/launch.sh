#!/bin/bash
set -euo pipefail

# SaladCloud Khmer OCR Training Launcher
# Quick deployment script for 8x L40S GPU training

echo "🥗 SaladCloud Khmer OCR Training Launcher"
echo "=========================================="

# Check if running from correct directory
if [[ ! -f "salad-deploy.py" ]]; then
    echo "❌ Please run this script from the deploy/salad/ directory"
    echo "   cd deploy/salad && ./launch.sh"
    exit 1
fi

# Load environment variables from .env.salad
if [[ -f ".env.salad" ]]; then
    echo "📋 Loading environment from .env.salad"
    export $(grep -v '^#' .env.salad | xargs)
else
    echo "❌ .env.salad file not found"
    echo "   Please create it with your API keys"
    exit 1
fi

# Verify required environment variables
required_vars=("SALAD_API_KEY" "WEBHOOK_SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "❌ Missing required environment variable: $var"
        echo "   Please add it to .env.salad"
        exit 1
    fi
done

echo "✅ Environment variables loaded"
echo "🔑 API Key: ${SALAD_API_KEY:0:20}..."
echo "🔐 Webhook Secret: ${WEBHOOK_SECRET_KEY:0:20}..."

# Show configuration summary
echo ""
echo "🚀 Deployment Configuration:"
echo "   Container Group: $CONTAINER_GROUP_NAME"
echo "   Project: $PROJECT_NAME"
echo "   GPUs: $NUM_GPUS x L40S"
echo "   Batch Size: $TOTAL_BATCH_SIZE"
echo "   Estimated Cost: ~\$15 for 6-hour training"

# Confirm deployment
echo ""
read -p "🤔 Proceed with deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

echo ""
echo "🚀 Starting deployment..."

# Check Python dependencies
echo "🐍 Checking Python dependencies..."
python -c "import requests, yaml, json; print('✅ Dependencies available')" || {
    echo "❌ Missing Python dependencies. Installing..."
    pip install requests pyyaml
}

# Run the deployment
echo "🎯 Deploying to SaladCloud..."
python salad-deploy.py

# Check deployment result
if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Next Steps:"
    echo "   1. Monitor training at: https://portal.salad.com"
    echo "   2. Check webhook endpoint for progress updates"
    echo "   3. Training will take ~5-6 hours (~\$13-15 total cost)"
    echo "   4. Results will be packaged as khmer_ocr_trained_models.tar.gz"
    echo ""
    echo "📱 Webhook notifications will be sent to:"
    echo "   ${MODEL_OUTPUT_URL:-https://webhook.site/test}"
    echo ""
    echo "💡 Tip: Keep this terminal open to see deployment monitoring"
else
    echo "❌ Deployment failed!"
    echo "   Check the error messages above for troubleshooting"
    exit 1
fi