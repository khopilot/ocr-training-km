#!/bin/bash
# Complete Khmer OCR Pipeline - Training to Inference to API

echo "🚀 COMPLETE KHMER OCR PIPELINE"
echo "======================================"
echo ""

cd /ocr-training-km
git pull

# Step 1: Test with pre-trained models first
echo "📊 Step 1: Testing with pre-trained models..."
echo "----------------------------------------"

# Test on a sample image
if [ -f "data/train/synth_000001.png" ]; then
    echo "Testing OCR on sample image..."
    python infer/predict.py --image data/train/synth_000001.png
else
    echo "No test image found. Generating one..."
    python scripts/generate_khmer_synthetic.py 10
    python infer/predict.py --image data/train/synth_000001.png
fi

echo ""
echo "📊 Step 2: Running API Server..."
echo "----------------------------------------"

# Fix and start API server
cd service
python app.py &
API_PID=$!
echo "API Server started with PID: $API_PID"

# Wait for server to start
sleep 3

# Test API endpoint
echo "Testing API health endpoint..."
curl -X GET "http://localhost:8000/health" || echo "API not responding"

# Kill the API server
kill $API_PID 2>/dev/null

cd ..

echo ""
echo "✅ Pipeline test complete!"
echo ""
echo "======================================"
echo "NEXT STEPS FOR PRODUCTION:"
echo "======================================"
echo ""
echo "1. For better OCR accuracy:"
echo "   - Train for more epochs (50-100)"
echo "   - Use more training data"
echo "   - Fine-tune on Khmer-specific data"
echo ""
echo "2. To run inference:"
echo "   python infer/predict.py --image <your_image.png>"
echo ""
echo "3. To start API server:"
echo "   cd service && python app.py"
echo "   Then visit: http://localhost:8000/docs"
echo ""
echo "4. To train with real weights (GPU required):"
echo "   Use Docker: make docker-train-rec"
echo "   Or cloud GPU: Use SaladCloud/Colab"
echo ""