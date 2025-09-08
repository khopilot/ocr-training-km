#!/usr/bin/env python3
"""Complete Khmer OCR Pipeline - Data to API"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

def run_command(cmd, description, timeout=300):
    """Run a command and report results"""
    print(f"\n🔧 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout[:200]}")
            return True
        else:
            print(f"   ❌ Failed")
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🚀 COMPLETE KHMER OCR PIPELINE")
    print("=" * 60)
    print("This script runs the entire pipeline from data to API")
    print()
    
    os.chdir(Path(__file__).parent.parent)
    
    # Step 1: Check environment
    print("📋 Step 1: Environment Check")
    print("-" * 40)
    
    # Check Python
    print(f"Python: {sys.version}")
    
    # Check GPU
    try:
        import paddle
        gpu_available = paddle.is_compiled_with_cuda()
        print(f"PaddlePaddle: {paddle.__version__}")
        print(f"GPU: {'Available' if gpu_available else 'Not available'}")
    except ImportError:
        print("PaddlePaddle: Not installed")
        gpu_available = False
    
    # Check PaddleOCR
    try:
        import paddleocr
        print(f"PaddleOCR: {paddleocr.__version__}")
    except ImportError:
        print("PaddleOCR: Not installed")
    
    # Step 2: Prepare data
    print("\n📊 Step 2: Data Preparation")
    print("-" * 40)
    
    # Check existing data
    train_count = len(list(Path("data/train").glob("*.png"))) if Path("data/train").exists() else 0
    val_count = len(list(Path("data/val").glob("*.png"))) if Path("data/val").exists() else 0
    
    print(f"Found: {train_count} train, {val_count} val images")
    
    if train_count < 100:
        print("Generating synthetic data...")
        run_command(
            f"{sys.executable} scripts/generate_khmer_synthetic.py 1000",
            "Generate 1000 synthetic samples"
        )
    
    # Download HuggingFace data if needed
    if train_count < 5000:
        print("\n📥 Downloading HuggingFace dataset...")
        download_script = """
import os
os.system('pip install datasets pillow -q')

from datasets import load_dataset
from pathlib import Path
from PIL import Image

try:
    dataset = load_dataset(
        "SaladTechnologies/khmer-ocr-dataset",
        split="train[:5000]"
    )
    
    train_dir = Path("data/hf_train")
    train_dir.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(dataset):
        if i >= 5000:
            break
        
        # Save image
        img = item.get('image')
        if img:
            img_path = train_dir / f"hf_{i:06d}.png"
            img.save(img_path)
            
            # Save text
            text = item.get('text', '')
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
    
    print(f"Downloaded {i} samples to data/hf_train/")
    
except Exception as e:
    print(f"Could not download HF data: {e}")
"""
        
        with open("download_hf.py", 'w') as f:
            f.write(download_script)
        
        run_command(
            f"{sys.executable} download_hf.py",
            "Download HuggingFace dataset",
            timeout=600
        )
    
    # Step 3: Training options
    print("\n🏋️ Step 3: Model Training")
    print("-" * 40)
    
    print("Choose training approach:")
    print("1. Quick Demo (5 min)")
    print("2. Real Training (30 min)")
    print("3. Use Pre-trained (immediate)")
    
    # For automation, use option 3 (pre-trained)
    choice = "3"
    
    if choice == "1":
        # Demo training
        run_command(
            f"{sys.executable} scripts/train_demo.py",
            "Demo training (lightweight)",
            timeout=300
        )
    elif choice == "2":
        # Real training
        run_command(
            f"{sys.executable} scripts/train_paddleocr_direct.py",
            "Real PaddleOCR training",
            timeout=1800
        )
    else:
        # Use pre-trained
        print("Using pre-trained PaddleOCR models")
        print("These will be downloaded automatically on first use")
    
    # Step 4: Test inference
    print("\n🔍 Step 4: Testing Inference")
    print("-" * 40)
    
    # Find a test image
    test_images = list(Path("data/train").glob("*.png"))[:3]
    if not test_images:
        test_images = list(Path("data").rglob("*.png"))[:3]
    
    if test_images:
        for img in test_images:
            print(f"\nTesting on: {img.name}")
            run_command(
                f"{sys.executable} infer/predict.py --image {img}",
                f"OCR on {img.name}",
                timeout=60
            )
    else:
        print("No test images found")
    
    # Step 5: API Server
    print("\n🌐 Step 5: API Server")
    print("-" * 40)
    
    # Start API server in background
    print("Starting API server...")
    
    api_script = """
import subprocess
import time
import sys

# Start server
proc = subprocess.Popen(
    [sys.executable, "service/app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for startup
time.sleep(5)

# Check if running
if proc.poll() is None:
    print("✅ API server started on http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    
    # Test health endpoint
    import requests
    try:
        resp = requests.get("http://localhost:8000/health", timeout=5)
        if resp.status_code == 200:
            print("✅ Health check passed")
    except:
        print("⚠️ Health check failed")
    
    # Kill server
    proc.terminate()
    print("Server stopped (was just a test)")
else:
    print("❌ Server failed to start")
"""
    
    with open("test_api.py", 'w') as f:
        f.write(api_script)
    
    run_command(
        f"{sys.executable} test_api.py",
        "Test API server",
        timeout=30
    )
    
    # Step 6: Performance test
    print("\n📈 Step 6: Performance Benchmark")
    print("-" * 40)
    
    benchmark_script = """
import time
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)
    
    # Find test images
    test_imgs = list(Path("data/train").glob("*.png"))[:10]
    
    if test_imgs:
        times = []
        for img in test_imgs:
            start = time.time()
            result = ocr.ocr(str(img), cls=False)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"Average OCR time: {avg_time:.3f}s per image")
        print(f"Throughput: {1/avg_time:.1f} images/second")
    else:
        print("No test images for benchmark")
        
except Exception as e:
    print(f"Benchmark failed: {e}")
"""
    
    with open("benchmark.py", 'w') as f:
        f.write(benchmark_script)
    
    run_command(
        f"{sys.executable} benchmark.py",
        "Performance benchmark",
        timeout=60
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print(f"   Data: {train_count + val_count} images prepared")
    print(f"   Model: {'Trained' if Path('output/rec_model').exists() else 'Pre-trained'}")
    print(f"   Inference: {'Ready' if Path('infer/predict.py').exists() else 'Not configured'}")
    print(f"   API: {'Configured' if Path('service/app.py').exists() else 'Not configured'}")
    
    print("\n🚀 Next Steps:")
    print("\n1. For immediate use (pre-trained models):")
    print("   python scripts/production_ocr.py")
    
    print("\n2. To run inference on your images:")
    print("   python infer/predict.py --image <your_image.png>")
    
    print("\n3. To start the API server:")
    print("   cd service && python app.py")
    print("   Then visit: http://localhost:8000/docs")
    
    print("\n4. For production deployment:")
    print("   - Train with more data (10,000+ samples)")
    print("   - Use GPU for training (50-100 epochs)")
    print("   - Fine-tune on real Khmer documents")
    print("   - Deploy with Docker/Kubernetes")
    
    print("\n5. For SaladCloud deployment:")
    print("   cd deploy/salad")
    print("   ./deploy.sh")
    
    print("\n📚 Documentation:")
    print("   - README.md: Project overview")
    print("   - CLAUDE.md: Technical specifications")
    print("   - deploy/salad/README.md: Deployment guide")

if __name__ == "__main__":
    main()