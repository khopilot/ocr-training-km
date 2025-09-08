#!/usr/bin/env python3
"""Verify PaddleOCR setup and GPU availability"""

import sys
import os

def verify_setup():
    print("🔍 Verifying Khmer OCR Training Setup...")
    print("=" * 50)
    
    # 1. Check Python version
    print(f"✅ Python version: {sys.version}")
    
    # 2. Check NumPy
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    # 3. Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    # 4. Check PaddlePaddle
    try:
        import paddle
        print(f"✅ PaddlePaddle version: {paddle.__version__}")
        
        # Check if GPU is available
        if paddle.is_compiled_with_cuda():
            print(f"✅ PaddlePaddle compiled with CUDA")
            gpu_count = paddle.device.cuda.device_count()
            print(f"✅ Available GPUs: {gpu_count}")
            if gpu_count > 0:
                for i in range(gpu_count):
                    props = paddle.device.cuda.get_device_properties(i)
                    print(f"   GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  PaddlePaddle not compiled with CUDA - will use CPU (slow)")
            
    except ImportError as e:
        print(f"❌ PaddlePaddle import failed: {e}")
        return False
    
    # 5. Check PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print(f"✅ PaddleOCR imported successfully")
        
        # Try to initialize PaddleOCR
        print("Testing PaddleOCR initialization...")
        ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)
        print("✅ PaddleOCR initialized successfully")
        
    except ImportError as e:
        print(f"❌ PaddleOCR import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  PaddleOCR initialization warning: {e}")
        print("   This might be normal if models haven't been downloaded yet")
    
    # 6. Check data directory
    print("\n📁 Checking data directories...")
    data_dirs = ['data/train', 'data/val', 'data/test', 'data/synth']
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✅ {dir_path}: {count} images")
        else:
            print(f"❌ {dir_path}: Directory not found")
    
    # 7. Check config files
    print("\n📄 Checking config files...")
    config_files = [
        'train/configs/rec_kh.yaml',
        'train/configs/dbnet.yaml',
        'train/configs/rec_kh_hf.yaml',
        'train/configs/dbnet_hf.yaml'
    ]
    for config in config_files:
        if os.path.exists(config):
            print(f"✅ {config}: Found")
        else:
            print(f"❌ {config}: Not found")
    
    print("\n" + "=" * 50)
    print("✅ Setup verification complete!")
    print("\nNext steps:")
    print("1. Run: python scripts/prepare_data_lists.py")
    print("2. Run: bash scripts/download_pretrained.sh")
    print("3. Start training: python train/train_demo.py")
    
    return True

if __name__ == "__main__":
    verify_setup()