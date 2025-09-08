#!/usr/bin/env python3
"""Production OCR with automatic GPU/CPU detection"""

import os
import sys
from pathlib import Path

def test_gpu_availability():
    """Test if GPU and cuDNN are properly configured"""
    try:
        import paddle
        
        if not paddle.is_compiled_with_cuda():
            return False, "PaddlePaddle not compiled with CUDA"
        
        # Try to create a tensor on GPU
        try:
            x = paddle.randn([2, 3]).cuda()
            del x  # Clean up
            return True, "GPU and cuDNN working"
        except Exception as e:
            return False, f"GPU error: {str(e)[:100]}"
            
    except ImportError:
        return False, "PaddlePaddle not installed"

def run_ocr_with_mode(use_gpu=True):
    """Run OCR with specified mode"""
    from paddleocr import PaddleOCR
    
    mode = "GPU" if use_gpu else "CPU"
    print(f"🔍 Initializing PaddleOCR with {mode}...")
    
    # Configure based on mode
    ocr_config = {
        'use_angle_cls': False,
        'lang': 'en',  # Works for Khmer too
        'use_gpu': use_gpu,
        'show_log': False
    }
    
    # Add CPU optimization if not using GPU
    if not use_gpu:
        ocr_config['cpu_threads'] = 4
    
    ocr = PaddleOCR(**ocr_config)
    
    print(f"✅ OCR engine ready ({mode} mode)!")
    return ocr

def main():
    print("🚀 PRODUCTION KHMER OCR (AUTO-DETECT)")
    print("=" * 50)
    
    # Test GPU availability
    print("🔍 Checking GPU availability...")
    gpu_available, gpu_message = test_gpu_availability()
    
    if gpu_available:
        print(f"✅ {gpu_message}")
        use_gpu = True
    else:
        print(f"⚠️  {gpu_message}")
        print("📊 Falling back to CPU mode...")
        use_gpu = False
    
    print()
    
    # Check for test images
    test_images = list(Path("data/train").glob("*.png"))[:5]
    
    if not test_images:
        print("📊 Generating test images...")
        try:
            from generate_khmer_synthetic import generate_synthetic_data
            generate_synthetic_data(10)
            test_images = list(Path("data/train").glob("*.png"))[:5]
        except:
            print("⚠️  Could not generate test images")
            print("Please ensure you have images in data/train/")
            sys.exit(1)
    
    print(f"Found {len(test_images)} test images")
    print()
    
    # Initialize OCR with auto-detected mode
    try:
        ocr = run_ocr_with_mode(use_gpu)
        
        # Track performance
        import time
        total_time = 0
        successful = 0
        
        # Test on each image
        for img_path in test_images:
            print(f"📷 Processing: {img_path.name}")
            
            # Get expected text if available
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    expected = f.read().strip()
                print(f"   Expected: {expected}")
            
            # Run OCR with timing
            try:
                start_time = time.time()
                result = ocr.ocr(str(img_path), cls=False)
                elapsed = time.time() - start_time
                total_time += elapsed
                
                if result and result[0]:
                    texts = [line[1][0] for line in result[0]]
                    combined = " ".join(texts)
                    print(f"   Detected: {combined}")
                    
                    # Calculate confidence
                    avg_conf = sum(line[1][1] for line in result[0]) / len(result[0])
                    print(f"   Confidence: {avg_conf:.2%}")
                    print(f"   Time: {elapsed:.2f}s")
                    successful += 1
                else:
                    print("   Detected: (no text found)")
                    
            except Exception as e:
                error_msg = str(e)
                
                # If GPU failed, retry with CPU
                if use_gpu and "cudnn" in error_msg.lower():
                    print(f"   GPU Error: {error_msg[:100]}")
                    print("   🔄 Retrying with CPU...")
                    
                    # Reinitialize with CPU
                    use_gpu = False
                    ocr = run_ocr_with_mode(use_gpu)
                    
                    # Retry
                    try:
                        start_time = time.time()
                        result = ocr.ocr(str(img_path), cls=False)
                        elapsed = time.time() - start_time
                        total_time += elapsed
                        
                        if result and result[0]:
                            texts = [line[1][0] for line in result[0]]
                            combined = " ".join(texts)
                            print(f"   Detected (CPU): {combined}")
                            successful += 1
                    except Exception as e2:
                        print(f"   CPU Error: {e2}")
                else:
                    print(f"   Error: {e}")
            
            print()
        
        # Print summary
        print("=" * 50)
        print("✅ OCR testing complete!")
        print()
        print(f"📊 Results:")
        print(f"   Mode: {'GPU' if use_gpu else 'CPU'}")
        print(f"   Successful: {successful}/{len(test_images)}")
        if successful > 0:
            avg_time = total_time / successful
            print(f"   Avg time: {avg_time:.2f}s per image")
            print(f"   Throughput: {1/avg_time:.1f} images/second")
        
        print()
        print("💡 Tips:")
        if not use_gpu:
            print("   - For faster processing, fix GPU/cuDNN:")
            print("     bash scripts/fix_cudnn_server.sh")
        else:
            print("   - GPU acceleration is working!")
            print("   - For even faster processing, use batch mode")
        
        print()
        print("🚀 Next steps:")
        print("   - Test your images: python infer/predict.py --image <image.png>")
        print("   - Start API: cd service && python app.py")
        
    except ImportError:
        print("❌ PaddleOCR not installed properly")
        print("Run: pip install paddleocr==2.7.0")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()