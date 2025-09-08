#!/usr/bin/env python3
"""Diagnostic script for PaddlePaddle GPU/cuDNN setup"""

import os
import sys
import subprocess

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None

def main():
    print("🔍 PADDLEPADDLE GPU DIAGNOSTIC")
    print("=" * 60)
    print()
    
    # 1. Check system info
    print("📊 System Information:")
    print("-" * 40)
    
    # OS info
    os_info = run_command("lsb_release -a 2>/dev/null | grep Description | cut -f2")
    if os_info:
        print(f"OS: {os_info}")
    else:
        print(f"OS: {sys.platform}")
    
    # Python version
    print(f"Python: {sys.version.split()[0]}")
    
    # Check for NVIDIA driver
    print()
    print("🎮 GPU Information:")
    print("-" * 40)
    
    nvidia_smi = run_command("nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>/dev/null")
    if nvidia_smi:
        print(f"GPU: {nvidia_smi}")
        
        # Get more detailed CUDA info
        cuda_version = run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
        if cuda_version:
            print(f"CUDA Runtime: {cuda_version}")
    else:
        print("❌ No NVIDIA GPU detected or nvidia-smi not available")
    
    # Check CUDA installation
    print()
    print("🔧 CUDA Installation:")
    print("-" * 40)
    
    nvcc = run_command("nvcc --version | grep release")
    if nvcc:
        print(f"NVCC: {nvcc}")
    else:
        print("❌ NVCC not found (CUDA toolkit may not be installed)")
    
    cuda_home = os.environ.get('CUDA_HOME', '')
    if cuda_home:
        print(f"CUDA_HOME: {cuda_home}")
    else:
        print("⚠️  CUDA_HOME not set")
    
    # Check cuDNN
    print()
    print("📦 cuDNN Status:")
    print("-" * 40)
    
    # Check common cuDNN locations
    cudnn_locations = [
        "/usr/lib/x86_64-linux-gnu/libcudnn.so",
        "/usr/local/cuda/lib64/libcudnn.so",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.9"
    ]
    
    cudnn_found = False
    for loc in cudnn_locations:
        if os.path.exists(loc):
            print(f"✅ Found cuDNN: {loc}")
            cudnn_found = True
            
            # Try to get version
            cudnn_version = run_command(f"strings {loc} | grep CUDNN | head -1")
            if cudnn_version:
                print(f"   Version info: {cudnn_version}")
            break
    
    if not cudnn_found:
        print("❌ cuDNN not found in standard locations")
        print("   Install with: sudo apt-get install libcudnn8")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        print(f"LD_LIBRARY_PATH: {ld_path[:100]}...")
    else:
        print("⚠️  LD_LIBRARY_PATH not set")
    
    # Check PaddlePaddle
    print()
    print("🐍 PaddlePaddle Status:")
    print("-" * 40)
    
    try:
        import paddle
        print(f"✅ PaddlePaddle version: {paddle.__version__}")
        
        # Check CUDA compilation
        cuda_compiled = paddle.is_compiled_with_cuda()
        print(f"   Compiled with CUDA: {cuda_compiled}")
        
        if cuda_compiled:
            # Try to get CUDA version paddle was compiled with
            try:
                cuda_version = paddle.version.cuda()
                cudnn_version = paddle.version.cudnn()
                print(f"   Built for CUDA: {cuda_version}")
                print(f"   Built for cuDNN: {cudnn_version}")
            except:
                pass
            
            # Test GPU functionality
            print()
            print("🧪 Testing GPU functionality:")
            
            try:
                # Set environment to help find cuDNN
                os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
                
                # Try simple tensor operation
                x = paddle.randn([2, 3])
                x = x.cuda()  # Move to GPU
                y = paddle.randn([2, 3]).cuda()
                z = x + y  # GPU computation
                result = z.numpy()  # Move back to CPU
                print("   ✅ GPU tensor operations: Working")
                
                # Check available GPUs
                gpu_count = paddle.device.cuda.device_count()
                print(f"   Available GPUs: {gpu_count}")
                
                if gpu_count > 0:
                    for i in range(gpu_count):
                        props = paddle.device.cuda.get_device_properties(i)
                        print(f"   GPU {i}: {props.name}")
                        print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
                
            except Exception as e:
                print(f"   ❌ GPU operations failed: {e}")
                
                # Check if it's cuDNN specific
                if "cudnn" in str(e).lower():
                    print()
                    print("   📋 cuDNN Error Detected!")
                    print("   This is the issue preventing GPU usage.")
                    print()
                    print("   🔧 To fix:")
                    print("   1. Run: bash scripts/fix_cudnn_server.sh")
                    print("   2. Or reinstall PaddlePaddle for your CUDA version:")
                    print("      pip uninstall paddlepaddle paddlepaddle-gpu")
                    print("      pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/cudann/stable.html")
                    print("      (replace 'cudann' with your CUDA version, e.g., 'cuda120' for CUDA 12.0)")
        else:
            print("   ❌ PaddlePaddle CPU-only version installed")
            print("   Install GPU version:")
            print("   pip uninstall paddlepaddle")
            print("   pip install paddlepaddle-gpu==2.6.1")
            
    except ImportError:
        print("❌ PaddlePaddle not installed")
        print("   Install with: pip install paddlepaddle-gpu==2.6.1")
    except Exception as e:
        print(f"❌ Error testing PaddlePaddle: {e}")
    
    # Check PaddleOCR
    print()
    print("📷 PaddleOCR Status:")
    print("-" * 40)
    
    try:
        import paddleocr
        print(f"✅ PaddleOCR version: {paddleocr.__version__}")
        
        # Try to initialize
        from paddleocr import PaddleOCR
        
        print("   Testing OCR initialization...")
        try:
            # Try GPU first
            ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, show_log=False)
            print("   ✅ GPU mode: Working")
        except Exception as e:
            if "cudnn" in str(e).lower():
                print(f"   ❌ GPU mode failed: cuDNN issue")
            else:
                print(f"   ❌ GPU mode failed: {str(e)[:100]}")
            
            # Try CPU
            try:
                ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
                print("   ✅ CPU mode: Working")
            except Exception as e:
                print(f"   ❌ CPU mode also failed: {e}")
                
    except ImportError:
        print("❌ PaddleOCR not installed")
        print("   Install with: pip install paddleocr==2.7.0")
    except Exception as e:
        print(f"❌ Error testing PaddleOCR: {e}")
    
    # Summary and recommendations
    print()
    print("=" * 60)
    print("📊 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    
    # Determine the main issue
    if not nvidia_smi:
        print("❌ Main Issue: No GPU detected")
        print("   Solution: This machine may not have a GPU or NVIDIA drivers")
        print("   Use CPU mode: python scripts/production_ocr_cpu.py")
    elif not cudnn_found:
        print("❌ Main Issue: cuDNN not installed")
        print("   Solution:")
        print("   1. Run: bash scripts/fix_cudnn_server.sh")
        print("   2. Then: source ~/.bashrc")
        print("   3. Test: python scripts/production_ocr_auto.py")
    elif 'paddle' in sys.modules and not paddle.is_compiled_with_cuda():
        print("❌ Main Issue: PaddlePaddle CPU-only version")
        print("   Solution:")
        print("   1. pip uninstall paddlepaddle")
        print("   2. pip install paddlepaddle-gpu==2.6.1")
        print("   3. Test: python scripts/production_ocr_auto.py")
    else:
        print("✅ Setup looks good!")
        print("   Try: python scripts/production_ocr.py")
    
    print()
    print("🚀 Quick Commands:")
    print("   - Auto-detect mode: python scripts/production_ocr_auto.py")
    print("   - CPU-only mode: python scripts/production_ocr_cpu.py")
    print("   - Fix cuDNN: bash scripts/fix_cudnn_server.sh")

if __name__ == "__main__":
    main()