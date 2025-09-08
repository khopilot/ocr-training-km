#!/bin/bash
# Fix cuDNN on SaladCloud Ubuntu 24.04 Server

echo "🔧 FIXING CUDNN FOR PADDLEPADDLE"
echo "=================================="
echo ""

# Detect if running as root
if [ "$EUID" -eq 0 ]; then
    echo "✅ Running as root"
    SUDO=""
else
    echo "ℹ️  Running as user, will use sudo"
    SUDO="sudo"
fi

# Check CUDA version
echo "📊 Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1-2)
    echo "Found CUDA version: $CUDA_VERSION"
else
    echo "❌ nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

# Install cuDNN based on CUDA version
echo ""
echo "📦 Installing cuDNN..."

# For CUDA 12.x on Ubuntu 24.04
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "Installing cuDNN for CUDA 12.x..."
    
    # Method 1: Using apt (recommended)
    $SUDO apt-get update
    $SUDO apt-get install -y libcudnn8 libcudnn8-dev
    
    # If above fails, try manual installation
    if [ $? -ne 0 ]; then
        echo "Trying alternative installation method..."
        
        # Download cuDNN manually (CUDA 12.x compatible)
        wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2404-9.0.0_1.0-1_amd64.deb
        $SUDO dpkg -i cudnn-local-repo-ubuntu2404-9.0.0_1.0-1_amd64.deb
        $SUDO cp /var/cudnn-local-repo-ubuntu2404-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
        $SUDO apt-get update
        $SUDO apt-get install -y cudnn
        rm cudnn-local-repo-ubuntu2404-9.0.0_1.0-1_amd64.deb
    fi
    
elif [[ "$CUDA_VERSION" == "11."* ]]; then
    echo "Installing cuDNN for CUDA 11.x..."
    
    # For CUDA 11.x
    $SUDO apt-get update
    $SUDO apt-get install -y libcudnn8=8.9.7.29-1+cuda11.8 libcudnn8-dev=8.9.7.29-1+cuda11.8
    
else
    echo "⚠️  Unknown CUDA version: $CUDA_VERSION"
    echo "Trying generic cuDNN installation..."
    $SUDO apt-get update
    $SUDO apt-get install -y libcudnn8 libcudnn8-dev
fi

# Set environment variables
echo ""
echo "🔧 Setting environment variables..."

# Add to current session
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

# Add to bashrc for persistence
echo "# CUDA and cuDNN paths" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc

# Also add to /etc/environment for system-wide
if [ "$EUID" -eq 0 ]; then
    echo "LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:\$LD_LIBRARY_PATH\"" >> /etc/environment
fi

# Verify cuDNN installation
echo ""
echo "📋 Verifying cuDNN installation..."

# Check if libcudnn is installed
if ls /usr/lib/x86_64-linux-gnu/libcudnn* 2>/dev/null || ls /usr/local/cuda/lib64/libcudnn* 2>/dev/null; then
    echo "✅ cuDNN libraries found"
    
    # Try to get version
    if [ -f /usr/include/cudnn_version.h ]; then
        grep CUDNN_MAJOR /usr/include/cudnn_version.h
    fi
else
    echo "⚠️  cuDNN libraries not found in standard locations"
fi

# Test with Python
echo ""
echo "🐍 Testing PaddlePaddle GPU support..."

python3 << EOF
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import paddle
    print(f"PaddlePaddle version: {paddle.__version__}")
    print(f"CUDA available: {paddle.is_compiled_with_cuda()}")
    
    if paddle.is_compiled_with_cuda():
        # Try to create a tensor on GPU
        try:
            x = paddle.randn([2, 3]).cuda()
            print("✅ GPU tensor creation successful!")
        except Exception as e:
            print(f"❌ GPU tensor creation failed: {e}")
    else:
        print("❌ PaddlePaddle not compiled with CUDA support")
        print("You may need to reinstall: pip install paddlepaddle-gpu==2.6.1")
        
except ImportError:
    print("❌ PaddlePaddle not installed")
    print("Install with: pip install paddlepaddle-gpu==2.6.1")
except Exception as e:
    print(f"❌ Error: {e}")
EOF

echo ""
echo "=================================="
echo "📊 Next steps:"
echo ""
echo "1. If cuDNN is now working:"
echo "   python scripts/production_ocr.py"
echo ""
echo "2. If still having issues, use CPU mode:"
echo "   python scripts/production_ocr_cpu.py"
echo ""
echo "3. You may need to restart your shell or run:"
echo "   source ~/.bashrc"
echo ""
echo "4. If PaddlePaddle needs GPU version:"
echo "   pip uninstall paddlepaddle paddlepaddle-gpu"
echo "   pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/cudaxxx/stable.html"
echo "   (replace xxx with your CUDA version, e.g., cuda120 for CUDA 12.0)"
echo ""