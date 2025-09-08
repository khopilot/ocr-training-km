#!/bin/bash
# Fixed SaladCloud Ubuntu Setup Script for Khmer OCR Training
# Resolves numpy conflicts and dependency issues

set -e  # Exit on error

echo "🥗 SaladCloud Ubuntu Setup for Khmer OCR Training (Fixed)"
echo "=========================================================="

# Check if running on Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        echo "❌ This script is designed for Ubuntu. Detected: $ID"
        exit 1
    fi
    echo "✅ Ubuntu environment detected"
    cat /etc/lsb-release
else
    echo "❌ Cannot detect OS. This script requires Ubuntu."
    exit 1
fi

# Check GPU availability
echo -e "\n🔍 Checking GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers available"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
else
    echo "⚠️  NVIDIA drivers not found. GPU training will not be available."
fi

# Create environment variables file
echo -e "\n📝 Creating environment configuration..."
cat > ~/.bashrc_salad << 'EOF'
# SaladCloud Environment Variables
export SALAD_MACHINE_ID=${SALAD_MACHINE_ID:-$(hostname)}
export SALAD_CONTAINER_GROUP_ID=${SALAD_CONTAINER_GROUP_ID:-local}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
# Set OMP_NUM_THREADS to 1 for PaddlePaddle optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Add local bin to PATH
export PATH=$HOME/.local/bin:$PATH

# KenLM paths
export KENLM_ROOT=$HOME/kenlm
export PATH=$KENLM_ROOT/build/bin:$PATH

echo "✅ SaladCloud environment loaded"
EOF

# Source the environment
source ~/.bashrc_salad

# Install system dependencies
echo -e "\n📦 Installing system dependencies..."
# Check if running as root
if [ "$EUID" -eq 0 ]; then
    # Running as root, no sudo needed
    apt-get update
    apt-get install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        libgomp1 \
        libopenblas-dev \
        libboost-all-dev \
        libeigen3-dev \
        swig \
        pkg-config \
        libssl-dev \
        libffi-dev \
        htop \
        tree \
        tmux \
        vim \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-dev
else
    # Not root, use sudo
    sudo apt-get update
    sudo apt-get install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        libgomp1 \
        libopenblas-dev \
        libboost-all-dev \
        libeigen3-dev \
        swig \
        pkg-config \
        libssl-dev \
        libffi-dev \
        htop \
        tree \
        tmux \
        vim \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libglib2.0-dev
fi

echo "✅ System dependencies installed"

# Setup Python virtual environment
VENV_PATH="/opt/venv"
if [ -d "$VENV_PATH" ]; then
    echo "💡 Using existing Python environment at $VENV_PATH"
    source $VENV_PATH/bin/activate
else
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
fi

# Upgrade pip and essential tools
echo -e "\n🔧 Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Clone repository if not already present
echo -e "\n📥 Khmer OCR repository setup..."
if [ -d "/ocr-training-km" ]; then
    echo "✅ Repository already available at /ocr-training-km"
    cd /ocr-training-km
else
    echo "📥 Cloning repository..."
    git clone https://github.com/khopilot/ocr-training-km.git /ocr-training-km
    cd /ocr-training-km
fi

# Install Python dependencies in correct order
echo -e "\n🐍 Installing Python dependencies (fixed order)..."

# Detect Python version and choose appropriate requirements file
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Step 1: Install numpy first
echo "Installing numpy 1.26.4..."
pip install numpy==1.26.4

# Step 2: Choose requirements file based on Python version
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "Using Python 3.12 compatible requirements..."
    REQUIREMENTS_FILE="deploy/salad/requirements-py312.txt"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    echo "Using Python 3.11 compatible requirements..."
    REQUIREMENTS_FILE="deploy/salad/requirements-salad.txt"
else
    echo "⚠️  Python $PYTHON_VERSION may not be fully supported. Using default requirements..."
    REQUIREMENTS_FILE="deploy/salad/requirements-salad.txt"
fi

# Step 3: Install core dependencies
echo "Installing core dependencies from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE || {
    echo "⚠️  Some packages failed to install. Trying manual installation..."
    # Manual fallback for critical packages
    pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66 Pillow==10.4.0
    pip install fastapi==0.112.0 uvicorn[standard]==0.30.5 pydantic==2.8.2
    pip install tqdm pyyaml requests
}

# Step 3: Install PaddlePaddle GPU with no-deps to avoid numpy conflict
echo "Installing PaddlePaddle GPU..."
pip install --no-deps paddlepaddle-gpu==2.6.1

# Step 4: Install missing PaddlePaddle dependencies manually
echo "Installing PaddlePaddle dependencies..."
pip install \
    httpx==0.24.1 \
    decorator==5.1.1 \
    astor==0.8.1 \
    opt-einsum==3.3.0 \
    protobuf==3.20.3

# Step 5: Install PaddleOCR with no-deps
echo "Installing PaddleOCR..."
pip install --no-deps paddleocr==2.7.0

# Step 6: Install any missing PaddleOCR dependencies
echo "Ensuring all dependencies are installed..."
pip install \
    attrdict==2.0.1 \
    fire==0.5.0 \
    PyMuPDF==1.23.8 || echo "⚠️  PyMuPDF installation failed (Python 3.12 issue)"

# Step 7: Install project in development mode
echo -e "\n📦 Installing project in development mode..."
cd /ocr-training-km
pip install -e . --no-deps

# Build KenLM if not already built
echo -e "\n🔧 Setting up KenLM..."
if [ -d "$HOME/kenlm/build" ]; then
    echo "✅ KenLM already built at $HOME/kenlm"
else
    if [ -d "$HOME/kenlm" ]; then
        echo "KenLM directory exists but not built. Building..."
        cd $HOME/kenlm
    else
        echo "Cloning KenLM repository..."
        cd $HOME
        git clone https://github.com/kpu/kenlm.git
        cd kenlm
    fi
    
    echo "Building KenLM..."
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    echo "✅ KenLM built successfully"
fi

# Verify installation
echo -e "\n✅ Verifying installation..."
cd /ocr-training-km
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import paddle; print(f'PaddlePaddle version: {paddle.__version__}')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR imported successfully')"
python -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"

# Create startup script
echo -e "\n📝 Creating startup script..."
cat > /ocr-training-km/start.sh << 'EOF'
#!/bin/bash
source ~/.bashrc_salad
source /opt/venv/bin/activate
cd /ocr-training-km

echo "🚀 SaladCloud Khmer OCR Environment Ready"
echo "Machine ID: ${SALAD_MACHINE_ID}"
echo "Container Group: ${SALAD_CONTAINER_GROUP_ID}"
echo "Python: $(python --version)"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "Working directory: $(pwd)"

# Start training or API server based on environment variable
if [ "$SALAD_MODE" == "training" ]; then
    echo "Starting training mode..."
    bash deploy/salad/training-entrypoint-fixed.sh
elif [ "$SALAD_MODE" == "api" ]; then
    echo "Starting API server..."
    python service/app.py
else
    echo "No mode specified. Set SALAD_MODE to 'training' or 'api'"
    exec bash
fi
EOF

chmod +x /ocr-training-km/start.sh

echo -e "\n✅ Setup complete! Environment is ready for Khmer OCR training."
echo "To start: bash /ocr-training-km/start.sh"