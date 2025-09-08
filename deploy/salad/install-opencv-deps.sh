#!/bin/bash
# Quick script to install OpenCV dependencies for headless environments

echo "📦 Installing OpenCV dependencies for headless environment..."

# Detect Ubuntu version
if [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    UBUNTU_VERSION="${DISTRIB_RELEASE}"
    echo "Ubuntu version: $UBUNTU_VERSION"
else
    UBUNTU_VERSION="unknown"
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    # Running as root, no sudo needed
    apt-get update
    
    # For Ubuntu 24.04, package names have changed
    if [[ "$UBUNTU_VERSION" == "24.04" ]]; then
        echo "Installing packages for Ubuntu 24.04..."
        apt-get install -y \
            libgl1 \
            libgl1-mesa-dri \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender1 \
            libgomp1 \
            libglib2.0-dev \
            ffmpeg \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev
    else
        # For older Ubuntu versions
        apt-get install -y \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libglib2.0-dev \
            libgtk2.0-dev \
            pkg-config
    fi
else
    # Not root, use sudo
    sudo apt-get update
    
    if [[ "$UBUNTU_VERSION" == "24.04" ]]; then
        echo "Installing packages for Ubuntu 24.04..."
        sudo apt-get install -y \
            libgl1 \
            libgl1-mesa-dri \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender1 \
            libgomp1 \
            libglib2.0-dev \
            ffmpeg \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev
    else
        sudo apt-get install -y \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libglib2.0-dev \
            libgtk2.0-dev \
            pkg-config
    fi
fi

echo "✅ OpenCV dependencies installed"

# Test OpenCV import
echo "Testing OpenCV import..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" && echo "✅ OpenCV working!" || echo "❌ OpenCV still has issues"