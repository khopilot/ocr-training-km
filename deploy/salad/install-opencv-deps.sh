#!/bin/bash
# Quick script to install OpenCV dependencies for headless environments

echo "📦 Installing OpenCV dependencies for headless environment..."

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    # Running as root, no sudo needed
    apt-get update
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
else
    # Not root, use sudo
    sudo apt-get update
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

echo "✅ OpenCV dependencies installed"

# Test OpenCV import
echo "Testing OpenCV import..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" && echo "✅ OpenCV working!" || echo "❌ OpenCV still has issues"