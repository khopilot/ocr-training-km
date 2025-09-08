#!/bin/bash
# Alternative fix for OpenCV in headless environment using opencv-python-headless

echo "🔧 Fixing OpenCV for headless environment..."

# Uninstall regular opencv-python and install headless version
echo "Switching to opencv-python-headless..."
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless==4.6.0.66

# Install minimal required libraries for Ubuntu 24.04
echo "Installing minimal required libraries..."
if [ "$EUID" -eq 0 ]; then
    apt-get update
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1
else
    sudo apt-get update
    sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1
fi

# Test import
echo "Testing OpenCV import..."
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} working!')" || {
    echo "❌ OpenCV still has issues. Trying alternative fix..."
    
    # Alternative: Install libgl1 specifically
    if [ "$EUID" -eq 0 ]; then
        apt-get install -y libgl1
    else
        sudo apt-get install -y libgl1
    fi
    
    # Create symlink if needed
    if [ ! -f /usr/lib/x86_64-linux-gnu/libGL.so.1 ]; then
        echo "Creating libGL.so.1 symlink..."
        if [ -f /usr/lib/x86_64-linux-gnu/libGL.so ]; then
            ln -s /usr/lib/x86_64-linux-gnu/libGL.so /usr/lib/x86_64-linux-gnu/libGL.so.1
        fi
    fi
    
    # Final test
    python -c "import cv2; print(f'✅ OpenCV {cv2.__version__} working!')" || echo "❌ Manual intervention may be needed"
}

echo "Done!"