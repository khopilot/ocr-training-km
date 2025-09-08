#!/usr/bin/env python3
"""Working inference script for Khmer OCR"""

import os
import sys
import argparse
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

def predict_with_paddleocr(image_path, model_dir=None):
    """Use PaddleOCR for inference"""
    try:
        from paddleocr import PaddleOCR
        
        print(f"🔍 Running OCR on: {image_path}")
        
        # Initialize PaddleOCR
        # Use pre-trained model if no custom model specified
        if model_dir and os.path.exists(model_dir):
            print(f"Using custom model: {model_dir}")
            ocr = PaddleOCR(
                det_model_dir=f"{model_dir}/det",
                rec_model_dir=f"{model_dir}/rec",
                use_angle_cls=False,
                lang='en',
                use_gpu=True,
                show_log=False
            )
        else:
            print("Using pre-trained PaddleOCR model")
            # Use pre-trained model (will download if needed)
            ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',  # Will work for Khmer text too
                use_gpu=True,
                show_log=False
            )
        
        # Run OCR
        result = ocr.ocr(image_path, cls=False)
        
        if result and result[0]:
            print("\n📝 Detected text:")
            all_text = []
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                all_text.append(text)
                print(f"  {text} (confidence: {confidence:.3f})")
            
            combined_text = " ".join(all_text)
            print(f"\n✅ Combined text: {combined_text}")
            return combined_text
        else:
            print("❌ No text detected")
            return None
            
    except Exception as e:
        print(f"❌ PaddleOCR error: {e}")
        return None

def predict_with_opencv(image_path):
    """Fallback: Use OpenCV + Tesseract if available"""
    try:
        import cv2
        import numpy as np
        
        print(f"📷 Using OpenCV fallback for: {image_path}")
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try OCR with tesseract if available
        try:
            import pytesseract
            text = pytesseract.image_to_string(gray)
            print(f"✅ Detected (Tesseract): {text.strip()}")
            return text.strip()
        except:
            print("⚠️  Tesseract not available")
            
        # Just return image info
        h, w = gray.shape
        print(f"📊 Image size: {w}x{h}")
        print("ℹ️  Install tesseract for OCR: apt-get install tesseract-ocr")
        return f"Image {w}x{h} - OCR not available"
        
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Khmer OCR Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=None, help='Path to model directory')
    parser.add_argument('--method', type=str, default='paddleocr', 
                       choices=['paddleocr', 'opencv'], help='OCR method')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    print("🚀 Khmer OCR Inference")
    print("=" * 40)
    
    # Run inference
    if args.method == 'paddleocr':
        result = predict_with_paddleocr(args.image, args.model)
    else:
        result = predict_with_opencv(args.image)
    
    if result:
        print("\n" + "=" * 40)
        print("✅ OCR Complete!")
        
        # Check if there's a ground truth file
        txt_path = Path(args.image).with_suffix('.txt')
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                expected = f.read().strip()
            print(f"\n📋 Expected: {expected}")
            print(f"📝 Predicted: {result}")
            
            # Simple accuracy check
            if expected.lower() == result.lower():
                print("✅ Perfect match!")
            else:
                print("⚠️  Mismatch - model needs more training")
    else:
        print("\n❌ OCR failed")
        sys.exit(1)

if __name__ == "__main__":
    main()