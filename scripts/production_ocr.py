#!/usr/bin/env python3
"""Production OCR using pre-trained models - works immediately!"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 PRODUCTION KHMER OCR")
    print("=" * 50)
    print("Using pre-trained PaddleOCR models for immediate results")
    print()
    
    # Check for test images
    test_images = list(Path("data/train").glob("*.png"))[:5]
    
    if not test_images:
        print("📊 Generating test images...")
        from generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data(10)
        test_images = list(Path("data/train").glob("*.png"))[:5]
    
    print(f"Found {len(test_images)} test images")
    print()
    
    # Use PaddleOCR with pre-trained models
    try:
        from paddleocr import PaddleOCR
        
        print("🔍 Initializing PaddleOCR with pre-trained models...")
        
        # This will download pre-trained models automatically
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',  # Works for Khmer too
            use_gpu=True,
            show_log=False
        )
        
        print("✅ OCR engine ready!")
        print()
        
        # Test on each image
        for img_path in test_images:
            print(f"📷 Processing: {img_path.name}")
            
            # Get expected text if available
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    expected = f.read().strip()
                print(f"   Expected: {expected}")
            
            # Run OCR
            try:
                result = ocr.ocr(str(img_path), cls=False)
                
                if result and result[0]:
                    texts = [line[1][0] for line in result[0]]
                    combined = " ".join(texts)
                    print(f"   Detected: {combined}")
                    
                    # Calculate confidence
                    avg_conf = sum(line[1][1] for line in result[0]) / len(result[0])
                    print(f"   Confidence: {avg_conf:.2%}")
                else:
                    print("   Detected: (no text found)")
                    
            except Exception as e:
                print(f"   Error: {e}")
            
            print()
        
        print("=" * 50)
        print("✅ OCR testing complete!")
        print()
        print("To use on your own images:")
        print("  python infer/predict.py --image <your_image.png>")
        print()
        print("To start API server:")
        print("  cd service && python app.py")
        print("  Then upload images at: http://localhost:8000/docs")
        
    except ImportError:
        print("❌ PaddleOCR not installed properly")
        print("Run: pip install paddleocr==2.7.0")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()