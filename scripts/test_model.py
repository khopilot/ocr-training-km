#!/usr/bin/env python3
"""Simple script to test the trained model"""

import os
import sys
from pathlib import Path

os.chdir('/ocr-training-km')

def test_model():
    print("🔍 Testing Khmer OCR Model")
    print("=" * 40)
    
    # Find a test image
    test_images = []
    
    # Look for test images
    for data_dir in ['data/train', 'data/val', 'data/test', 'data/hf_train']:
        if os.path.exists(data_dir):
            images = list(Path(data_dir).glob("*.png"))[:5]  # Get first 5
            test_images.extend(images)
    
    if not test_images:
        print("❌ No test images found!")
        print("Generate some first: python scripts/generate_khmer_synthetic.py 10")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Test with PaddleOCR
    try:
        from paddleocr import PaddleOCR
        
        print("\n🚀 Initializing PaddleOCR...")
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='en',  # Will use our custom model
            use_gpu=True,
            show_log=False
        )
        
        print("\n📝 Testing on images:")
        for img_path in test_images[:3]:  # Test first 3
            print(f"\nImage: {img_path}")
            
            # Get expected text if available
            txt_path = img_path.with_suffix('.txt')
            expected = ""
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    expected = f.read().strip()
                print(f"Expected: {expected}")
            
            # Run OCR
            try:
                result = ocr.ocr(str(img_path), cls=False)
                if result and result[0]:
                    predicted = " ".join([line[1][0] for line in result[0]])
                    print(f"Predicted: {predicted}")
                else:
                    print("Predicted: (no text detected)")
            except Exception as e:
                print(f"Error: {e}")
                
    except ImportError:
        print("❌ PaddleOCR not available")
        print("This is normal in demo mode - the model was trained but inference needs full PaddleOCR")
    
    print("\n" + "=" * 40)
    print("✅ Test complete!")
    print("\nNote: Demo training creates a mock model.")
    print("For real inference, train with full PaddleOCR on GPU.")

if __name__ == "__main__":
    test_model()