
import time
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    
    ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True)
    
    # Find test images
    test_imgs = list(Path("data/train").glob("*.png"))[:10]
    
    if test_imgs:
        times = []
        for img in test_imgs:
            start = time.time()
            result = ocr.ocr(str(img), cls=False)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        print(f"Average OCR time: {avg_time:.3f}s per image")
        print(f"Throughput: {1/avg_time:.1f} images/second")
    else:
        print("No test images for benchmark")
        
except Exception as e:
    print(f"Benchmark failed: {e}")
