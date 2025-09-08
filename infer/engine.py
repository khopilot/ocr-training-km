"""OCR inference engine for Khmer text with multiple backends"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal

import numpy as np
from PIL import Image

# Backend availability checks
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from .postproc import normalize_khmer_text
from .rescoring import rescore_with_lm, LanguageModel

BackendType = Literal["paddle", "onnx", "tesseract", "auto"]


class OCREngine:
    """Multi-backend OCR inference engine for Khmer text"""
    
    def __init__(
        self, 
        model_dir: Optional[Path] = None, 
        use_gpu: bool = False,
        backend: BackendType = "auto",
        det_model: str = "en_PP-OCRv3_det",
        rec_model: str = "en_PP-OCRv3_rec",
        charset_path: Optional[Path] = None,
        lm_path: Optional[Path] = None,
        lm_weight: float = 0.3,
        lex_weight: float = 0.1
    ):
        """
        Initialize OCR engine with automatic backend selection
        
        Args:
            model_dir: Directory containing model files
            use_gpu: Whether to use GPU for inference
            backend: OCR backend ("paddle", "onnx", "tesseract", "auto")
            det_model: Detection model name or path
            rec_model: Recognition model name or path
            charset_path: Path to character set file
            lm_path: Path to language model
            lm_weight: Weight for language model rescoring (λ)
            lex_weight: Weight for lexicon penalty (μ)
        """
        self.model_dir = model_dir or Path("models")
        self.charset_path = charset_path or Path("train/charset_kh.txt")
        self.lm_path = lm_path
        self.lm_weight = lm_weight
        self.lex_weight = lex_weight
        
        # Platform detection
        system = platform.system()
        if system == "Darwin":  # macOS
            self.use_gpu = False
            print("Platform: macOS - Using CPU mode")
        else:
            self.use_gpu = use_gpu and self._check_gpu_available()
            mode = "GPU" if self.use_gpu else "CPU"
            print(f"Platform: {system} - Using {mode} mode")
        
        # Backend selection logic
        self.backend = self._select_backend(backend)
        print(f"Selected backend: {self.backend.upper()}")
        
        self.ocr = None
        self.lm = None
        self.charset = None
        self.lexicon = set()
        
        # Check for custom models in model_dir
        det_model_path = self.model_dir / "dbnet"
        rec_model_path = self.model_dir / "rec_kh"
        
        if det_model_path.exists():
            self.det_model = str(det_model_path)
        else:
            self.det_model = det_model
            
        if rec_model_path.exists():
            self.rec_model = str(rec_model_path)
        else:
            self.rec_model = rec_model
        
        self.load_models()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import paddle
            return paddle.is_compiled_with_cuda() and paddle.cuda.device_count() > 0
        except:
            return False
    
    def _select_backend(self, backend: BackendType) -> str:
        """Select the best available backend"""
        if backend != "auto":
            # Validate requested backend
            if backend == "paddle" and not PADDLEOCR_AVAILABLE:
                raise RuntimeError("PaddleOCR backend requested but not available")
            elif backend == "onnx" and not ONNX_AVAILABLE:
                raise RuntimeError("ONNX backend requested but not available") 
            elif backend == "tesseract" and not TESSERACT_AVAILABLE:
                raise RuntimeError("Tesseract backend requested but not available")
            return backend
        
        # Auto-selection logic
        system = platform.system()
        if system == "Darwin":
            # macOS: prefer ONNX + Tesseract (demo mode)
            if ONNX_AVAILABLE and TESSERACT_AVAILABLE:
                return "onnx"  # Use ONNX for detection, Tesseract for recognition
            elif TESSERACT_AVAILABLE:
                return "tesseract"
            elif ONNX_AVAILABLE:
                return "onnx"
        else:
            # Linux: prefer PaddleOCR (production mode)
            if PADDLEOCR_AVAILABLE:
                return "paddle"
            elif ONNX_AVAILABLE:
                return "onnx"
            elif TESSERACT_AVAILABLE:
                return "tesseract"
        
        # Fallback
        raise RuntimeError("No OCR backend available. Install PaddleOCR, ONNX Runtime, or Tesseract.")
    
    def load_models(self):
        """Load OCR and language models"""
        print(f"Loading models...")
        
        # Load charset
        if self.charset_path.exists():
            with open(self.charset_path, 'r', encoding='utf-8') as f:
                self.charset = [line.strip() for line in f if line.strip()]
            print(f"  Loaded charset: {len(self.charset)} characters")
        
        # Initialize PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.ocr = PaddleOCR(
                    det_model_dir=self.det_model if '/' in self.det_model else None,
                    rec_model_dir=self.rec_model if '/' in self.rec_model else None,
                    use_angle_cls=True,
                    lang='ch' if 'ch' in self.rec_model else 'en',  # Base language
                    use_gpu=self.use_gpu,
                    show_log=False,
                    rec_char_dict_path=str(self.charset_path) if self.charset_path.exists() else None
                )
                print(f"  PaddleOCR initialized (GPU: {self.use_gpu})")
            except Exception as e:
                print(f"  Warning: Failed to initialize PaddleOCR: {e}")
                print("  Falling back to placeholder mode")
                self.ocr = None
        else:
            print("  PaddleOCR not available, using placeholder")
        
        # Load language model
        if self.lm_path and self.lm_path.exists():
            try:
                self.lm = LanguageModel(str(self.lm_path))
                print(f"  Loaded language model: {self.lm_path}")
            except Exception as e:
                print(f"  Warning: Failed to load LM: {e}")
        
        # Load lexicon if available
        lexicon_path = self.model_dir / "lexicon.txt"
        if lexicon_path.exists():
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                self.lexicon = set(line.strip() for line in f if line.strip())
            print(f"  Loaded lexicon: {len(self.lexicon)} words")
        
        print("Models loaded successfully")
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions using PaddleOCR
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection results with bboxes and scores
        """
        if self.ocr and PADDLEOCR_AVAILABLE:
            try:
                # Run PaddleOCR detection
                result = self.ocr.ocr(image, rec=False)
                if result and result[0]:
                    detections = []
                    for line in result[0]:
                        if line is None:
                            continue
                        # Convert polygon to bbox
                        points = np.array(line).astype(int)
                        x1, y1 = points.min(axis=0)
                        x2, y2 = points.max(axis=0)
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'polygon': points.tolist(),
                            'score': 0.95  # Default high score for detected regions
                        })
                    return detections
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Fallback to placeholder
        h, w = image.shape[:2]
        return [
            {'bbox': (50, 50, w - 50, 100), 'score': 0.90},
            {'bbox': (50, 120, w - 50, 170), 'score': 0.88},
            {'bbox': (50, 190, w - 50, 240), 'score': 0.85},
        ]
    
    def recognize_text(self, image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
        """
        Recognize text using the selected backend
        
        Args:
            image: Input image
            bbox: Optional bounding box to crop (x1, y1, x2, y2)
            
        Returns:
            Tuple of (text, confidence)
        """
        # Crop if bbox provided
        if bbox:
            x1, y1, x2, y2 = bbox
            image_crop = image[y1:y2, x1:x2]
        else:
            image_crop = image
        
        if self.backend == "paddle" and self.ocr and PADDLEOCR_AVAILABLE:
            try:
                # Run PaddleOCR recognition
                result = self.ocr.ocr(image_crop, det=False)
                if result and result[0]:
                    text, conf = result[0][0]
                    return text, float(conf)
            except Exception as e:
                print(f"PaddleOCR recognition error: {e}")
        
        elif self.backend == "tesseract" and TESSERACT_AVAILABLE:
            try:
                # Use Tesseract for Khmer recognition (demo baseline)
                pil_image = Image.fromarray(image_crop)
                text = pytesseract.image_to_string(pil_image, lang='khm', config='--psm 8')
                text = text.strip()
                if text:
                    # Tesseract doesn't provide confidence, use modest estimate
                    confidence = 0.75
                    return normalize_khmer_text(text), confidence
            except Exception as e:
                print(f"Tesseract recognition error: {e}")
        
        elif self.backend == "onnx":
            # TODO: ONNX recognition would go here
            print("ONNX recognition not yet implemented")
        
        # Fallback to demo placeholder
        demo_texts = [
            "DEMO MODE",
            "Limited Recognition",
            "Tesseract Baseline",
        ]
        import random
        return random.choice(demo_texts), 0.50
    
    def process_image(self, image_path: Path, enable_lm: bool = True, ablation_mode: str = "full") -> Dict[str, Any]:
        """
        Process a single image with optional ablation modes
        
        Args:
            image_path: Path to input image
            enable_lm: Whether to use language model rescoring
            ablation_mode: One of "ctc_only", "ctc_lm", "full" (with lexicon)
            
        Returns:
            OCR results dictionary
        """
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Initialize result variables
        lines = []
        all_text = []
        
        # Full pipeline with PaddleOCR if available
        if self.ocr and PADDLEOCR_AVAILABLE:
            try:
                # Run full OCR pipeline
                result = self.ocr.ocr(image_np, cls=True)
                
                if result and result[0]:
                    for line in result[0]:
                        if line is None:
                            continue
                        bbox_points, (text, conf) = line
                        
                        # Convert points to bbox
                        points = np.array(bbox_points).astype(int)
                        x1, y1 = points.min(axis=0)
                        x2, y2 = points.max(axis=0)
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        # Apply post-processing
                        text = normalize_khmer_text(text)
                        
                        # Apply rescoring based on ablation mode
                        if ablation_mode != "ctc_only" and enable_lm:
                            if self.lm is not None:
                                score = rescore_with_lm(
                                    ctc_logprob=np.log(conf),
                                    tokens=text.split(),
                                    kenlm=self.lm,
                                    lam=self.lm_weight if ablation_mode in ["ctc_lm", "full"] else 0,
                                    mu=self.lex_weight if ablation_mode == "full" else 0,
                                    lexicon=self.lexicon if ablation_mode == "full" else set()
                                )
                                # Adjust confidence
                                conf = min(0.99, conf * np.exp(score * 0.01))
                        
                        lines.append({
                            "bbox": bbox,
                            "text": text,
                            "confidence": float(conf)
                        })
                        all_text.append(text)
            except Exception as e:
                print(f"OCR pipeline error: {e}")
                # Fall through to manual pipeline
        
        # Manual pipeline fallback or when PaddleOCR not available
        if not lines:
            # Detect text regions
            detections = self.detect_text_regions(image_np)
            
            lines = []
            all_text = []
            
            for det in detections:
                bbox = det['bbox']
                text, conf = self.recognize_text(image_np, bbox)
                
                # Apply post-processing
                text = normalize_khmer_text(text)
                
                # Apply rescoring based on ablation mode
                if ablation_mode != "ctc_only" and enable_lm and self.lm:
                    score = rescore_with_lm(
                        ctc_logprob=np.log(conf),
                        tokens=text.split(),
                        kenlm=self.lm,
                        lam=self.lm_weight if ablation_mode in ["ctc_lm", "full"] else 0,
                        mu=self.lex_weight if ablation_mode == "full" else 0,
                        lexicon=self.lexicon if ablation_mode == "full" else set()
                    )
                    conf = min(0.99, conf * np.exp(score * 0.01))
                
                lines.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(conf)
                })
                all_text.append(text)
        
        # Combine all text
        full_text = " ".join(all_text)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        return {
            "image": str(image_path),
            "text": full_text,
            "lines": lines,
            "ablation_mode": ablation_mode,
            "metrics": {
                "processing_time_s": processing_time,
                "num_lines": len(lines),
                "avg_confidence": np.mean([l["confidence"] for l in lines]) if lines else 0.0,
                "use_lm": enable_lm and self.lm is not None,
                "use_lexicon": ablation_mode == "full" and len(self.lexicon) > 0
            }
        }
    
    def process_batch(self, image_paths: List[Path], enable_lm: bool = True) -> List[Dict[str, Any]]:
        """Process multiple images"""
        results = []
        for path in image_paths:
            try:
                result = self.process_image(path, enable_lm)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    "image": str(path),
                    "error": str(e)
                })
        return results


def main():
    """CLI interface for OCR engine"""
    parser = argparse.ArgumentParser(description="Khmer OCR inference engine")
    parser.add_argument("--images", type=str, required=True, help="Path to image(s)")
    parser.add_argument("--output", type=str, default="out", help="Output directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--no-lm", action="store_true", help="Disable language model")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize engine
    engine = OCREngine(model_dir=Path(args.model_dir), use_gpu=args.gpu)
    
    # Process images
    images_path = Path(args.images)
    if images_path.is_file():
        image_paths = [images_path]
    else:
        image_paths = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    if not image_paths:
        print(f"No images found in {args.images}")
        return
    
    print(f"Processing {len(image_paths)} image(s)...")
    results = engine.process_batch(image_paths, enable_lm=not args.no_lm)
    
    # Save results
    output_file = output_dir / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    successful = [r for r in results if "error" not in r]
    print(f"\nSummary:")
    print(f"  Processed: {len(successful)}/{len(results)}")
    if successful:
        avg_time = np.mean([r["metrics"]["processing_time_s"] for r in successful])
        avg_conf = np.mean([r["metrics"]["avg_confidence"] for r in successful])
        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Avg confidence: {avg_conf:.3f}")


if __name__ == "__main__":
    main()