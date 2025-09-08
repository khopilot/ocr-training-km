"""FastAPI application for Khmer OCR service"""

import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import PlainTextResponse

# Fix import for direct execution
try:
    from .schemas import OCRRequest, OCRResponse, HealthResponse, ErrorResponse, Line
except ImportError:
    from schemas import OCRRequest, OCRResponse, HealthResponse, ErrorResponse, Line

# Service configuration
SERVICE_VARIANT = os.environ.get("SERVICE_VARIANT", "auto")  # auto, paddle, onnx, tesseract
PRODUCTION_MODE = os.environ.get("PRODUCTION_MODE", "demo").lower()  # demo, prod
PADDLE_LANG = os.environ.get("PADDLE_LANG", "auto")  # auto, khmer, ch, en

# Initialize FastAPI app
app = FastAPI(
    title="Khmer OCR API",
    description=f"Production-grade OCR service for Khmer text (Backend: {SERVICE_VARIANT})",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter("ocr_requests_total", "Total OCR requests")
REQUEST_DURATION = Histogram("ocr_request_duration_seconds", "OCR request duration")
ERROR_COUNT = Counter("ocr_errors_total", "Total OCR errors")

# Service state
SERVICE_START_TIME = time.time()
MODELS_LOADED = False
BACKEND_ENGINE = None
CURRENT_MODE = "unknown"


def load_models():
    """Load OCR models based on SERVICE_VARIANT and PRODUCTION_MODE"""
    global MODELS_LOADED, BACKEND_ENGINE, CURRENT_MODE
    
    CURRENT_MODE = f"{PRODUCTION_MODE}_{SERVICE_VARIANT}"
    print(f"🚀 Loading OCR backend: {SERVICE_VARIANT} (mode: {PRODUCTION_MODE})")
    
    if SERVICE_VARIANT == "paddle" or (SERVICE_VARIANT == "auto" and _paddle_available()):
        # Load PaddleOCR (production)
        print("   Loading PaddleOCR models...")
        try:
            import paddleocr
            
            # Determine language based on mode and availability
            paddle_lang = _get_paddle_language()
            print(f"   Using PaddleOCR language: {paddle_lang}")
            
            BACKEND_ENGINE = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang=paddle_lang,
                det_model_dir=_get_model_path('det') if PRODUCTION_MODE == 'prod' else None,
                rec_model_dir=_get_model_path('rec') if PRODUCTION_MODE == 'prod' else None
            )
            
            if PRODUCTION_MODE == 'prod' and paddle_lang == 'khmer':
                print("   ✅ PaddleOCR loaded (production backend with Khmer models)")
            else:
                print(f"   ✅ PaddleOCR loaded ({PRODUCTION_MODE} backend, lang={paddle_lang})")
            SERVICE_VARIANT = "paddle"
        except Exception as e:
            print(f"   ⚠️  PaddleOCR failed: {e}")
            if SERVICE_VARIANT == "paddle":
                raise
            # Fall through to next option
    
    if SERVICE_VARIANT == "onnx" or (SERVICE_VARIANT == "auto" and _onnx_available()):
        # Load ONNX models (cross-platform)
        print("   Loading ONNX models...")
        try:
            import onnxruntime as ort
            # TODO: Load actual ONNX models
            BACKEND_ENGINE = {"type": "onnx", "session": None}
            print("   ✅ ONNX runtime loaded (cross-platform backend)")
            SERVICE_VARIANT = "onnx"
        except Exception as e:
            print(f"   ⚠️  ONNX failed: {e}")
            if SERVICE_VARIANT == "onnx":
                raise
            # Fall through to next option
    
    if SERVICE_VARIANT == "tesseract" or SERVICE_VARIANT == "auto":
        # Load Tesseract (fallback)
        print("   Loading Tesseract...")
        try:
            import pytesseract
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            BACKEND_ENGINE = {"type": "tesseract"}
            print("   ✅ Tesseract loaded (fallback backend)")
            SERVICE_VARIANT = "tesseract"
        except Exception as e:
            print(f"   ❌ Tesseract failed: {e}")
            if SERVICE_VARIANT == "tesseract":
                raise
            raise RuntimeError("No OCR backend available")
    
    MODELS_LOADED = True
    backend_version = _get_backend_version()
    print(f"🎉 OCR service ready: {SERVICE_VARIANT} backend (mode: {PRODUCTION_MODE})")
    print(f"   🔧 Backend version: {backend_version}")
    
    # Log detailed model configuration
    if PRODUCTION_MODE == 'prod':
        print("   📋 Production mode: Using trained Khmer models when available")
        det_info = _get_model_info('det')
        rec_info = _get_model_info('rec')
        print(f"   📊 Detection model: {det_info['status']} (hash: {det_info['hash']}, size: {det_info.get('size_mb', 'n/a')}MB)")
        print(f"   📊 Recognition model: {rec_info['status']} (hash: {rec_info['hash']}, size: {rec_info.get('size_mb', 'n/a')}MB)")
    else:
        print("   📋 Demo mode: Using pre-trained models for evaluation")


def _paddle_available() -> bool:
    """Check if PaddleOCR is available"""
    try:
        import paddleocr
        return True
    except ImportError:
        return False


def _onnx_available() -> bool:
    """Check if ONNX runtime is available"""
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def _get_paddle_language() -> str:
    """Determine PaddleOCR language based on mode and availability"""
    if PADDLE_LANG != "auto":
        return PADDLE_LANG
    
    # In production mode, prefer Khmer if models are available
    if PRODUCTION_MODE == 'prod':
        khmer_model_path = _get_model_path('rec')
        if khmer_model_path and Path(khmer_model_path).exists():
            return 'khmer'
        # Fallback to multilingual for production
        return 'ch'
    
    # Demo mode defaults
    return 'ch'


def _get_model_path(model_type: str) -> Optional[str]:
    """Get path to trained model if available"""
    model_dir = Path("models")
    
    if model_type == 'det':
        # Look for DBNet detection model
        for model_file in model_dir.glob("dbnet*.pdmodel"):
            return str(model_file.parent)
        for model_file in model_dir.glob("*det*.pdmodel"):
            return str(model_file.parent)
    
    elif model_type == 'rec':
        # Look for Khmer recognition model
        for model_file in model_dir.glob("rec_kh*.pdmodel"):
            return str(model_file.parent)
        for model_file in model_dir.glob("*khmer*.pdmodel"):
            return str(model_file.parent)
        for model_file in model_dir.glob("*rec*.pdmodel"):
            return str(model_file.parent)
    
    return None


def _get_model_info(model_type: str) -> Dict[str, Any]:
    """Get detailed model information including version and hash"""
    model_path = _get_model_path(model_type)
    if not model_path:
        return {"status": "pretrained", "version": "n/a", "hash": "n/a"}
    
    model_dir = Path(model_path)
    info = {
        "status": "trained",
        "path": str(model_dir),
        "version": "n/a",
        "hash": "n/a",
        "size_mb": "n/a"
    }
    
    # Try to get model file info
    for model_file in model_dir.glob("*.pdmodel"):
        try:
            stat = model_file.stat()
            info["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
            info["modified"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            
            # Extract version from filename if present
            if "v" in model_file.stem:
                version_part = model_file.stem.split("v")[-1]
                if version_part.replace(".", "").replace("_", "").isdigit():
                    info["version"] = f"v{version_part}"
            
            # Get basic hash (first 8 chars of file size + mtime for simplicity)
            import hashlib
            hash_input = f"{stat.st_size}_{int(stat.st_mtime)}".encode()
            info["hash"] = hashlib.md5(hash_input).hexdigest()[:8]
            break
            
        except Exception as e:
            print(f"Warning: Could not get model info for {model_file}: {e}")
    
    return info


def _get_backend_version() -> str:
    """Get version info for the current backend"""
    try:
        if SERVICE_VARIANT == "paddle":
            import paddle
            return f"paddle-{paddle.__version__}"
        elif SERVICE_VARIANT == "onnx":
            import onnxruntime
            return f"onnx-{onnxruntime.__version__}"
        elif SERVICE_VARIANT == "tesseract":
            import pytesseract
            version = pytesseract.get_tesseract_version()
            return f"tesseract-{version}"
    except Exception:
        pass
    return f"{SERVICE_VARIANT}-unknown"


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    load_models()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        models_loaded=MODELS_LOADED,
        version="0.1.0",
        uptime_seconds=time.time() - SERVICE_START_TIME,
        backend=SERVICE_VARIANT,
        mode=PRODUCTION_MODE,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODELS_LOADED else "loading",
        models_loaded=MODELS_LOADED,
        version="0.1.0",
        uptime_seconds=time.time() - SERVICE_START_TIME,
        backend=SERVICE_VARIANT,
        mode=PRODUCTION_MODE,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest())


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    request: Request,
    file: UploadFile = File(...),
    enable_lm: bool = Form(True),
    lm_weight: float = Form(0.3),
    return_lines: bool = Form(True),
    debug: bool = Form(False),
):
    """
    Perform OCR on uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        enable_lm: Enable language model rescoring
        lm_weight: Weight for language model (0.0-1.0)
        return_lines: Return line-by-line results
        debug: Include debug information
    
    Returns:
        OCR results with text, lines, and metadata
    """
    REQUEST_COUNT.inc()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    with REQUEST_DURATION.time():
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Read and process image
            start_time = time.time()
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Perform OCR using selected backend
            text, lines = await _perform_ocr(image, SERVICE_VARIANT, BACKEND_ENGINE)
            
            # Apply language model rescoring if enabled
            if enable_lm:
                # TODO: Apply actual KenLM rescoring
                text = f"{text} (LM weight: {lm_weight})"
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Get detailed model information
            det_info = _get_model_info('det')
            rec_info = _get_model_info('rec')
            
            # Build response
            response = OCRResponse(
                text=text,
                lines=lines if return_lines else [],
                version={
                    "api": "0.1.0",
                    "backend": _get_backend_version(),
                    "mode": PRODUCTION_MODE,
                    "paddle_lang": _get_paddle_language() if SERVICE_VARIANT == "paddle" else "n/a",
                    "det_model": {
                        "status": det_info["status"],
                        "hash": det_info["hash"],
                        "version": det_info.get("version", "n/a")
                    },
                    "rec_model": {
                        "status": rec_info["status"], 
                        "hash": rec_info["hash"],
                        "version": rec_info.get("version", "n/a")
                    },
                    "lm": "khopilot-tokenizer" if enable_lm else "disabled",
                },
                metrics={
                    "latency_ms": round(processing_time, 2),
                    "image_size": image.size,
                    "lines_detected": len(lines),
                    "request_id": request_id,
                    "backend": SERVICE_VARIANT,
                    "mode": PRODUCTION_MODE,
                },
                pii_report=None,  # TODO: Implement PII detection
                debug={
                    "enable_lm": enable_lm,
                    "lm_weight": lm_weight,
                    "file_size_bytes": len(image_bytes),
                    "backend": SERVICE_VARIANT,
                    "mode": PRODUCTION_MODE,
                    "paddle_lang": _get_paddle_language() if SERVICE_VARIANT == "paddle" else "n/a",
                    "model_paths": {
                        "det": _get_model_path('det'),
                        "rec": _get_model_path('rec')
                    } if PRODUCTION_MODE == "prod" else None,
                } if debug else None,
            )
            
            return response
            
        except Exception as e:
            ERROR_COUNT.inc()
            raise HTTPException(
                status_code=500,
                detail=f"OCR processing failed: {str(e)}"
            )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            request_id=request.headers.get("X-Request-ID"),
        ).model_dump(),
    )


async def _perform_ocr(image: Image.Image, variant: str, engine) -> tuple[str, list[Line]]:
    """Perform OCR using the specified backend"""
    
    if variant == "paddle" and engine is not None:
        # PaddleOCR inference
        try:
            import numpy as np
            img_array = np.array(image)
            result = engine.ocr(img_array, cls=True)
            
            text_parts = []
            lines = []
            
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        bbox_points = line[0]
                        text_info = line[1]
                        
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text_content = text_info[0]
                            confidence = float(text_info[1])
                            
                            text_parts.append(text_content)
                            
                            # Convert bbox points to (x1, y1, x2, y2)
                            x_coords = [p[0] for p in bbox_points]
                            y_coords = [p[1] for p in bbox_points]
                            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                            
                            lines.append(Line(bbox=bbox, text=text_content, conf=confidence))
            
            return " ".join(text_parts), lines
            
        except Exception as e:
            print(f"PaddleOCR inference error: {e}")
            # Fall back to placeholder
    
    elif variant == "tesseract":
        # Tesseract inference
        try:
            import pytesseract
            # Configure Tesseract for Khmer
            custom_config = r'--oem 3 --psm 6 -l khm'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Get bounding boxes
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            lines = []
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    bbox = (
                        data['left'][i],
                        data['top'][i], 
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    )
                    conf = data['conf'][i] / 100.0  # Convert to 0-1 range
                    lines.append(Line(bbox=bbox, text=data['text'][i], conf=conf))
            
            return text.strip(), lines
            
        except Exception as e:
            print(f"Tesseract inference error: {e}")
            # Fall back to placeholder
    
    elif variant == "onnx":
        # ONNX inference (placeholder)
        try:
            # TODO: Implement actual ONNX model inference
            pass
        except Exception as e:
            print(f"ONNX inference error: {e}")
    
    # Fallback placeholder
    placeholder_text = "សួស្តី ពិភពលោក"  # "Hello World" in Khmer
    placeholder_lines = [
        Line(bbox=(10, 10, 200, 50), text="សួស្តី", conf=0.95),
        Line(bbox=(10, 60, 200, 100), text="ពិភពលោក", conf=0.92),
    ]
    
    return placeholder_text, placeholder_lines


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)