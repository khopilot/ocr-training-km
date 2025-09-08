"""API schemas for Khmer OCR service"""

from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field


class Line(BaseModel):
    """OCR result for a single text line"""
    
    bbox: Tuple[int, int, int, int] = Field(
        description="Bounding box coordinates (x1, y1, x2, y2)"
    )
    text: str = Field(description="Recognized text")
    conf: float = Field(ge=0.0, le=1.0, description="Confidence score")


class OCRRequest(BaseModel):
    """OCR request parameters"""
    
    enable_lm: bool = Field(default=True, description="Enable language model rescoring")
    lm_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Language model weight")
    return_lines: bool = Field(default=True, description="Return line-by-line results")
    debug: bool = Field(default=False, description="Include debug information")


class OCRResponse(BaseModel):
    """OCR response with results and metadata"""
    
    text: str = Field(description="Complete recognized text")
    lines: List[Line] = Field(description="Line-by-line results with bounding boxes")
    version: Dict[str, str] = Field(description="Model versions")
    metrics: Dict[str, Any] = Field(description="Performance metrics")
    pii_report: Optional[Dict[str, Any]] = Field(
        default=None, description="PII detection report (if enabled)"
    )
    debug: Optional[Dict[str, Any]] = Field(
        default=None, description="Debug information (if requested)"
    )


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(description="Service status")
    models_loaded: bool = Field(description="Whether models are loaded")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    backend: str = Field(description="OCR backend in use (paddle/onnx/tesseract)")
    mode: str = Field(description="Service mode (demo/prod)")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")