"""Evaluation package for Khmer OCR"""

from .cer import (
    calculate_cer,
    calculate_wer,
    calculate_metrics,
    evaluate_batch,
    get_error_analysis,
)
from .harness import EvaluationHarness

__all__ = [
    "calculate_cer",
    "calculate_wer",
    "calculate_metrics",
    "evaluate_batch",
    "get_error_analysis",
    "EvaluationHarness",
]