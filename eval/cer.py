"""Character Error Rate (CER) calculation for Khmer OCR evaluation

Includes diacritic-specific metrics and detection IoU calculations.
"""

import unicodedata
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from collections import defaultdict, Counter


def levenshtein_distance(ref: str, hyp: str) -> int:
    """
    Calculate Levenshtein distance between two strings
    
    Args:
        ref: Reference (ground truth) string
        hyp: Hypothesis (predicted) string
        
    Returns:
        Edit distance
    """
    if len(ref) == 0:
        return len(hyp)
    if len(hyp) == 0:
        return len(ref)
    
    # Create distance matrix
    matrix = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
    
    # Initialize first column and row
    for i in range(len(ref) + 1):
        matrix[i][0] = i
    for j in range(len(hyp) + 1):
        matrix[0][j] = j
    
    # Fill in the rest of the matrix
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # deletion
                matrix[i][j - 1] + 1,      # insertion
                matrix[i - 1][j - 1] + cost  # substitution
            )
    
    return matrix[len(ref)][len(hyp)]


def is_khmer_diacritic(char: str) -> bool:
    """
    Check if character is a Khmer diacritic
    
    Args:
        char: Character to check
        
    Returns:
        True if diacritic
    """
    code = ord(char)
    # Khmer diacritic ranges
    return (
        (0x17B4 <= code <= 0x17D3) or  # Various signs and marks
        (code == 0x17DD) or             # KHMER SIGN ATTHACAN
        (0x17E0 <= code <= 0x17E9)      # KHMER DIGITS (sometimes treated as diacritics)
    )


def extract_diacritics(text: str) -> Tuple[str, List[str]]:
    """
    Extract diacritics from text
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (base text without diacritics, list of diacritics)
    """
    base = []
    diacritics = []
    
    for char in text:
        if is_khmer_diacritic(char):
            diacritics.append(char)
        else:
            base.append(char)
    
    return ''.join(base), diacritics


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        CER as a percentage (0-100)
    """
    if not reference:
        return 100.0 if hypothesis else 0.0
    
    distance = levenshtein_distance(reference, hypothesis)
    cer = (distance / len(reference)) * 100
    
    return min(100.0, cer)  # Cap at 100%


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        WER as a percentage (0-100)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 100.0 if hyp_words else 0.0
    
    distance = levenshtein_distance(" ".join(ref_words), " ".join(hyp_words))
    wer = (distance / len(" ".join(ref_words))) * 100
    
    return min(100.0, wer)


def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        Dictionary of metrics
    """
    return {
        "cer": calculate_cer(reference, hypothesis),
        "wer": calculate_wer(reference, hypothesis),
        "char_precision": calculate_precision(reference, hypothesis),
        "char_recall": calculate_recall(reference, hypothesis),
        "char_f1": calculate_f1(reference, hypothesis),
        "ref_length": len(reference),
        "hyp_length": len(hypothesis),
        "length_ratio": len(hypothesis) / len(reference) if reference else 0.0,
    }


def calculate_precision(reference: str, hypothesis: str) -> float:
    """Calculate character-level precision"""
    if not hypothesis:
        return 0.0
    
    ref_chars = set(enumerate(reference))
    hyp_chars = set(enumerate(hypothesis[:len(reference)]))
    
    if not hyp_chars:
        return 0.0
    
    correct = sum(1 for i, c in enumerate(hypothesis[:len(reference)]) 
                  if i < len(reference) and reference[i] == c)
    
    return (correct / len(hypothesis)) * 100


def calculate_recall(reference: str, hypothesis: str) -> float:
    """Calculate character-level recall"""
    if not reference:
        return 100.0 if not hypothesis else 0.0
    
    correct = sum(1 for i, c in enumerate(reference) 
                  if i < len(hypothesis) and hypothesis[i] == c)
    
    return (correct / len(reference)) * 100


def calculate_f1(reference: str, hypothesis: str) -> float:
    """Calculate F1 score"""
    precision = calculate_precision(reference, hypothesis)
    recall = calculate_recall(reference, hypothesis)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def evaluate_batch(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        
    Returns:
        Aggregated metrics
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    all_metrics = []
    for ref, hyp in zip(references, hypotheses):
        metrics = calculate_metrics(ref, hyp)
        all_metrics.append(metrics)
    
    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = np.mean(values)
        aggregated[f"{key}_std"] = np.std(values)
        aggregated[f"{key}_min"] = np.min(values)
        aggregated[f"{key}_max"] = np.max(values)
    
    return aggregated


def calculate_diacritic_accuracy(
    reference: str,
    hypothesis: str
) -> Dict[str, float]:
    """
    Calculate diacritic-specific accuracy metrics
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        Dictionary with diacritic metrics
    """
    ref_base, ref_diacritics = extract_diacritics(reference)
    hyp_base, hyp_diacritics = extract_diacritics(hypothesis)
    
    metrics = {
        "total_diacritics_ref": len(ref_diacritics),
        "total_diacritics_hyp": len(hyp_diacritics),
        "diacritic_precision": 0.0,
        "diacritic_recall": 0.0,
        "diacritic_f1": 0.0,
        "base_cer": calculate_cer(ref_base, hyp_base) if ref_base else 0.0,
    }
    
    if not ref_diacritics and not hyp_diacritics:
        metrics["diacritic_precision"] = 1.0
        metrics["diacritic_recall"] = 1.0
        metrics["diacritic_f1"] = 1.0
        return metrics
    
    # Count diacritic matches
    ref_counter = Counter(ref_diacritics)
    hyp_counter = Counter(hyp_diacritics)
    
    correct = sum((ref_counter & hyp_counter).values())
    
    if hyp_diacritics:
        metrics["diacritic_precision"] = correct / len(hyp_diacritics)
    
    if ref_diacritics:
        metrics["diacritic_recall"] = correct / len(ref_diacritics)
    
    if metrics["diacritic_precision"] + metrics["diacritic_recall"] > 0:
        metrics["diacritic_f1"] = (
            2 * metrics["diacritic_precision"] * metrics["diacritic_recall"] /
            (metrics["diacritic_precision"] + metrics["diacritic_recall"])
        )
    
    return metrics


def calculate_detection_iou(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate IoU metrics for detection boxes
    
    Args:
        pred_boxes: List of predicted boxes (x1, y1, x2, y2)
        gt_boxes: List of ground truth boxes
        threshold: IoU threshold for matching
        
    Returns:
        Dictionary with IoU metrics
    """
    def box_iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    metrics = {
        "num_pred_boxes": len(pred_boxes),
        "num_gt_boxes": len(gt_boxes),
        "mean_iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    
    if not pred_boxes or not gt_boxes:
        return metrics
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = box_iou(pred_box, gt_box)
    
    # Find best matches
    matched_pred = set()
    matched_gt = set()
    ious = []
    
    while True:
        # Find best remaining match
        max_iou = 0
        best_i, best_j = -1, -1
        
        for i in range(len(pred_boxes)):
            if i in matched_pred:
                continue
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    best_i, best_j = i, j
        
        if max_iou < threshold:
            break
        
        matched_pred.add(best_i)
        matched_gt.add(best_j)
        ious.append(max_iou)
    
    # Calculate metrics
    if ious:
        metrics["mean_iou"] = np.mean(ious)
    
    metrics["precision"] = len(matched_pred) / len(pred_boxes) if pred_boxes else 0
    metrics["recall"] = len(matched_gt) / len(gt_boxes) if gt_boxes else 0
    
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2 * metrics["precision"] * metrics["recall"] /
            (metrics["precision"] + metrics["recall"])
        )
    
    return metrics


def levenshtein_ops(ref: str, hyp: str) -> Tuple[int, int, int]:
    """
    Get operation counts from Levenshtein alignment
    
    Args:
        ref: Reference string
        hyp: Hypothesis string
        
    Returns:
        Tuple of (insertions, deletions, substitutions)
    """
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack to count operations
    i, j = m, n
    insertions = deletions = substitutions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    
    return insertions, deletions, substitutions


def analyze_substitution_patterns(
    reference: str,
    hypothesis: str,
    top_k: int = 10
) -> Dict[str, any]:
    """
    Analyze common substitution patterns
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        top_k: Number of top patterns to return
        
    Returns:
        Dictionary with substitution patterns
    """
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack to find substitutions
    substitutions = Counter()
    i, j = m, n
    
    while i > 0 and j > 0:
        if reference[i-1] == hypothesis[j-1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            substitutions[(reference[i-1], hypothesis[j-1])] += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    
    top_patterns = substitutions.most_common(top_k)
    
    return {
        "total_substitutions": sum(substitutions.values()),
        "unique_patterns": len(substitutions),
        "top_patterns": [
            {"from": p[0][0], "to": p[0][1], "count": p[1]}
            for p in top_patterns
        ]
    }


def analyze_character_classes(
    reference: str,
    hypothesis: str
) -> Dict[str, any]:
    """
    Analyze errors by character class
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        Dictionary with character class analysis
    """
    classes = {
        "consonants": [],
        "vowels": [],
        "diacritics": [],
        "digits": [],
        "punctuation": [],
        "spaces": [],
        "other": []
    }
    
    for char in reference:
        code = ord(char)
        if 0x1780 <= code <= 0x17A2:
            classes["consonants"].append(char)
        elif 0x17A3 <= code <= 0x17B3:
            classes["vowels"].append(char)
        elif is_khmer_diacritic(char):
            classes["diacritics"].append(char)
        elif 0x17E0 <= code <= 0x17E9:
            classes["digits"].append(char)
        elif char in "\u17d4\u17d5\u17d6\u17d8":
            classes["punctuation"].append(char)
        elif char.isspace():
            classes["spaces"].append(char)
        else:
            classes["other"].append(char)
    
    class_metrics = {}
    for class_name, chars in classes.items():
        if chars:
            class_metrics[class_name] = {
                "count": len(chars),
                "proportion": len(chars) / len(reference) if reference else 0
            }
    
    return class_metrics


def get_error_analysis(
    reference: str,
    hypothesis: str,
    include_diacritics: bool = True,
    include_patterns: bool = True
) -> Dict[str, any]:
    """
    Detailed error analysis with diacritic and pattern analysis
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        include_diacritics: Include diacritic analysis
        include_patterns: Include substitution pattern analysis
        
    Returns:
        Dictionary with comprehensive error analysis
    """
    ops = levenshtein_ops(reference, hypothesis)
    
    analysis = {
        "total_chars_ref": len(reference),
        "total_chars_hyp": len(hypothesis),
        "insertions": ops[0],
        "deletions": ops[1],
        "substitutions": ops[2],
        "cer": calculate_cer(reference, hypothesis),
        "wer": calculate_wer(reference, hypothesis)
    }
    
    if include_diacritics:
        diacritic_metrics = calculate_diacritic_accuracy(reference, hypothesis)
        analysis["diacritics"] = diacritic_metrics
    
    if include_patterns:
        patterns = analyze_substitution_patterns(reference, hypothesis)
        analysis["patterns"] = patterns
    
    analysis["char_classes"] = analyze_character_classes(reference, hypothesis)
    
    return analysis


def confusion_matrix_chars(
    references: List[str],
    hypotheses: List[str],
    charset: Optional[str] = None
) -> Dict[str, Dict[str, int]]:
    """
    Build character-level confusion matrix
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        charset: Optional character set to consider
        
    Returns:
        Confusion matrix as nested dictionary
    """
    matrix = {}
    
    for ref, hyp in zip(references, hypotheses):
        for i in range(min(len(ref), len(hyp))):
            ref_char = ref[i]
            hyp_char = hyp[i]
            
            if charset and (ref_char not in charset or hyp_char not in charset):
                continue
            
            if ref_char not in matrix:
                matrix[ref_char] = {}
            
            if hyp_char not in matrix[ref_char]:
                matrix[ref_char][hyp_char] = 0
            
            matrix[ref_char][hyp_char] += 1
    
    return matrix