"""Language model rescoring for OCR output"""

from typing import List, Optional, Set
import numpy as np


class LanguageModel:
    """Placeholder language model class"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # TODO: Load actual KenLM model
    
    def score(self, text: str) -> float:
        """Score a text sequence"""
        # Placeholder: return random score
        return np.random.uniform(-10, 0)


def rescore_with_lm(
    ctc_logprob: float,
    tokens: List[str],
    kenlm: Optional[LanguageModel],
    lam: float = 0.3,
    mu: float = 0.1,
    lexicon: Optional[Set[str]] = None
) -> float:
    """
    Rescore OCR output with language model
    
    Args:
        ctc_logprob: CTC log probability from OCR
        tokens: List of tokens (words)
        kenlm: Language model instance
        lam: Language model weight (lambda)
        mu: Lexicon penalty weight
        lexicon: Set of valid words
        
    Returns:
        Combined score
    """
    # Base CTC score
    score = ctc_logprob
    
    # Add language model score
    if kenlm is not None and tokens:
        text = " ".join(tokens)
        lm_score = kenlm.score(text)
        score += lam * lm_score
    
    # Add lexicon penalty
    if lexicon is not None and tokens:
        oov_count = sum(1 for token in tokens if token not in lexicon)
        oov_penalty = -1.0 * oov_count
        score += mu * oov_penalty
    
    return score


def beam_search_decode(
    logits: np.ndarray,
    beam_width: int = 10,
    lm: Optional[LanguageModel] = None,
    lm_weight: float = 0.3
) -> List[tuple]:
    """
    Beam search decoding with language model
    
    Args:
        logits: CTC output logits
        beam_width: Beam size
        lm: Language model
        lm_weight: LM weight
        
    Returns:
        List of (text, score) tuples
    """
    # TODO: Implement actual beam search
    # For now, return placeholder results
    placeholder_results = [
        ("សួស្តី ពិភពលោក", -2.5),
        ("សួស្ដី ពិភពលោក", -2.8),
        ("សួស្តី ពិភពលោគ", -3.2),
    ]
    return placeholder_results[:beam_width]


def rescore_nbest(
    candidates: List[tuple],
    lm: Optional[LanguageModel] = None,
    lexicon: Optional[Set[str]] = None,
    weights: Optional[dict] = None
) -> List[tuple]:
    """
    Rescore N-best list with multiple features
    
    Args:
        candidates: List of (text, score) tuples
        lm: Language model
        lexicon: Valid word set
        weights: Feature weights
        
    Returns:
        Reranked list of (text, score) tuples
    """
    if not candidates:
        return []
    
    weights = weights or {"ctc": 1.0, "lm": 0.3, "lex": 0.1}
    
    rescored = []
    for text, ctc_score in candidates:
        total_score = weights["ctc"] * ctc_score
        
        # Add LM score
        if lm is not None:
            lm_score = lm.score(text)
            total_score += weights["lm"] * lm_score
        
        # Add lexicon score
        if lexicon:
            words = text.split()
            in_vocab_ratio = sum(1 for w in words if w in lexicon) / len(words) if words else 0
            total_score += weights["lex"] * in_vocab_ratio
        
        rescored.append((text, total_score))
    
    # Sort by score (higher is better)
    rescored.sort(key=lambda x: x[1], reverse=True)
    
    return rescored


class LexiconChecker:
    """Check words against a lexicon"""
    
    def __init__(self, lexicon_path: Optional[str] = None):
        self.lexicon = set()
        if lexicon_path:
            self.load_lexicon(lexicon_path)
    
    def load_lexicon(self, path: str):
        """Load lexicon from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.lexicon = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Lexicon file not found: {path}")
    
    def check(self, word: str) -> bool:
        """Check if word is in lexicon"""
        return word in self.lexicon
    
    def coverage(self, text: str) -> float:
        """Calculate lexicon coverage for text"""
        words = text.split()
        if not words:
            return 0.0
        in_lexicon = sum(1 for w in words if w in self.lexicon)
        return in_lexicon / len(words)