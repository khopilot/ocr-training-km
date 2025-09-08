#!/usr/bin/env python3
"""Khmer tokenizer wrapper with char-to-subword alignment

Loads khopilot/km-tokenizer-khmer (8000 vocab SentencePiece unigram) via transformers
when available, with fallback to local SentencePiece model. Handles alignment between
CTC character sequences and KenLM subword tokens.
"""

import os
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

# Try to import transformers (optional dependency)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, will use local SentencePiece model")

# SentencePiece is required
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    raise ImportError("SentencePiece is required for tokenization")


class KhmerTokenizer:
    """Unified Khmer tokenizer with char/subword alignment"""
    
    def __init__(
        self,
        tokenizer_id: Optional[str] = None,
        tokenizer_path: Optional[Path] = None,
        vocab_size: int = 8000
    ):
        """
        Initialize Khmer tokenizer
        
        Args:
            tokenizer_id: HuggingFace model ID (default: khopilot/km-tokenizer-khmer)
            tokenizer_path: Local SentencePiece model path (fallback)
            vocab_size: Vocabulary size (for training new model if needed)
        """
        self.tokenizer_id = tokenizer_id or os.environ.get(
            "TOKENIZER_ID", "khopilot/km-tokenizer-khmer"
        )
        self.tokenizer_path = tokenizer_path or os.environ.get("TOKENIZER_PATH")
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.sp = None
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Khmer Unicode ranges for validation
        self.khmer_ranges = [
            (0x1780, 0x17FF),  # Khmer block
            (0x19E0, 0x19FF),  # Khmer Symbols
        ]
    
    def _load_tokenizer(self):
        """Load tokenizer with fallback strategy"""
        # Try HuggingFace transformers first
        if TRANSFORMERS_AVAILABLE and self.tokenizer_id:
            try:
                print(f"Loading tokenizer from HuggingFace: {self.tokenizer_id}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_id,
                    use_fast=False  # Use slow tokenizer for better control
                )
                print(f"✓ Loaded {self.tokenizer_id} (vocab size: {len(self.tokenizer)})")
                return
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
        
        # Fallback to local SentencePiece model
        if self.tokenizer_path and Path(self.tokenizer_path).exists():
            print(f"Loading local SentencePiece model: {self.tokenizer_path}")
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(str(self.tokenizer_path))
            print(f"✓ Loaded local model (vocab size: {self.sp.GetPieceSize()})")
            return
        
        # Last resort: train minimal model
        print("Warning: No tokenizer available, using character-level fallback")
        self._create_char_tokenizer()
    
    def _create_char_tokenizer(self):
        """Create character-level tokenizer as fallback"""
        # This is a minimal fallback - just split by characters
        self.char_mode = True
    
    def normalize_khmer(self, text: str) -> str:
        """
        Normalize Khmer text for consistent tokenization
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # 1. Unicode NFC normalization
        text = unicodedata.normalize('NFC', text)
        
        # 2. Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        
        # 3. Reorder diacritics (simplified - full implementation would be more complex)
        # Khmer diacritics should follow specific ordering rules
        # This is a placeholder for proper Khmer normalization
        
        # 4. Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str, return_offsets: bool = False) -> List[str]:
        """
        Tokenize text into subwords
        
        Args:
            text: Input text
            return_offsets: Whether to return character offsets
            
        Returns:
            List of tokens (and optionally offsets)
        """
        # Normalize first
        text = self.normalize_khmer(text)
        
        if self.tokenizer:
            # Use HuggingFace tokenizer
            tokens = self.tokenizer.tokenize(text)
            if return_offsets:
                # Get offsets for alignment
                encoding = self.tokenizer(text, return_offsets_mapping=True)
                return tokens, encoding.offset_mapping
            return tokens
        
        elif self.sp:
            # Use SentencePiece
            tokens = self.sp.EncodeAsPieces(text)
            if return_offsets:
                # Calculate offsets manually
                offsets = []
                pos = 0
                for token in tokens:
                    # Remove SentencePiece markers
                    clean_token = token.replace('▁', '')
                    if clean_token:
                        start = text.find(clean_token, pos)
                        if start != -1:
                            end = start + len(clean_token)
                            offsets.append((start, end))
                            pos = end
                        else:
                            offsets.append((pos, pos))
                    else:
                        offsets.append((pos, pos))
                return tokens, offsets
            return tokens
        
        else:
            # Character-level fallback
            tokens = list(text)
            if return_offsets:
                offsets = [(i, i+1) for i in range(len(text))]
                return tokens, offsets
            return tokens
    
    def align_ctc_to_subwords(
        self,
        ctc_chars: List[str],
        ctc_scores: Optional[np.ndarray] = None
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Align CTC character output to subword tokens
        
        Args:
            ctc_chars: List of characters from CTC decoder
            ctc_scores: Optional character-level confidence scores
            
        Returns:
            Tuple of (subword tokens, aggregated scores)
        """
        # Join characters to form text
        text = ''.join(ctc_chars)
        
        # Tokenize into subwords with offsets
        tokens, offsets = self.tokenize(text, return_offsets=True)
        
        # Aggregate scores if provided
        if ctc_scores is not None:
            token_scores = []
            for start, end in offsets:
                if start < len(ctc_scores) and end <= len(ctc_scores):
                    # Average score for this token
                    token_scores.append(np.mean(ctc_scores[start:end]))
                else:
                    token_scores.append(0.0)
            return tokens, np.array(token_scores)
        
        return tokens, None
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.tokenizer:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        elif self.sp:
            return self.sp.DecodeIds(token_ids)
        else:
            # Character-level fallback
            return ''.join([chr(tid) for tid in token_ids if tid < 0x110000])
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.tokenizer:
            return len(self.tokenizer)
        elif self.sp:
            return self.sp.GetPieceSize()
        else:
            return 256  # Basic ASCII + extended
    
    def score_alignment(
        self,
        text: str,
        lm_score: float,
        ctc_score: float,
        scale_factor: float = 2.3026
    ) -> float:
        """
        Combine LM and CTC scores with proper scaling
        
        Args:
            text: Text to score
            lm_score: Language model score (log10)
            ctc_score: CTC score (ln)
            scale_factor: Conversion factor from log10 to ln (default: ln(10))
            
        Returns:
            Combined score
        """
        # Convert log10 to ln for consistent scale
        lm_score_ln = lm_score * scale_factor
        
        # Combine scores (weighted average)
        # This is simplified - production would use learned weights
        combined = 0.7 * ctc_score + 0.3 * lm_score_ln
        
        return combined


def create_tokenizer_from_corpus(
    corpus_path: Path,
    output_path: Path,
    vocab_size: int = 8000
) -> KhmerTokenizer:
    """
    Train a new SentencePiece tokenizer from corpus
    
    Args:
        corpus_path: Path to training corpus
        output_path: Path to save model
        vocab_size: Vocabulary size
        
    Returns:
        Trained tokenizer
    """
    print(f"Training SentencePiece model from {corpus_path}")
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(output_path),
        vocab_size=vocab_size,
        model_type='unigram',  # Best for Khmer
        character_coverage=0.9995,
        num_threads=4,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<num>', '<url>', '<email>', '<phone>']
    )
    
    model_path = Path(f"{output_path}.model")
    return KhmerTokenizer(tokenizer_path=model_path)


def main():
    """CLI for testing tokenizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Khmer tokenizer utility")
    parser.add_argument("--tokenizer-id", default="khopilot/km-tokenizer-khmer")
    parser.add_argument("--tokenizer-path", type=Path, help="Local model path")
    parser.add_argument("--text", default="សួស្តី ពិភពលោក")
    parser.add_argument("--train", type=Path, help="Train from corpus")
    parser.add_argument("--output", type=Path, help="Output model path")
    
    args = parser.parse_args()
    
    if args.train:
        # Train new model
        tokenizer = create_tokenizer_from_corpus(
            args.train,
            args.output or Path("khmer_tokenizer"),
            vocab_size=8000
        )
    else:
        # Load existing model
        tokenizer = KhmerTokenizer(
            tokenizer_id=args.tokenizer_id,
            tokenizer_path=args.tokenizer_path
        )
    
    # Test tokenization
    print(f"\nText: {args.text}")
    normalized = tokenizer.normalize_khmer(args.text)
    print(f"Normalized: {normalized}")
    
    tokens = tokenizer.tokenize(normalized)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # Test alignment
    ctc_chars = list(normalized)
    aligned_tokens, _ = tokenizer.align_ctc_to_subwords(ctc_chars)
    print(f"Aligned tokens: {aligned_tokens}")
    
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()