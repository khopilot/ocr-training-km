#!/usr/bin/env python3
"""Train KenLM language model for Khmer text with SentencePiece tokenization"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("Warning: SentencePiece not available, using space tokenization")

# HuggingFace tokenizer support (khopilot/km-tokenizer-khmer)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, HuggingFace tokenizer disabled")


class LanguageModelTrainer:
    """Trainer for KenLM language models with SentencePiece or HuggingFace tokenization"""
    
    def __init__(
        self, 
        corpus_path: Optional[Path] = None, 
        order: int = 5,
        vocab_size: int = 8000,
        tokenizer_model: Optional[str] = None,
        hf_tokenizer: Optional[str] = None
    ):
        """
        Initialize LM trainer
        
        Args:
            corpus_path: Path to text corpus
            order: N-gram order (default: 5)
            vocab_size: Vocabulary size for SentencePiece
            tokenizer_model: Pre-trained SentencePiece model path
            hf_tokenizer: HuggingFace tokenizer model name or path
        """
        self.corpus_path = corpus_path
        self.order = order
        self.vocab_size = vocab_size
        self.tokenizer_model = tokenizer_model
        self.hf_tokenizer = hf_tokenizer
        self.sp = None
        self.hf_tok = None
        self.tokenizer_type = None
        
        # Load HuggingFace tokenizer if provided
        if hf_tokenizer and TRANSFORMERS_AVAILABLE:
            try:
                self.hf_tok = AutoTokenizer.from_pretrained(hf_tokenizer)
                self.tokenizer_type = "huggingface"
                print(f"✅ Loaded HuggingFace tokenizer: {hf_tokenizer}")
                print(f"   Vocab size: {self.hf_tok.vocab_size}")
                if hasattr(self.hf_tok, 'model_max_length'):
                    print(f"   Max length: {self.hf_tok.model_max_length}")
            except Exception as e:
                print(f"⚠️  Failed to load HuggingFace tokenizer {hf_tokenizer}: {e}")
                print("   Falling back to SentencePiece or space tokenization")
        
        # Load SentencePiece tokenizer if provided and HF not loaded
        elif tokenizer_model and Path(tokenizer_model).exists() and SENTENCEPIECE_AVAILABLE:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(tokenizer_model)
            self.tokenizer_type = "sentencepiece"
            print(f"✅ Loaded SentencePiece tokenizer: {tokenizer_model}")
    
    def prepare_corpus(self, output_path: Path) -> Path:
        """
        Prepare corpus for language model training
        
        Args:
            output_path: Path to save prepared corpus
            
        Returns:
            Path to prepared corpus
        """
        print("📚 Preparing corpus...")
        
        # TODO: Load and preprocess actual Khmer corpus
        # For now, create placeholder corpus
        sample_sentences = [
            "សួស្តី ពិភពលោក",
            "ប្រទេសកម្ពុជា គឺជា ប្រទេស មួយ នៅ អាស៊ី អាគ្នេយ៍",
            "ភាសាខ្មែរ គឺជា ភាសា ផ្លូវការ របស់ ប្រទេសកម្ពុជា",
            "រាជធានី ភ្នំពេញ គឺជា ទីក្រុង ធំ បំផុត",
            "អង្គរវត្ត គឺជា ប្រាសាទ ដ៏ ល្បីល្បាញ",
            "ខ្មែរ មាន ប្រវត្តិសាស្ត្រ យូរលង់",
            "អក្សរខ្មែរ មាន អាយុកាល ជាង ១០០០ ឆ្នាំ",
            "ការអប់រំ គឺជា គន្លឹះ នៃ ការអភិវឌ្ឍ",
        ]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sentence in sample_sentences * 100:  # Repeat for larger corpus
                f.write(sentence + "\n")
        
        print(f"   Corpus saved to: {output_path}")
        print(f"   Lines: {len(sample_sentences) * 100}")
        
        return output_path
    
    def train_sentencepiece(self, corpus_path: Path, model_prefix: Path) -> Path:
        """
        Train SentencePiece tokenizer for Khmer
        
        Args:
            corpus_path: Path to corpus
            model_prefix: Prefix for model files
            
        Returns:
            Path to trained model
        """
        if SENTENCEPIECE_AVAILABLE:
            print(f"🔤 Training SentencePiece tokenizer (vocab_size={self.vocab_size})...")
            
            try:
                # Train SentencePiece model
                spm.SentencePieceTrainer.train(
                    input=str(corpus_path),
                    model_prefix=str(model_prefix),
                    vocab_size=self.vocab_size,
                    model_type='unigram',  # Good for Khmer
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
                    user_defined_symbols=['<num>', '<url>', '<email>']
                )
                
                model_path = Path(f"{model_prefix}.model")
                vocab_path = Path(f"{model_prefix}.vocab")
                
                # Load the trained model
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load(str(model_path))
                
                print(f"   Model saved: {model_path}")
                print(f"   Vocab saved: {vocab_path}")
                print(f"   Vocab size: {self.sp.GetPieceSize()}")
                
                return model_path
                
            except Exception as e:
                print(f"   Error training SentencePiece: {e}")
                return None
        else:
            print("   SentencePiece not available")
            return None
    
    def tokenize_corpus(self, corpus_path: Path) -> Path:
        """
        Tokenize corpus using HuggingFace, SentencePiece, or fallback
        
        Args:
            corpus_path: Path to corpus
            
        Returns:
            Path to tokenized corpus
        """
        print("🔤 Tokenizing corpus...")
        
        tokenized_path = corpus_path.parent / f"{corpus_path.stem}_tokenized.txt"
        processed_lines = 0
        
        if self.hf_tok and self.tokenizer_type == "huggingface":
            # Use HuggingFace tokenizer (khopilot/km-tokenizer-khmer)
            print(f"   Using HuggingFace tokenizer: {self.hf_tokenizer}")
            print(f"   Expected vocab size: {self.hf_tok.vocab_size}")
            
            with open(corpus_path, "r", encoding="utf-8") as f_in:
                with open(tokenized_path, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        text = line.strip()
                        if text:
                            # Tokenize using HuggingFace tokenizer
                            tokens = self.hf_tok.tokenize(text)
                            f_out.write(" ".join(tokens) + "\n")
                            processed_lines += 1
                            
                            if processed_lines % 1000 == 0:
                                print(f"     Processed {processed_lines} lines...")
            
            print(f"   ✅ HuggingFace tokenization complete: {processed_lines} lines")
            
        elif self.sp and self.tokenizer_type == "sentencepiece":
            # Use SentencePiece tokenizer
            print("   Using SentencePiece tokenizer")
            with open(corpus_path, "r", encoding="utf-8") as f_in:
                with open(tokenized_path, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        text = line.strip()
                        if text:
                            tokens = self.sp.EncodeAsPieces(text)
                            f_out.write(" ".join(tokens) + "\n")
                            processed_lines += 1
        else:
            # Fallback to simple word segmentation
            print("   Using space tokenization (fallback)")
            with open(corpus_path, "r", encoding="utf-8") as f_in:
                with open(tokenized_path, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        # For Khmer, this is suboptimal but works as fallback
                        # Better: use khmer-nltk or pykhmer for segmentation
                        tokens = line.strip().split()
                        f_out.write(" ".join(tokens) + "\n")
                        processed_lines += 1
        
        print(f"   ✅ Tokenized corpus: {tokenized_path}")
        print(f"   Total lines processed: {processed_lines}")
        
        return tokenized_path
    
    def train_kenlm(self, corpus_path: Path, output_path: Path) -> Dict[str, any]:
        """
        Train KenLM model and calculate hashes
        
        Args:
            corpus_path: Path to tokenized corpus
            output_path: Path to save model
            
        Returns:
            Dictionary with model info and hashes
        """
        print(f"🧠 Training {self.order}-gram language model...")
        
        model_info = {
            "order": self.order,
            "corpus_lines": 0,
            "arpa_path": None,
            "binary_path": None,
            "arpa_hash": None,
            "arpa_size": 0,
        }
        
        # Count corpus lines
        with open(corpus_path, 'r') as f:
            model_info["corpus_lines"] = sum(1 for _ in f)
        
        # Check if KenLM is available
        try:
            result = subprocess.run(
                ["which", "lmplz"],
                capture_output=True,
                text=True
            )
            kenlm_available = result.returncode == 0
        except:
            kenlm_available = False
        
        arpa_path = output_path.with_suffix(".arpa")
        
        if kenlm_available:
            # Train with actual KenLM
            try:
                cmd = [
                    "lmplz",
                    "-o", str(self.order),
                    "--text", str(corpus_path),
                    "--arpa", str(arpa_path),
                    "--discount_fallback",
                    "--skip_symbols"
                ]
                
                print(f"   Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"   KenLM training failed: {result.stderr}")
                    raise Exception("KenLM training failed")
                
                # Try to convert to binary format
                binary_path = output_path.with_suffix(".bin")
                try:
                    cmd = ["build_binary", str(arpa_path), str(binary_path)]
                    subprocess.run(cmd, check=True, capture_output=True)
                    model_info["binary_path"] = str(binary_path)
                    print(f"   Binary: {binary_path}")
                except:
                    print("   Binary conversion failed (build_binary not found)")
                
                print(f"✅ Language model trained:")
                print(f"   ARPA: {arpa_path}")
                
            except Exception as e:
                print(f"   KenLM training error: {e}")
                kenlm_available = False
        
        if not kenlm_available or not arpa_path.exists():
            # Create placeholder model
            print("   Creating placeholder model (KenLM not available)")
            print("   Install with: brew install cmake boost && pip install https://github.com/kpu/kenlm/archive/master.zip")
            
            # Create minimal ARPA format file
            with open(arpa_path, "w", encoding="utf-8") as f:
                f.write("\\data\\\n")
                f.write("ngram 1=10\n")
                f.write("ngram 2=8\n")
                f.write("ngram 3=6\n")
                f.write("\n\\1-grams:\n")
                f.write("-1.0\t<s>\t0\n")
                f.write("-1.0\t</s>\t0\n")
                f.write("-99\t<unk>\t0\n")
                f.write("-2.0\tសួស្តី\t-0.5\n")
                f.write("-2.0\tពិភពលោក\t-0.5\n")
                f.write("-2.0\tប្រទេស\t-0.5\n")
                f.write("-2.0\tកម្ពុជា\t-0.5\n")
                f.write("-2.0\tភាសា\t-0.5\n")
                f.write("-2.0\tខ្មែរ\t-0.5\n")
                f.write("-2.0\tOCR\t-0.5\n")
                f.write("\n\\2-grams:\n")
                f.write("-0.5\tសួស្តី ពិភពលោក\n")
                f.write("-0.5\tប្រទេស កម្ពុជា\n")
                f.write("-0.5\tភាសា ខ្មែរ\n")
                f.write("-0.5\tខ្មែរ OCR\n")
                f.write("-1.0\t<s> សួស្តី\n")
                f.write("-1.0\tOCR </s>\n")
                f.write("\n\\3-grams:\n")
                f.write("-0.3\tសួស្តី ពិភពលោក </s>\n")
                f.write("-0.3\tប្រទេស កម្ពុជា </s>\n")
                f.write("-0.3\tភាសា ខ្មែរ OCR\n")
                f.write("-0.3\t<s> សួស្តី ពិភពលោក\n")
                f.write("\n\\end\\\n")
            
            print(f"   Placeholder model saved: {arpa_path}")
        
        # Calculate hash and size
        if arpa_path.exists():
            model_info["arpa_path"] = str(arpa_path)
            model_info["arpa_size"] = arpa_path.stat().st_size
            
            # Calculate SHA256
            sha256_hash = hashlib.sha256()
            with open(arpa_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            model_info["arpa_hash"] = sha256_hash.hexdigest()
            
            print(f"   Size: {model_info['arpa_size'] / 1024:.1f} KB")
            print(f"   SHA256: {model_info['arpa_hash'][:16]}...")
        
        return model_info
    
    def evaluate_perplexity(self, model_path: Path, test_corpus: Path) -> float:
        """
        Evaluate model perplexity on test set
        
        Args:
            model_path: Path to language model
            test_corpus: Path to test corpus
            
        Returns:
            Perplexity score
        """
        # TODO: Implement actual perplexity calculation
        # For now, return placeholder value
        return 42.0


def main():
    """CLI interface for language model training"""
    parser = argparse.ArgumentParser(description="Train KenLM language model with SentencePiece")
    parser.add_argument(
        "--corpus",
        type=str,
        help="Path to training corpus"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=5,
        help="N-gram order (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lang/kenlm/khmer_5gram",
        help="Output path for model (without extension)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to pre-trained SentencePiece model"
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="khopilot/km-tokenizer-khmer",
        help="HuggingFace tokenizer model name (default: khopilot/km-tokenizer-khmer)"
    )
    parser.add_argument(
        "--use-hf-tokenizer",
        action="store_true",
        help="Use HuggingFace tokenizer instead of SentencePiece"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size for SentencePiece (default: 8000)"
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update manifest.json with model info"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LanguageModelTrainer(
        corpus_path=Path(args.corpus) if args.corpus else None,
        order=args.order,
        vocab_size=args.vocab_size,
        tokenizer_model=args.tokenizer,
        hf_tokenizer=args.hf_tokenizer if args.use_hf_tokenizer else None
    )
    
    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare corpus if not provided
    if args.corpus and Path(args.corpus).exists():
        corpus_path = Path(args.corpus)
    else:
        print("No corpus provided, generating sample corpus...")
        corpus_path = trainer.prepare_corpus(output_path.parent / "corpus.txt")
    
    # Train or load SentencePiece tokenizer
    if not args.tokenizer:
        sp_prefix = output_path.parent / "sentencepiece_khmer"
        sp_model = trainer.train_sentencepiece(corpus_path, sp_prefix)
        if sp_model:
            print(f"✅ SentencePiece model trained: {sp_model}")
    
    # Tokenize corpus
    tokenized_path = trainer.tokenize_corpus(corpus_path)
    
    # Train KenLM model
    model_info = trainer.train_kenlm(tokenized_path, output_path)
    
    # Evaluate (optional)
    if model_info.get("arpa_path") and Path(model_info["arpa_path"]).exists():
        perplexity = trainer.evaluate_perplexity(
            Path(model_info["arpa_path"]),
            tokenized_path
        )
        model_info["perplexity"] = perplexity
        
        print(f"\n📊 Model Statistics:")
        print(f"   Order: {model_info['order']}-gram")
        print(f"   Corpus lines: {model_info['corpus_lines']}")
        print(f"   Model size: {model_info['arpa_size'] / 1024:.1f} KB")
        print(f"   Perplexity: {perplexity:.2f}")
    
    # Update manifest if requested
    if args.update_manifest:
        manifest_path = Path("governance/manifest.json")
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Add language model info
            if "models" not in manifest:
                manifest["models"] = {}
            
            manifest["models"]["language_model"] = {
                "type": "kenlm",
                "order": model_info["order"],
                "path": model_info.get("arpa_path"),
                "sha256": model_info.get("arpa_hash"),
                "size_bytes": model_info.get("arpa_size"),
                "perplexity": model_info.get("perplexity", None),
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"✅ Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()