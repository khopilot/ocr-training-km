#!/usr/bin/env python3
"""Build unified Khmer lexicon from multiple dictionary datasets.

Combines vocabulary from:
1. khmer-dictionary-44k - Royal Academy dictionary
2. google-khmer-lexicon - Google's Khmer lexicon
3. lexicon-kh - Additional vocabulary

Outputs:
- Unified vocabulary file for KenLM training
- Lexicon for OCR post-processing
- Character set extraction
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from datasets import load_dataset
    import unicodedata
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Missing dependencies. Install with: pip install datasets")


@dataclass
class LexiconEntry:
    """Single lexicon entry with metadata."""
    word: str
    source: str
    frequency: int = 1
    pronunciation: str = ""
    pos: str = ""  # Part of speech
    
    def normalized(self) -> str:
        """Return Unicode-normalized form."""
        return unicodedata.normalize("NFC", self.word)


class KhmerTextProcessor:
    """Process and validate Khmer text."""
    
    # Khmer Unicode ranges
    KHMER_MAIN = range(0x1780, 0x17FF)  # Main Khmer block
    KHMER_SYMBOLS = range(0x19E0, 0x19FF)  # Khmer symbols
    
    # Common Khmer punctuation
    KHMER_PUNCTUATION = {
        "។",  # Khmer period (khan)
        "៕",  # Khmer question mark
        "៖",  # Khmer colon
        "ៗ",  # Khmer repetition mark
        "៘",  # Khmer avakrahasanya
        "៙",  # Khmer mut
        "៚",  # Khmer triple period
    }
    
    @classmethod
    def is_khmer_char(cls, char: str) -> bool:
        """Check if character is Khmer."""
        code = ord(char)
        return (code in cls.KHMER_MAIN or 
                code in cls.KHMER_SYMBOLS or 
                char in cls.KHMER_PUNCTUATION)
    
    @classmethod
    def is_valid_khmer_word(cls, word: str) -> bool:
        """Check if word contains valid Khmer characters."""
        if not word:
            return False
        
        # Allow some ASCII (numbers, common punctuation)
        allowed_ascii = set("0123456789-_.")
        
        for char in word:
            if not (cls.is_khmer_char(char) or 
                   char in allowed_ascii or
                   char.isspace()):
                return False
        
        # Must have at least one Khmer character
        return any(cls.is_khmer_char(c) for c in word)
    
    @classmethod
    def extract_khmer_words(cls, text: str) -> List[str]:
        """Extract individual Khmer words from text."""
        # Split on spaces and punctuation
        words = re.split(r'[\s\u200b]+', text)  # Include zero-width space
        
        valid_words = []
        for word in words:
            # Clean word
            word = word.strip()
            word = re.sub(r'[^\u1780-\u17FF\u19E0-\u19FF0-9\-_.]', '', word)
            
            if cls.is_valid_khmer_word(word):
                valid_words.append(word)
        
        return valid_words
    
    @classmethod
    def extract_characters(cls, text: str) -> Set[str]:
        """Extract unique Khmer characters from text."""
        chars = set()
        for char in text:
            if cls.is_khmer_char(char):
                chars.add(char)
        return chars


def load_dictionary_44k(input_dir: Path) -> List[LexiconEntry]:
    """Load khmer-dictionary-44k dataset."""
    print("📚 Loading khmer-dictionary-44k...")
    
    entries = []
    dataset = load_dataset("seanghay/khmer-dictionary-44k", cache_dir=str(input_dir))
    
    if "train" in dataset:
        for item in dataset["train"]:
            word = item.get("word", "").strip()
            if not word:
                continue
            
            entry = LexiconEntry(
                word=word,
                source="khmer-dictionary-44k",
                pronunciation=item.get("pro", ""),
                pos=item.get("pos", "")
            )
            entries.append(entry)
    
    print(f"  ✅ Loaded {len(entries)} entries")
    return entries


def load_google_lexicon(input_dir: Path) -> List[LexiconEntry]:
    """Load google-khmer-lexicon dataset."""
    print("📚 Loading google-khmer-lexicon...")
    
    entries = []
    dataset = load_dataset("seanghay/google-khmer-lexicon", cache_dir=str(input_dir))
    
    if "train" in dataset:
        for item in dataset["train"]:
            word = item.get("word", "").strip()
            if not word:
                continue
            
            entry = LexiconEntry(
                word=word,
                source="google-khmer-lexicon",
                pronunciation=item.get("pro", "")
            )
            entries.append(entry)
    
    print(f"  ✅ Loaded {len(entries)} entries")
    return entries


def load_lexicon_kh(input_dir: Path) -> List[LexiconEntry]:
    """Load lexicon-kh dataset."""
    print("📚 Loading lexicon-kh...")
    
    entries = []
    dataset = load_dataset("seanghay/lexicon-kh", cache_dir=str(input_dir))
    
    if "train" in dataset:
        for item in dataset["train"]:
            # Adjust field names based on actual dataset structure
            word = item.get("word", item.get("text", "")).strip()
            if not word:
                continue
            
            entry = LexiconEntry(
                word=word,
                source="lexicon-kh"
            )
            entries.append(entry)
    
    print(f"  ✅ Loaded {len(entries)} entries")
    return entries


def build_unified_lexicon(entries: List[LexiconEntry]) -> Tuple[Dict[str, LexiconEntry], Set[str]]:
    """Build unified lexicon removing duplicates and extracting characters.
    
    Returns:
        Tuple of (unified lexicon dict, character set)
    """
    print("\n🔨 Building unified lexicon...")
    
    processor = KhmerTextProcessor()
    unified = {}
    all_chars = set()
    word_freq = Counter()
    
    # Process all entries
    for entry in entries:
        # Normalize word
        word_norm = entry.normalized()
        
        # Validate Khmer word
        if not processor.is_valid_khmer_word(word_norm):
            continue
        
        # Extract characters
        chars = processor.extract_characters(word_norm)
        all_chars.update(chars)
        
        # Track frequency
        word_freq[word_norm] += 1
        
        # Keep first occurrence or merge metadata
        if word_norm not in unified:
            unified[word_norm] = entry
            unified[word_norm].word = word_norm  # Use normalized form
        else:
            # Merge metadata if from different source
            if entry.source != unified[word_norm].source:
                unified[word_norm].source += f",{entry.source}"
            # Keep pronunciation if missing
            if not unified[word_norm].pronunciation and entry.pronunciation:
                unified[word_norm].pronunciation = entry.pronunciation
    
    # Update frequencies
    for word, freq in word_freq.items():
        if word in unified:
            unified[word].frequency = freq
    
    print(f"  ✅ Unified: {len(unified)} unique words")
    print(f"  ✅ Characters: {len(all_chars)} unique characters")
    
    return unified, all_chars


def write_lexicon_files(
    lexicon: Dict[str, LexiconEntry],
    chars: Set[str],
    output_dir: Path
) -> None:
    """Write lexicon and character files."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort words by frequency (descending) then alphabetically
    sorted_words = sorted(
        lexicon.values(),
        key=lambda e: (-e.frequency, e.word)
    )
    
    # Write main lexicon file (word per line)
    lexicon_file = output_dir / "khmer_lexicon.txt"
    with open(lexicon_file, "w", encoding="utf-8") as f:
        for entry in sorted_words:
            f.write(f"{entry.word}\n")
    print(f"  📝 Lexicon: {lexicon_file}")
    
    # Write lexicon with metadata (JSON)
    metadata_file = output_dir / "khmer_lexicon_metadata.json"
    metadata = []
    for entry in sorted_words:
        metadata.append({
            "word": entry.word,
            "frequency": entry.frequency,
            "source": entry.source,
            "pronunciation": entry.pronunciation,
            "pos": entry.pos
        })
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  📝 Metadata: {metadata_file}")
    
    # Write lexicon for KenLM (with frequencies)
    kenlm_file = output_dir / "khmer_lexicon_kenlm.txt"
    with open(kenlm_file, "w", encoding="utf-8") as f:
        for entry in sorted_words:
            # KenLM format: word \t frequency
            f.write(f"{entry.word}\t{entry.frequency}\n")
    print(f"  📝 KenLM lexicon: {kenlm_file}")
    
    # Write character set file
    charset_file = output_dir / "charset_khmer.txt"
    sorted_chars = sorted(chars, key=lambda c: ord(c))
    with open(charset_file, "w", encoding="utf-8") as f:
        for char in sorted_chars:
            f.write(f"{char}\n")
    print(f"  📝 Character set: {charset_file}")
    
    # Write character set with Unicode info
    charset_info_file = output_dir / "charset_khmer_info.json"
    char_info = []
    for char in sorted_chars:
        char_info.append({
            "char": char,
            "unicode": f"U+{ord(char):04X}",
            "name": unicodedata.name(char, "UNKNOWN"),
            "category": unicodedata.category(char)
        })
    
    with open(charset_info_file, "w", encoding="utf-8") as f:
        json.dump(char_info, f, ensure_ascii=False, indent=2)
    print(f"  📝 Character info: {charset_info_file}")


def generate_statistics(lexicon: Dict[str, LexiconEntry], chars: Set[str]) -> Dict:
    """Generate statistics about the lexicon."""
    stats = {
        "total_words": len(lexicon),
        "total_characters": len(chars),
        "sources": {},
        "word_length_distribution": {},
        "character_categories": {}
    }
    
    # Source distribution
    source_counts = Counter()
    for entry in lexicon.values():
        for source in entry.source.split(","):
            source_counts[source] += 1
    stats["sources"] = dict(source_counts)
    
    # Word length distribution
    length_counts = Counter()
    for word in lexicon.keys():
        length_counts[len(word)] += 1
    stats["word_length_distribution"] = dict(sorted(length_counts.items()))
    
    # Character categories
    cat_counts = Counter()
    for char in chars:
        cat = unicodedata.category(char)
        cat_counts[cat] += 1
    stats["character_categories"] = dict(cat_counts)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build unified Khmer lexicon")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/hf_datasets"),
        help="Input directory with downloaded HF datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lang/lexicon"),
        help="Output directory for lexicon files"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["dictionary", "google", "lexicon-kh", "all"],
        default=["all"],
        help="Which sources to include"
    )
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Use only open-source lexicons (skip proprietary sources like Google)"
    )
    
    args = parser.parse_args()
    
    if not HAS_DEPS:
        print("Error: Required packages not installed.")
        print("Run: pip install datasets")
        return 1
    
    # Load datasets
    all_entries = []
    
    # Define proprietary vs open sources
    proprietary_sources = ["google"]
    open_sources = ["dictionary", "lexicon-kh"]
    
    # Filter sources based on --open-only flag
    active_sources = []
    if "all" in args.sources:
        if args.open_only:
            active_sources = open_sources
            print("🔒 Open-only mode: Skipping proprietary sources (Google)")
        else:
            active_sources = open_sources + proprietary_sources
    else:
        active_sources = [s for s in args.sources if s != "all"]
        if args.open_only:
            active_sources = [s for s in active_sources if s not in proprietary_sources]
            skipped = [s for s in args.sources if s in proprietary_sources]
            if skipped:
                print(f"🔒 Open-only mode: Skipping proprietary sources: {', '.join(skipped)}")
    
    if "dictionary" in active_sources:
        try:
            entries = load_dictionary_44k(args.input_dir / "khmer-dictionary-44k")
            all_entries.extend(entries)
        except Exception as e:
            print(f"  ⚠️  Failed to load dictionary: {e}")
    
    if "google" in active_sources:
        try:
            entries = load_google_lexicon(args.input_dir / "google-khmer-lexicon")
            all_entries.extend(entries)
        except Exception as e:
            print(f"  ⚠️  Failed to load Google lexicon: {e}")
    
    if "lexicon-kh" in active_sources:
        try:
            entries = load_lexicon_kh(args.input_dir / "lexicon-kh")
            all_entries.extend(entries)
        except Exception as e:
            print(f"  ⚠️  Failed to load lexicon-kh: {e}")
    
    if not all_entries:
        print("❌ No entries loaded!")
        return 1
    
    # Build unified lexicon
    lexicon, chars = build_unified_lexicon(all_entries)
    
    # Generate statistics
    stats = generate_statistics(lexicon, chars)
    
    # Write output files
    print("\n📝 Writing output files...")
    write_lexicon_files(lexicon, chars, args.output_dir)
    
    # Write statistics
    stats_file = args.output_dir / "lexicon_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  📝 Statistics: {stats_file}")
    
    # Summary
    print("\n" + "="*50)
    print("📊 Lexicon Build Summary:")
    print(f"  Total entries processed: {len(all_entries)}")
    print(f"  Unique words: {stats['total_words']}")
    print(f"  Unique characters: {stats['total_characters']}")
    print(f"  Sources: {', '.join(stats['sources'].keys())}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())