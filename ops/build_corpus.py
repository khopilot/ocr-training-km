#!/usr/bin/env python3
"""
Build training corpus from converted HuggingFace datasets for KenLM training.

Aggregates text from PaddleOCR formatted datasets and creates a clean corpus
for language model training with the khopilot/km-tokenizer-khmer tokenizer.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def extract_text_from_paddle_labels(label_file: Path) -> List[str]:
    """Extract text from PaddleOCR label file format."""
    texts = []
    
    if not label_file.exists():
        logging.warning(f"Label file not found: {label_file}")
        return texts
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # PaddleOCR format: image_path\ttext or image_path\tjson_annotations
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue
                
                text_part = parts[1]
                
                # Check if it's JSON (detection format)
                if text_part.startswith('[') or text_part.startswith('{'):
                    try:
                        # Parse JSON and extract transcriptions
                        data = json.loads(text_part)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'transcription' in item:
                                    if item['transcription'].strip():
                                        texts.append(item['transcription'].strip())
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as plain text
                        if text_part.strip():
                            texts.append(text_part.strip())
                else:
                    # Plain text format (recognition)
                    if text_part.strip():
                        texts.append(text_part.strip())
                        
            except Exception as e:
                logging.debug(f"Error processing line {line_no} in {label_file}: {e}")
                continue
    
    return texts


def build_corpus(input_dirs: List[Path], output_path: Path, 
                min_length: int = 2, max_length: int = 200,
                remove_duplicates: bool = True) -> Dict:
    """
    Build corpus from converted PaddleOCR datasets.
    
    Args:
        input_dirs: List of directories containing PaddleOCR format data
        output_path: Output corpus file path
        min_length: Minimum text length (characters)
        max_length: Maximum text length (characters) 
        remove_duplicates: Whether to remove duplicate texts
    
    Returns:
        Statistics dictionary
    """
    all_texts = []
    stats = {
        "total_files_processed": 0,
        "total_texts_extracted": 0,
        "texts_filtered": 0,
        "final_corpus_size": 0,
        "duplicates_removed": 0,
        "source_distribution": Counter(),
        "length_distribution": {
            "min": float('inf'),
            "max": 0,
            "avg": 0.0
        },
        "character_count": Counter(),
        "unique_characters": set()
    }
    
    logging.info(f"🔄 Building corpus from {len(input_dirs)} directories...")
    
    for input_dir in input_dirs:
        if not input_dir.exists():
            logging.warning(f"Directory not found: {input_dir}")
            continue
        
        logging.info(f"  Processing directory: {input_dir}")
        dir_texts = []
        
        # Process all splits (train, val, test)
        for split in ["train", "val", "test"]:
            split_dir = input_dir / split
            if split_dir.exists():
                label_file = split_dir / "label.txt"
                texts = extract_text_from_paddle_labels(label_file)
                dir_texts.extend(texts)
                stats["total_files_processed"] += 1
                logging.info(f"    {split}: {len(texts)} texts extracted")
        
        # Also check root level label.txt files
        root_label = input_dir / "label.txt"
        if root_label.exists():
            texts = extract_text_from_paddle_labels(root_label)
            dir_texts.extend(texts)
            stats["total_files_processed"] += 1
            logging.info(f"    root: {len(texts)} texts extracted")
        
        all_texts.extend(dir_texts)
        stats["source_distribution"][input_dir.name] = len(dir_texts)
        logging.info(f"  ✅ Total from {input_dir.name}: {len(dir_texts)} texts")
    
    stats["total_texts_extracted"] = len(all_texts)
    logging.info(f"📊 Total texts extracted: {len(all_texts)}")
    
    # Filter by length
    filtered_texts = []
    for text in all_texts:
        text_len = len(text)
        if min_length <= text_len <= max_length:
            filtered_texts.append(text)
            # Update length statistics
            stats["length_distribution"]["min"] = min(
                stats["length_distribution"]["min"], text_len
            )
            stats["length_distribution"]["max"] = max(
                stats["length_distribution"]["max"], text_len
            )
            # Count characters
            for char in text:
                stats["character_count"][char] += 1
                stats["unique_characters"].add(char)
        else:
            stats["texts_filtered"] += 1
    
    logging.info(f"📏 After length filtering: {len(filtered_texts)} texts")
    logging.info(f"   Filtered out: {stats['texts_filtered']} texts")
    
    # Remove duplicates if requested
    if remove_duplicates:
        unique_texts = list(set(filtered_texts))
        stats["duplicates_removed"] = len(filtered_texts) - len(unique_texts)
        filtered_texts = unique_texts
        logging.info(f"🔍 After deduplication: {len(filtered_texts)} texts")
        logging.info(f"   Duplicates removed: {stats['duplicates_removed']}")
    
    # Calculate average length
    if filtered_texts:
        stats["length_distribution"]["avg"] = sum(len(t) for t in filtered_texts) / len(filtered_texts)
        if stats["length_distribution"]["min"] == float('inf'):
            stats["length_distribution"]["min"] = 0
    
    # Write corpus
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in filtered_texts:
            f.write(text + '\n')
    
    stats["final_corpus_size"] = len(filtered_texts)
    
    # Convert sets to counts for JSON serialization
    stats["source_distribution"] = dict(stats["source_distribution"])
    stats["character_count"] = dict(stats["character_count"])
    stats["unique_characters"] = len(stats["unique_characters"])
    
    logging.info(f"✅ Corpus written to: {output_path}")
    logging.info(f"📊 Final corpus size: {stats['final_corpus_size']} texts")
    logging.info(f"📊 Unique characters: {stats['unique_characters']}")
    logging.info(f"📊 Length range: {stats['length_distribution']['min']}-{stats['length_distribution']['max']} (avg: {stats['length_distribution']['avg']:.1f})")
    
    return stats


def write_corpus_report(stats: Dict, output_dir: Path) -> None:
    """Write corpus building statistics report."""
    report_path = output_dir / "corpus_report.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logging.info(f"📄 Corpus report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Build corpus from PaddleOCR datasets")
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Input directories with PaddleOCR format data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus/khmer_training_corpus.txt"),
        help="Output corpus file path"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum text length (characters)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum text length (characters)"
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate texts (default: remove duplicates)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate statistics report without building corpus"
    )
    
    args = parser.parse_args()
    
    if not args.input_dirs:
        logging.error("No input directories provided")
        return 1
    
    # Validate input directories
    valid_dirs = []
    for input_dir in args.input_dirs:
        if input_dir.exists():
            valid_dirs.append(input_dir)
        else:
            logging.warning(f"Input directory not found: {input_dir}")
    
    if not valid_dirs:
        logging.error("No valid input directories found")
        return 1
    
    try:
        # Build corpus
        stats = build_corpus(
            input_dirs=valid_dirs,
            output_path=args.output,
            min_length=args.min_length,
            max_length=args.max_length,
            remove_duplicates=not args.keep_duplicates
        )
        
        # Write report
        write_corpus_report(stats, args.output.parent)
        
        logging.info("🎉 Corpus building complete!")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("🛑 Corpus building interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"❌ Corpus building failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())