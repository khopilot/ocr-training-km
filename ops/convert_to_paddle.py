#!/usr/bin/env python3
"""Convert HuggingFace datasets to PaddleOCR training format.

PaddleOCR expects the following formats:
- Detection: image_path \t json_bbox_annotations
- Recognition: image_path \t text_label

This script converts:
1. SynthKhmer-10k → Detection + Recognition format
2. khmerfonts-info-previews → Recognition format
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

try:
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Missing dependencies. Install with: pip install datasets pillow numpy")


@dataclass
class BBox:
    """Bounding box for text detection."""
    x1: int
    y1: int
    x2: int
    y2: int
    text: str
    
    def to_paddle_format(self) -> List[List[int]]:
        """Convert to PaddleOCR format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
        return [
            [self.x1, self.y1],
            [self.x2, self.y1],
            [self.x2, self.y2],
            [self.x1, self.y2]
        ]


@dataclass
class DatasetSchema:
    """Dataset schema discovery and field mapping."""
    dataset_name: str
    expected_fields: Set[str]
    actual_fields: Set[str] = field(default_factory=set)
    missing_fields: Set[str] = field(default_factory=set)
    extra_fields: Set[str] = field(default_factory=set)
    field_types: Dict[str, type] = field(default_factory=dict)
    sample_values: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    null_counts: Dict[str, int] = field(default_factory=Counter)
    total_samples: int = 0
    
    def discover_schema(self, samples: List[Dict]) -> None:
        """Discover actual schema from sample data."""
        self.total_samples = len(samples)
        
        for sample in samples:
            if isinstance(sample, dict):
                self.actual_fields.update(sample.keys())
                
                for field_name, value in sample.items():
                    if value is None:
                        self.null_counts[field_name] += 1
                    else:
                        self.field_types[field_name] = type(value)
                        # Store sample values for inspection (limit to 5 per field)
                        if len(self.sample_values[field_name]) < 5:
                            self.sample_values[field_name].append(value)
        
        self.missing_fields = self.expected_fields - self.actual_fields
        self.extra_fields = self.actual_fields - self.expected_fields
    
    def get_mapping_report(self) -> Dict:
        """Generate field mapping report."""
        return {
            "dataset": self.dataset_name,
            "total_samples": self.total_samples,
            "schema_match": {
                "expected_fields": sorted(self.expected_fields),
                "actual_fields": sorted(self.actual_fields),
                "missing_fields": sorted(self.missing_fields),
                "extra_fields": sorted(self.extra_fields),
                "field_coverage": len(self.actual_fields & self.expected_fields) / len(self.expected_fields) * 100
            },
            "field_analysis": {
                field: {
                    "type": self.field_types.get(field, "unknown").__name__,
                    "null_rate": round(self.null_counts[field] / self.total_samples * 100, 2),
                    "sample_values": self.sample_values[field][:3]  # Show first 3 samples
                }
                for field in sorted(self.actual_fields)
            }
        }


def convert_synthkhmer_10k(
    input_dir: Path,
    output_dir: Path,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict:
    """Convert SynthKhmer-10k dataset to PaddleOCR format.
    
    Returns:
        Dictionary with conversion statistics
    """
    if not HAS_DEPS:
        raise ImportError("Required dependencies not installed")
    
    print("🔄 Converting SynthKhmer-10k dataset...")
    
    # Load dataset
    dataset = load_dataset("seanghay/SynthKhmer-10k", cache_dir=str(input_dir))
    
    # Initialize schema discovery
    expected_fields = {"image", "name", "date_of_birth", "gender", "id", "bbox"}
    schema = DatasetSchema("SynthKhmer-10k", expected_fields)
    
    if "train" not in dataset:
        logging.warning("No 'train' split found in dataset")
        return {"error": "No train split found"}
    
    data = dataset["train"]
    
    # Discover schema from first 100 samples
    schema_samples = [data[i] for i in range(min(100, len(data)))]
    schema.discover_schema(schema_samples)
    
    # Log schema discovery results
    mapping_report = schema.get_mapping_report()
    logging.info(f"📋 Schema Discovery for {schema.dataset_name}:")
    logging.info(f"  Field coverage: {mapping_report['schema_match']['field_coverage']:.1f}%")
    logging.info(f"  Missing fields: {mapping_report['schema_match']['missing_fields']}")
    logging.info(f"  Extra fields: {mapping_report['schema_match']['extra_fields']}")
    
    # Create output directories
    det_dir = output_dir / "detection"
    rec_dir = output_dir / "recognition"
    
    for split in ["train", "val", "test"]:
        (det_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (rec_dir / split / "images").mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    stats = {
        "total_images": 0,
        "total_bboxes": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0
    }
    
    det_labels = {"train": [], "val": [], "test": []}
    rec_labels = {"train": [], "val": [], "test": []}
    total_samples = len(data)
    
    # Calculate split indices
    train_end = int(total_samples * split_ratio[0])
    val_end = train_end + int(total_samples * split_ratio[1])
    
    # Schema-based field mapping
    def safe_extract_field(sample: Dict, field: str, default: Any = None) -> Any:
        """Safely extract field with logging."""
        if field in sample and sample[field] is not None:
            return sample[field]
        if field not in schema.missing_fields:  # Only log if we expected this field
            logging.debug(f"Missing or null field '{field}' in sample")
        return default
    
    # Track conversion statistics
    conversion_stats = {
        "total_processed": 0,
        "total_dropped": 0,
        "drop_reasons": Counter(),
        "field_extractions": Counter(),
        "bbox_sources": Counter()
    }
    
    for idx, sample in enumerate(data):
        conversion_stats["total_processed"] += 1
        # Determine split
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        
        # Save image
        image = safe_extract_field(sample, "image")
        if image is None:
            conversion_stats["total_dropped"] += 1
            conversion_stats["drop_reasons"]["missing_image"] += 1
            continue
        
        image_name = f"synth_{idx:06d}.jpg"
        image_path = Path(split) / "images" / image_name
        
        # Save for detection
        det_image_path = det_dir / image_path
        if isinstance(image, Image.Image):
            image.save(det_image_path)
        
        # Save for recognition (copy)
        rec_image_path = rec_dir / image_path
        shutil.copy(det_image_path, rec_image_path)
        
        # Process annotations with intelligent field mapping
        bboxes = []
        
        # Try to extract actual bbox data if available
        bbox_data = safe_extract_field(sample, "bbox") or safe_extract_field(sample, "bboxes")
        
        # Field mapping with flexible bbox handling
        text_fields = [
            ("name", safe_extract_field(sample, "name")),
            ("date_of_birth", safe_extract_field(sample, "date_of_birth") or safe_extract_field(sample, "dob")),
            ("gender", safe_extract_field(sample, "gender")),
            ("id", safe_extract_field(sample, "id") or safe_extract_field(sample, "identifier"))
        ]
        
        # Default bbox coordinates (fallback if no real bbox data)
        default_bboxes = {
            "name": [[100, 50], [400, 50], [400, 100], [100, 100]],
            "date_of_birth": [[100, 120], [300, 120], [300, 170], [100, 170]],
            "gender": [[100, 190], [200, 190], [200, 240], [100, 240]],
            "id": [[100, 260], [350, 260], [350, 310], [100, 310]]
        }
        
        for field_name, field_value in text_fields:
            if field_value:
                conversion_stats["field_extractions"][field_name] += 1
                
                # Try to use real bbox data if available, otherwise use defaults
                if bbox_data and isinstance(bbox_data, (list, dict)):
                    # Handle different bbox data structures
                    if isinstance(bbox_data, list) and bbox_data:
                        points = bbox_data[0] if len(bbox_data) > 0 else default_bboxes[field_name]
                        conversion_stats["bbox_sources"]["real_data"] += 1
                    else:
                        points = default_bboxes[field_name]
                        conversion_stats["bbox_sources"]["synthetic"] += 1
                else:
                    points = default_bboxes[field_name]
                    conversion_stats["bbox_sources"]["synthetic"] += 1
                
                bbox = {
                    "points": points,
                    "transcription": str(field_value)
                }
                bboxes.append(bbox)
                
                # Add to recognition dataset
                rec_labels[split].append(f"{image_path}\t{field_value}")
        
        # Add to detection dataset
        if bboxes:
            det_labels[split].append(f"{image_path}\t{json.dumps(bboxes, ensure_ascii=False)}")
            stats["total_bboxes"] += len(bboxes)
        
        stats["total_images"] += 1
        stats[f"{split}_samples"] += 1
        
        if idx % 100 == 0:
            print(f"  Processed {idx}/{total_samples} images...")
    
    # Write label files
    for split in ["train", "val", "test"]:
        # Detection labels
        det_label_file = det_dir / split / "label.txt"
        with open(det_label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(det_labels[split]))
        print(f"  ✅ Detection {split}: {len(det_labels[split])} samples → {det_label_file}")
        
        # Recognition labels
        rec_label_file = rec_dir / split / "label.txt"
        with open(rec_label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(rec_labels[split]))
        print(f"  ✅ Recognition {split}: {len(rec_labels[split])} samples → {rec_label_file}")
    
    # Combine all statistics
    final_stats = {
        **stats,
        "schema_mapping": mapping_report,
        "conversion_stats": dict(conversion_stats),
        "data_quality": {
            "drop_rate": round(conversion_stats["total_dropped"] / conversion_stats["total_processed"] * 100, 2),
            "field_coverage": {field: count for field, count in conversion_stats["field_extractions"].items()},
            "bbox_quality": dict(conversion_stats["bbox_sources"])
        }
    }
    
    logging.info(f"📊 Conversion Summary:")
    logging.info(f"  Processed: {conversion_stats['total_processed']} samples")
    logging.info(f"  Dropped: {conversion_stats['total_dropped']} samples ({final_stats['data_quality']['drop_rate']}%)")
    logging.info(f"  Field extractions: {dict(conversion_stats['field_extractions'])}")
    logging.info(f"  Bbox sources: {dict(conversion_stats['bbox_sources'])}")
    
    return final_stats


def convert_khmerfonts_previews(
    input_dir: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict:
    """Convert khmerfonts-info-previews to recognition format with proper splitting.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        max_samples: Maximum number of samples to process
        split_ratio: Train/val/test split ratios
    
    Returns:
        Dictionary with conversion statistics and data quality metrics
    """
    if not HAS_DEPS:
        raise ImportError("Required dependencies not installed")
    
    print("🔄 Converting khmerfonts-info-previews dataset...")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Load dataset
    dataset = load_dataset("seanghay/khmerfonts-info-previews", cache_dir=str(input_dir))
    
    # Initialize schema discovery
    expected_fields = {"image", "text", "font", "sentence"}
    schema = DatasetSchema("khmerfonts-info-previews", expected_fields)
    
    if "train" not in dataset:
        logging.warning("No 'train' split found in dataset")
        return {"error": "No train split found"}
    
    data = dataset["train"]
    total_samples = len(data) if max_samples is None else min(max_samples, len(data))
    
    # Schema discovery on sample
    schema_samples = [data[i] for i in range(min(100, total_samples))]
    schema.discover_schema(schema_samples)
    
    # Log schema discovery results
    mapping_report = schema.get_mapping_report()
    logging.info(f"📋 Schema Discovery for {schema.dataset_name}:")
    logging.info(f"  Field coverage: {mapping_report['schema_match']['field_coverage']:.1f}%")
    logging.info(f"  Missing fields: {mapping_report['schema_match']['missing_fields']}")
    logging.info(f"  Extra fields: {mapping_report['schema_match']['extra_fields']}")
    
    # Collect unique sentences and fonts for stratified splitting
    sentence_font_pairs = set()
    unique_sentences = set()
    unique_fonts = set()
    
    for i in range(total_samples):
        sample = data[i]
        text = sample.get("text", "") or sample.get("sentence", "")
        font = sample.get("font", f"font_{i}")
        
        if text:
            unique_sentences.add(text)
            unique_fonts.add(font)
            sentence_font_pairs.add((text, font))
    
    logging.info(f"📊 Data Distribution:")
    logging.info(f"  Unique sentences: {len(unique_sentences)}")
    logging.info(f"  Unique fonts: {len(unique_fonts)}")
    logging.info(f"  Sentence-font pairs: {len(sentence_font_pairs)}")
    
    # Stratified splitting by sentences to avoid leakage
    sentences_list = list(unique_sentences)
    total_sent = len(sentences_list)
    
    train_sent_end = int(total_sent * split_ratio[0])
    val_sent_end = train_sent_end + int(total_sent * split_ratio[1])
    
    train_sentences = set(sentences_list[:train_sent_end])
    val_sentences = set(sentences_list[train_sent_end:val_sent_end])
    test_sentences = set(sentences_list[val_sent_end:])
    
    logging.info(f"🎯 Stratified Split by Sentences:")
    logging.info(f"  Train: {len(train_sentences)} sentences")
    logging.info(f"  Val: {len(val_sentences)} sentences")
    logging.info(f"  Test: {len(test_sentences)} sentences")
    
    # Create output directories for each split
    rec_dir = output_dir / "fonts_recognition"
    for split in ["train", "val", "test"]:
        (rec_dir / split / "images").mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_images": 0,
        "unique_sentences": len(unique_sentences),
        "unique_fonts": len(unique_fonts),
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0
    }
    
    rec_labels = {"train": [], "val": [], "test": []}
    conversion_stats = {
        "total_processed": 0,
        "total_dropped": 0,
        "drop_reasons": Counter(),
        "split_distribution": Counter()
    }
    
    for idx in range(total_samples):
        sample = data[idx]
        conversion_stats["total_processed"] += 1
        
        # Extract image and text with flexible field mapping
        image = sample.get("image")
        text = sample.get("text", "") or sample.get("sentence", "")
        font_name = sample.get("font", f"font_{idx}")
        
        if image is None or not text:
            conversion_stats["total_dropped"] += 1
            conversion_stats["drop_reasons"]["missing_data"] += 1
            continue
        
        # Determine split based on sentence (avoid leakage)
        if text in train_sentences:
            split = "train"
        elif text in val_sentences:
            split = "val"
        elif text in test_sentences:
            split = "test"
        else:
            # Fallback for edge cases
            split = "train"
        
        conversion_stats["split_distribution"][split] += 1
        
        # Save image to appropriate split
        image_name = f"font_{font_name}_{idx:06d}.jpg"
        image_path = Path(split) / "images" / image_name
        full_image_path = rec_dir / image_path
        
        if isinstance(image, Image.Image):
            image.save(full_image_path)
        
        # Add to split-specific labels
        rec_labels[split].append(f"{image_path}\t{text}")
        
        # Update statistics
        stats["total_images"] += 1
        stats[f"{split}_samples"] += 1
        
        if idx % 500 == 0:
            print(f"  Processed {idx}/{total_samples} images...")
    
    # Write label files for each split
    for split in ["train", "val", "test"]:
        label_file = rec_dir / split / "label.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(rec_labels[split]))
        print(f"  ✅ {split.title()} fonts: {len(rec_labels[split])} samples → {label_file}")
    
    # Final statistics with schema mapping
    final_stats = {
        **stats,
        "schema_mapping": mapping_report,
        "conversion_stats": dict(conversion_stats),
        "data_quality": {
            "drop_rate": round(conversion_stats["total_dropped"] / conversion_stats["total_processed"] * 100, 2),
            "split_distribution": dict(conversion_stats["split_distribution"]),
            "data_leakage_prevention": True,
            "stratified_by": "sentences"
        }
    }
    
    logging.info(f"📊 Font Conversion Summary:")
    logging.info(f"  Processed: {conversion_stats['total_processed']} samples")
    logging.info(f"  Dropped: {conversion_stats['total_dropped']} samples ({final_stats['data_quality']['drop_rate']}%)")
    logging.info(f"  Split distribution: {dict(conversion_stats['split_distribution'])}")
    logging.info(f"  📊 Unique sentences: {stats['unique_sentences']}")
    logging.info(f"  📊 Unique fonts: {stats['unique_fonts']}")
    
    return final_stats


def write_conversion_report(stats: Dict, output_dir: Path) -> None:
    """Write conversion statistics to a report file."""
    report_path = output_dir / "conversion_report.json"
    
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📄 Conversion report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HF datasets to PaddleOCR format")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/hf_datasets"),
        help="Input directory with downloaded HF datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/paddle_format"),
        help="Output directory for PaddleOCR format data"
    )
    parser.add_argument(
        "--dataset",
        choices=["synthkhmer", "fonts", "all"],
        default="all",
        help="Which dataset to convert"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratios (must sum to 1.0)"
    )
    parser.add_argument(
        "--max-font-samples",
        type=int,
        default=None,
        help="Maximum number of font samples to process"
    )
    
    args = parser.parse_args()
    
    if not HAS_DEPS:
        print("Error: Required packages not installed.")
        print("Run: pip install datasets pillow numpy")
        return 1
    
    # Validate split ratios
    if abs(sum(args.split_ratio) - 1.0) > 0.001:
        print("Error: Split ratios must sum to 1.0")
        return 1
    
    # Check if input directory exists and has data
    if not args.input_dir.exists():
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return 1
    
    # Check for any datasets
    dataset_dirs = list(args.input_dir.iterdir())
    if not dataset_dirs:
        print(f"❌ No datasets found in {args.input_dir}")
        print("Run: python ops/download_hf.py first")
        return 1
    
    print(f"Found {len(dataset_dirs)} potential datasets in {args.input_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    failed_conversions = []
    
    # Convert SynthKhmer-10k
    if args.dataset in ["synthkhmer", "all"]:
        synthkhmer_dir = args.input_dir / "SynthKhmer-10k"
        if synthkhmer_dir.exists():
            try:
                stats = convert_synthkhmer_10k(
                    synthkhmer_dir,
                    args.output_dir,
                    tuple(args.split_ratio)
                )
                all_stats["synthkhmer_10k"] = stats
            except Exception as e:
                print(f"❌ Failed to convert SynthKhmer-10k: {e}")
                failed_conversions.append("SynthKhmer-10k")
        else:
            print(f"⚠️  SynthKhmer-10k not found in {args.input_dir}")
    
    # Convert khmerfonts-info-previews
    if args.dataset in ["fonts", "all"]:
        fonts_dir = args.input_dir / "khmerfonts-info-previews"
        if fonts_dir.exists():
            try:
                stats = convert_khmerfonts_previews(
                    fonts_dir,
                    args.output_dir,
                    args.max_font_samples
                )
                all_stats["khmerfonts_previews"] = stats
            except Exception as e:
                print(f"❌ Failed to convert khmerfonts-info-previews: {e}")
                failed_conversions.append("khmerfonts-info-previews")
        else:
            print(f"⚠️  khmerfonts-info-previews not found in {args.input_dir}")
    
    # Check if any conversions succeeded
    if not all_stats:
        print("\n❌ No datasets were converted successfully!")
        if failed_conversions:
            print(f"Failed conversions: {', '.join(failed_conversions)}")
        return 1
    
    # Write report
    write_conversion_report(all_stats, args.output_dir)
    
    print(f"\n✅ Conversion complete! Converted {len(all_stats)} dataset(s)")
    if failed_conversions:
        print(f"⚠️  Failed: {', '.join(failed_conversions)}")
    
    return 0


if __name__ == "__main__":
    exit(main())