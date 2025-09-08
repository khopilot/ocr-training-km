#!/usr/bin/env python3
"""Validate OCR dataset format and charset coverage"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import cv2


def load_charset(charset_path: Path) -> Set[str]:
    """Load character set from file"""
    charset = set()
    with open(charset_path, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char and char != "\\n":
                charset.add(char)
            elif char == "\\n":
                charset.add("\n")
    return charset


def validate_rec_label(label_path: Path) -> Tuple[List[str], List[str]]:
    """
    Validate recognition label format: img_path<TAB>text
    
    Returns:
        Tuple of (valid_lines, errors)
    """
    valid_lines = []
    errors = []
    
    if not label_path.exists():
        errors.append(f"Label file not found: {label_path}")
        return valid_lines, errors
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                errors.append(f"Line {line_num}: Expected format 'img_path<TAB>text', got {len(parts)} parts")
                continue
                
            img_path, text = parts
            if not img_path:
                errors.append(f"Line {line_num}: Empty image path")
                continue
            if not text:
                errors.append(f"Line {line_num}: Empty text label")
                continue
                
            valid_lines.append(line)
    
    return valid_lines, errors


def validate_det_label(label_path: Path) -> Tuple[List[str], List[str]]:
    """
    Validate detection label format: img_path<TAB>[{"transcription":"...", "points":[[x,y],...]}]
    
    Returns:
        Tuple of (valid_lines, errors)
    """
    valid_lines = []
    errors = []
    
    if not label_path.exists():
        return valid_lines, [f"Label file not found: {label_path}"]
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                errors.append(f"Line {line_num}: Expected format 'img_path<TAB>json_array'")
                continue
                
            img_path, json_str = parts
            if not img_path:
                errors.append(f"Line {line_num}: Empty image path")
                continue
                
            try:
                annotations = json.loads(json_str)
                if not isinstance(annotations, list):
                    errors.append(f"Line {line_num}: JSON should be a list")
                    continue
                    
                for ann in annotations:
                    if 'transcription' not in ann:
                        errors.append(f"Line {line_num}: Missing 'transcription' field")
                    if 'points' not in ann:
                        errors.append(f"Line {line_num}: Missing 'points' field")
                        
                valid_lines.append(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
    
    return valid_lines, errors


def check_image_files(label_path: Path, valid_lines: List[str]) -> Tuple[int, List[str]]:
    """
    Check if image files exist and are readable
    
    Returns:
        Tuple of (valid_count, errors)
    """
    valid_count = 0
    errors = []
    base_dir = label_path.parent
    
    for line in valid_lines:
        img_path_str = line.split('\t')[0]
        img_path = base_dir / img_path_str
        
        if not img_path.exists():
            errors.append(f"Image not found: {img_path_str}")
            continue
            
        try:
            # Try to read image to verify it's valid
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Cannot read image: {img_path_str}")
            else:
                valid_count += 1
        except Exception as e:
            errors.append(f"Error reading {img_path_str}: {e}")
    
    return valid_count, errors


def calculate_charset_coverage(
    valid_lines: List[str],
    charset: Set[str],
    label_type: str = "rec"
) -> Dict[str, float]:
    """
    Calculate charset coverage statistics
    
    Returns:
        Dictionary with coverage metrics
    """
    all_chars = Counter()
    
    for line in valid_lines:
        if label_type == "rec":
            text = line.split('\t')[1]
            all_chars.update(text)
        elif label_type == "det":
            json_str = line.split('\t')[1]
            annotations = json.loads(json_str)
            for ann in annotations:
                text = ann.get('transcription', '')
                all_chars.update(text)
    
    total_chars = sum(all_chars.values())
    in_charset = 0
    out_charset = []
    
    for char, count in all_chars.items():
        if char in charset or char in [' ', '\t', '\n']:
            in_charset += count
        else:
            out_charset.append((char, count))
    
    coverage = (in_charset / total_chars * 100) if total_chars > 0 else 0
    
    return {
        'coverage_percent': coverage,
        'total_unique_chars': len(all_chars),
        'total_char_instances': total_chars,
        'in_charset_instances': in_charset,
        'out_of_charset': sorted(out_charset, key=lambda x: x[1], reverse=True)[:10]
    }


def validate_dataset(
    data_dir: Path,
    charset_path: Path,
    label_type: str = "rec",
    min_coverage: float = 99.0
) -> bool:
    """
    Validate complete dataset
    
    Returns:
        True if validation passes
    """
    print(f"\n{'='*60}")
    print(f"Dataset Validation Report")
    print(f"{'='*60}")
    print(f"Directory: {data_dir}")
    print(f"Label type: {label_type}")
    print(f"Charset: {charset_path}")
    
    # Load charset
    charset = load_charset(charset_path)
    print(f"\n✓ Loaded charset with {len(charset)} characters")
    
    all_valid = True
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"\n⚠️  {split} directory not found")
            continue
            
        label_file = split_dir / 'label.txt'
        if label_type == "det":
            label_file = split_dir / 'det_label.txt'
            
        print(f"\n📁 {split.upper()} Split")
        print(f"   Label file: {label_file.name}")
        
        # Validate label format
        if label_type == "rec":
            valid_lines, format_errors = validate_rec_label(label_file)
        else:
            valid_lines, format_errors = validate_det_label(label_file)
            
        print(f"   Valid lines: {len(valid_lines)}")
        if format_errors:
            print(f"   ❌ Format errors: {len(format_errors)}")
            for error in format_errors[:5]:
                print(f"      - {error}")
            if len(format_errors) > 5:
                print(f"      ... and {len(format_errors)-5} more")
            all_valid = False
        
        # Check images
        valid_images, img_errors = check_image_files(label_file, valid_lines)
        print(f"   Valid images: {valid_images}/{len(valid_lines)}")
        if img_errors:
            print(f"   ❌ Image errors: {len(img_errors)}")
            for error in img_errors[:3]:
                print(f"      - {error}")
            if len(img_errors) > 3:
                print(f"      ... and {len(img_errors)-3} more")
        
        # Calculate charset coverage
        if valid_lines:
            coverage = calculate_charset_coverage(valid_lines, charset, label_type)
            print(f"   Charset coverage: {coverage['coverage_percent']:.2f}%")
            
            if coverage['coverage_percent'] < min_coverage:
                print(f"   ❌ Coverage below {min_coverage}% threshold!")
                if coverage['out_of_charset']:
                    print(f"   Out-of-charset characters (top 10):")
                    for char, count in coverage['out_of_charset']:
                        print(f"      '{char}' (U+{ord(char):04X}): {count} instances")
                all_valid = False
            else:
                print(f"   ✅ Coverage meets {min_coverage}% threshold")
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ Dataset validation PASSED")
    else:
        print("❌ Dataset validation FAILED")
    print(f"{'='*60}\n")
    
    return all_valid


def main():
    """CLI interface for dataset validation"""
    parser = argparse.ArgumentParser(description="Validate OCR dataset")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Data directory containing train/val/test splits'
    )
    parser.add_argument(
        '--charset',
        type=str,
        default='train/charset_kh.txt',
        help='Path to charset file'
    )
    parser.add_argument(
        '--type',
        choices=['rec', 'det'],
        default='rec',
        help='Label type: recognition or detection'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=99.0,
        help='Minimum charset coverage percentage'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with error code if validation fails'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    charset_path = Path(args.charset)
    
    if not charset_path.exists():
        print(f"❌ Charset file not found: {charset_path}")
        sys.exit(1)
    
    # Run validation
    valid = validate_dataset(
        data_dir,
        charset_path,
        args.type,
        args.min_coverage
    )
    
    if args.strict and not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()