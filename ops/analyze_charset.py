#!/usr/bin/env python3
"""Analyze charset coverage in training datasets"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Set, List


def load_charset(charset_path: Path) -> Set[str]:
    """Load charset from file"""
    with open(charset_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def analyze_label_file(label_path: Path) -> Set[str]:
    """Extract all characters from a label file"""
    chars = set()
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                _, text = line.strip().split('\t', 1)
                chars.update(text)
    return chars


def main():
    parser = argparse.ArgumentParser(description="Analyze charset coverage")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--charset", type=Path, default=Path("train/charset_kh.txt"), help="Charset file")
    parser.add_argument("--output", type=Path, help="Output JSON report")
    
    args = parser.parse_args()
    
    # Load reference charset
    charset = load_charset(args.charset)
    print(f"Reference charset: {len(charset)} characters")
    
    # Analyze each dataset
    results = {}
    total_coverage = set()
    
    for subset in ["train", "val", "test", "synth"]:
        label_file = args.data_dir / subset / "label.txt"
        if label_file.exists():
            chars_found = analyze_label_file(label_file)
            total_coverage.update(chars_found)
            
            coverage_pct = len(chars_found & charset) / len(charset) * 100
            missing_chars = charset - chars_found
            extra_chars = chars_found - charset
            
            results[subset] = {
                "chars_found": len(chars_found),
                "charset_coverage": len(chars_found & charset),
                "coverage_pct": coverage_pct,
                "missing_count": len(missing_chars),
                "extra_count": len(extra_chars),
                "missing_chars": list(missing_chars)[:20],  # First 20 missing
                "extra_chars": list(extra_chars)[:20]      # First 20 extra
            }
            
            print(f"\n{subset.upper()}:")
            print(f"  Characters found: {len(chars_found)}")
            print(f"  Charset coverage: {len(chars_found & charset)}/{len(charset)} ({coverage_pct:.1f}%)")
            print(f"  Missing chars: {len(missing_chars)}")
            if missing_chars:
                missing_sample = list(missing_chars)[:10]
                print(f"    Sample: {missing_sample}")
    
    # Overall analysis
    total_coverage_pct = len(total_coverage & charset) / len(charset) * 100
    missing_overall = charset - total_coverage
    
    results["overall"] = {
        "total_chars": len(total_coverage),
        "charset_coverage": len(total_coverage & charset), 
        "coverage_pct": total_coverage_pct,
        "missing_count": len(missing_overall),
        "missing_chars": list(missing_overall)
    }
    
    print(f"\nOVERALL:")
    print(f"  Total unique chars: {len(total_coverage)}")
    print(f"  Charset coverage: {len(total_coverage & charset)}/{len(charset)} ({total_coverage_pct:.1f}%)")
    print(f"  Missing chars: {len(missing_overall)}")
    if missing_overall:
        print(f"    Missing: {list(missing_overall)[:20]}")
    
    # Acceptance criteria
    accept_threshold = 99.0
    results["acceptance"] = {
        "threshold_pct": accept_threshold,
        "passes": total_coverage_pct >= accept_threshold,
        "status": "PASS" if total_coverage_pct >= accept_threshold else "FAIL"
    }
    
    print(f"\nACCEPTANCE: {'✅ PASS' if total_coverage_pct >= accept_threshold else '❌ FAIL'}")
    print(f"Target: ≥{accept_threshold}%, Actual: {total_coverage_pct:.1f}%")
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()