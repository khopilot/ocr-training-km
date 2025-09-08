#!/usr/bin/env python3
"""Split dataset into train/val/test sets.

Reads a PaddleOCR-style label file from data/raw/label.txt and writes
split label.txt files into data/train, data/val, data/test based on the
provided ratio. If no raw label exists, creates placeholder files.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def read_labels(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                img, text = line.rstrip("\n").split("\t", 1)
                pairs.append((img, text))
    return pairs


def write_labels(path: Path, pairs: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for img, text in pairs:
            f.write(f"{img}\t{text}\n")


def split_triplet(pairs: List[Tuple[str, str]], ratios: Tuple[float, float, float]) -> Tuple[List, List, List]:
    n = len(pairs)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--ratio", type=str, default="0.8:0.1:0.1", help="split ratio")
    parser.add_argument("--raw", type=str, default="data/raw/label.txt", help="raw label path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_tokens = args.ratio.split(":")
    if len(ratio_tokens) != 3:
        raise ValueError("ratio must be in the form a:b:c, e.g., 0.8:0.1:0.1")
    ratios = tuple(float(x) for x in ratio_tokens)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    raw_path = Path(args.raw)
    if raw_path.exists():
        pairs = read_labels(raw_path)
        random.Random(args.seed).shuffle(pairs)
        train_pairs, val_pairs, test_pairs = split_triplet(pairs, ratios)
        write_labels(Path("data/train/label.txt"), train_pairs)
        write_labels(Path("data/val/label.txt"), val_pairs)
        write_labels(Path("data/test/label.txt"), test_pairs)
        print(f"✓ Wrote splits: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    else:
        # Placeholder structure
        for d in ["data/train", "data/val", "data/test"]:
            Path(d).mkdir(parents=True, exist_ok=True)
            write_labels(Path(d) / "label.txt", [])
        print("⚠️ No raw label file found; created empty splits")


if __name__ == "__main__":
    main()

