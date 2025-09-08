#!/usr/bin/env python3
"""Download Khmer OCR datasets from seanghay's HuggingFace profile.

This script downloads and prepares the following datasets:
1. SynthKhmer-10k: Synthetic Khmer ID documents with annotations
2. khmerfonts-info-previews: Font variation images
3. khmer-dictionary-44k: Dictionary entries for language modeling
4. google-khmer-lexicon: Vocabulary with pronunciations
5. lexicon-kh: Additional vocabulary entries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("Warning: huggingface_hub not installed. Install with: pip install datasets huggingface_hub")


@dataclass
class DatasetConfig:
    """Configuration for a HuggingFace dataset."""
    name: str
    repo_id: str
    dataset_type: str  # "ocr_images", "lexicon", "text_corpus"
    description: str
    priority: str  # "high", "medium", "low"
    expected_size: Optional[str] = None
    license: Optional[str] = None


# Define all seanghay datasets
SEANGHAY_DATASETS = [
    DatasetConfig(
        name="SynthKhmer-10k",
        repo_id="seanghay/SynthKhmer-10k",
        dataset_type="ocr_images",
        description="10k synthetic Khmer ID documents with bounding box annotations",
        priority="high",
        expected_size="10000 images",
        license="CC-BY-4.0"
    ),
    DatasetConfig(
        name="khmerfonts-info-previews",
        repo_id="seanghay/khmerfonts-info-previews",
        dataset_type="ocr_images",
        description="26.6k font preview images (2,972 fonts × 10 sentences)",
        priority="high",
        expected_size="26591 images",
        license="CC-BY-4.0"
    ),
    DatasetConfig(
        name="khmer-dictionary-44k",
        repo_id="seanghay/khmer-dictionary-44k",
        dataset_type="lexicon",
        description="44.7k dictionary entries from Royal Academy of Cambodia",
        priority="high",
        expected_size="44706 entries",
        license="Research use only"
    ),
    DatasetConfig(
        name="google-khmer-lexicon",
        repo_id="seanghay/google-khmer-lexicon",
        dataset_type="lexicon",
        description="69.4k word entries with pronunciations",
        priority="medium",
        expected_size="69414 entries",
        license=None
    ),
    DatasetConfig(
        name="lexicon-kh",
        repo_id="seanghay/lexicon-kh",
        dataset_type="lexicon",
        description="3.77k additional vocabulary entries",
        priority="low",
        expected_size="3770 entries",
        license=None
    ),
    DatasetConfig(
        name="km_large_text",
        repo_id="seanghay/km_large_text",
        dataset_type="text_corpus",
        description="Large text corpus for language modeling",
        priority="medium",
        expected_size=None,
        license=None
    )
]


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_dataset(config: DatasetConfig, output_dir: Path, use_cache: bool = True) -> Dict:
    """Download a single dataset and return metadata."""
    if not HAS_HF:
        raise ImportError("Please install huggingface_hub and datasets packages")
    
    dataset_dir = output_dir / config.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "name": config.name,
        "repo_id": config.repo_id,
        "type": config.dataset_type,
        "description": config.description,
        "priority": config.priority,
        "license": config.license,
        "local_path": str(dataset_dir),
        "status": "pending"
    }
    
    try:
        print(f"\n📥 Downloading {config.name} from {config.repo_id}...")
        
        # Use datasets library for structured data
        if config.dataset_type in ["ocr_images", "lexicon"]:
            dataset = load_dataset(config.repo_id, cache_dir=str(dataset_dir) if use_cache else None)
            
            # Save dataset info
            info_file = dataset_dir / "dataset_info.json"
            if hasattr(dataset, "info"):
                with open(info_file, "w") as f:
                    json.dump({
                        "features": str(dataset.info.features),
                        "num_rows": dataset.num_rows if hasattr(dataset, "num_rows") else None,
                        "splits": list(dataset.keys()) if hasattr(dataset, "keys") else []
                    }, f, indent=2)
            
            # Save sample data for inspection
            if "train" in dataset:
                sample_file = dataset_dir / "sample.json"
                samples = dataset["train"][:5] if len(dataset["train"]) >= 5 else dataset["train"][:]
                with open(sample_file, "w") as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
            
            metadata["status"] = "downloaded"
            metadata["num_samples"] = len(dataset["train"]) if "train" in dataset else None
            
        # Use snapshot_download for raw files
        else:
            snapshot_path = snapshot_download(
                repo_id=config.repo_id,
                repo_type="dataset",
                local_dir=str(dataset_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.git*", "*.gitattributes"]
            )
            metadata["status"] = "downloaded"
            metadata["snapshot_path"] = snapshot_path
        
        print(f"✅ Downloaded {config.name} to {dataset_dir}")
        
    except Exception as e:
        print(f"❌ Failed to download {config.name}: {e}")
        metadata["status"] = "failed"
        metadata["error"] = str(e)
    
    return metadata


def create_manifest(datasets_metadata: List[Dict], output_dir: Path) -> None:
    """Create a manifest file with all dataset information."""
    manifest_path = output_dir / "hf_datasets_manifest.json"
    
    manifest = {
        "source": "seanghay HuggingFace Profile",
        "datasets": datasets_metadata,
        "total_datasets": len(datasets_metadata),
        "downloaded": sum(1 for d in datasets_metadata if d["status"] == "downloaded"),
        "failed": sum(1 for d in datasets_metadata if d["status"] == "failed")
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Khmer OCR datasets from seanghay's HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hf_datasets"),
        help="Output directory for downloaded datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[d.name for d in SEANGHAY_DATASETS] + ["all"],
        default=["all"],
        help="Specific datasets to download or 'all'"
    )
    parser.add_argument(
        "--priority",
        choices=["high", "medium", "low", "all"],
        default="high",
        help="Download datasets by priority level"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached downloads"
    )
    
    args = parser.parse_args()
    
    if not HAS_HF:
        print("Error: Required packages not installed.")
        print("Run: pip install datasets huggingface_hub")
        return 1
    
    # Select datasets to download
    datasets_to_download = []
    
    if "all" in args.datasets:
        datasets_to_download = SEANGHAY_DATASETS
    else:
        datasets_to_download = [d for d in SEANGHAY_DATASETS if d.name in args.datasets]
    
    # Filter by priority if specified
    if args.priority != "all":
        datasets_to_download = [d for d in datasets_to_download if d.priority == args.priority]
    
    if not datasets_to_download:
        print("No datasets selected for download.")
        return 0
    
    print(f"🎯 Planning to download {len(datasets_to_download)} datasets:")
    for ds in datasets_to_download:
        print(f"  - {ds.name} ({ds.priority} priority)")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    all_metadata = []
    for config in datasets_to_download:
        metadata = download_dataset(config, args.output_dir, use_cache=not args.no_cache)
        all_metadata.append(metadata)
    
    # Create manifest
    create_manifest(all_metadata, args.output_dir)
    
    # Summary
    print("\n" + "="*50)
    print("📊 Download Summary:")
    print(f"  Total: {len(all_metadata)}")
    print(f"  ✅ Success: {sum(1 for m in all_metadata if m['status'] == 'downloaded')}")
    print(f"  ❌ Failed: {sum(1 for m in all_metadata if m['status'] == 'failed')}")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    exit(main())