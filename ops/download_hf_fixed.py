#!/usr/bin/env python3
"""Fixed HuggingFace downloader - no manual URL construction, proper API usage"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("Error: Required packages not installed")
    print("Run: pip install datasets huggingface_hub pillow")
    sys.exit(1)


def download_ocr_dataset(repo_id, output_dir, dataset_name):
    """Download OCR image dataset and save images + labels"""
    print(f"\n📥 Downloading {dataset_name} from {repo_id}...")
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset using HF API
        dataset = load_dataset(repo_id)
        
        # Process each split
        for split_name in dataset.keys():
            print(f"  Processing split: {split_name}")
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create label file
            labels = []
            
            # Iterate over rows and save images/text
            for idx, row in enumerate(dataset[split_name]):
                # Handle different column names
                # Try common image field names
                image = None
                for img_field in ['image', 'img', 'pixel_values', 'image_path']:
                    if img_field in row:
                        image = row[img_field]
                        break
                
                # Try common text field names
                text = None
                for text_field in ['text', 'sentence', 'label', 'transcription', 'caption']:
                    if text_field in row:
                        text = row[text_field]
                        break
                
                if image is not None:
                    # Save image
                    img_filename = f"{dataset_name}_{split_name}_{idx:06d}.png"
                    img_path = split_dir / img_filename
                    
                    if isinstance(image, Image.Image):
                        image.save(img_path)
                    else:
                        print(f"    Warning: Unknown image type for row {idx}")
                        continue
                    
                    # Save text if available
                    if text:
                        txt_path = img_path.with_suffix('.txt')
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(str(text))
                        
                        # Add to labels
                        labels.append(f"{img_filename}\t{text}")
                    else:
                        print(f"    Warning: No text for image {idx}")
            
            # Save label file
            if labels:
                label_file = split_dir / "labels.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(labels))
                print(f"  ✅ Saved {len(labels)} samples to {split_dir}")
            else:
                print(f"  ⚠️  No valid samples in {split_name}")
        
        print(f"✅ Downloaded {dataset_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e}")
        return False


def download_lexicon_dataset(repo_id, output_dir, dataset_name):
    """Download lexicon/dictionary dataset"""
    print(f"\n📥 Downloading lexicon {dataset_name} from {repo_id}...")
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        dataset = load_dataset(repo_id)
        
        all_words = []
        
        # Process all splits
        for split_name in dataset.keys():
            print(f"  Processing split: {split_name}")
            
            for row in dataset[split_name]:
                # Extract word/lexicon entries
                # Try different field names
                word = None
                for field in ['word', 'text', 'entry', 'term', 'khmer', 'km']:
                    if field in row:
                        word = row[field]
                        break
                
                if word:
                    all_words.append(str(word))
        
        if all_words:
            # Save lexicon
            lexicon_file = output_path / "lexicon.txt"
            with open(lexicon_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sorted(set(all_words))))
            
            print(f"✅ Saved {len(set(all_words))} unique words to {lexicon_file}")
            return True
        else:
            print(f"⚠️  No words found in {dataset_name}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e}")
        return False


def download_with_snapshot(repo_id, output_dir, dataset_name):
    """Fallback: Download using snapshot_download for raw files"""
    print(f"\n📥 Downloading {dataset_name} using snapshot...")
    
    output_path = Path(output_dir) / dataset_name
    
    try:
        # Download entire repository
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_path),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Downloaded snapshot to {snapshot_path}")
        
        # List what we got
        files = list(Path(snapshot_path).rglob("*"))
        print(f"  Found {len(files)} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument("--output-dir", default="data/hf_datasets", help="Output directory")
    parser.add_argument("--priority", choices=["high", "medium", "low", "all"], default="high")
    args = parser.parse_args()
    
    if not HAS_HF:
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define datasets to download
    high_priority = [
        ("seanghay/SynthKhmer-10k", "ocr", "SynthKhmer-10k"),
        ("seanghay/khmer-dictionary-44k", "lexicon", "khmer-dictionary-44k"),
    ]
    
    medium_priority = [
        ("seanghay/khmerfonts-info-previews", "ocr", "khmerfonts-info-previews"),
        ("seanghay/google-khmer-lexicon", "lexicon", "google-khmer-lexicon"),
    ]
    
    low_priority = [
        ("seanghay/lexicon-kh", "lexicon", "lexicon-kh"),
    ]
    
    # Select datasets based on priority
    datasets = []
    if args.priority in ["high", "all"]:
        datasets.extend(high_priority)
    if args.priority in ["medium", "all"]:
        datasets.extend(medium_priority)
    if args.priority in ["low", "all"]:
        datasets.extend(low_priority)
    
    if not datasets:
        print("No datasets to download")
        return 0
    
    print(f"📋 Planning to download {len(datasets)} datasets")
    
    success_count = 0
    failed_count = 0
    
    for repo_id, dataset_type, name in datasets:
        if dataset_type == "ocr":
            success = download_ocr_dataset(repo_id, output_dir, name)
        elif dataset_type == "lexicon":
            success = download_lexicon_dataset(repo_id, output_dir, name)
        else:
            # Fallback to snapshot download
            success = download_with_snapshot(repo_id, output_dir, name)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Downloaded: {success_count}/{len(datasets)}")
    print(f"Failed: {failed_count}/{len(datasets)}")
    
    # Fail if all high priority failed
    if args.priority == "high" and success_count == 0:
        print("❌ All high-priority downloads failed!")
        return 1
    
    # Success if at least one downloaded
    if success_count > 0:
        print(f"✅ Downloaded {success_count} datasets to {output_dir}")
        return 0
    else:
        print("❌ No datasets downloaded successfully")
        return 1


if __name__ == "__main__":
    sys.exit(main())