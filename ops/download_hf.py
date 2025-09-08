#!/usr/bin/env python3
"""HuggingFace downloader using ONLY official APIs - no manual URL construction"""

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


def download_synthkhmer_10k(output_dir):
    """Download SynthKhmer-10k OCR dataset"""
    print("\n📥 Downloading SynthKhmer-10k...")
    
    output_path = Path(output_dir) / "SynthKhmer-10k"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset using HF API - NO manual URLs!
        dataset = load_dataset("seanghay/SynthKhmer-10k")
        
        total_samples = 0
        
        # Process each split
        for split_name in dataset.keys():
            print(f"  Processing split: {split_name}")
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create label file
            labels = []
            valid_samples = 0
            
            # Print schema for debugging
            if len(dataset[split_name]) > 0:
                print(f"  Schema: {list(dataset[split_name][0].keys())}")
            
            # Iterate over rows - NO URL construction!
            for idx, row in enumerate(dataset[split_name]):
                # Try ALL possible field names for SynthKhmer
                text = None
                text_fields = ['text', 'label', 'sentence', 'kh_text', 'khmer', 
                              'name', 'id', 'date_of_birth', 'gender', 'address',
                              'transcription', 'caption']
                
                for field in text_fields:
                    if field in row and row[field]:
                        text = str(row[field]).strip()
                        if text:  # Non-empty text
                            break
                
                # Get image
                image = None
                if 'image' in row and row['image'] is not None:
                    image = row['image']
                elif 'img' in row and row['img'] is not None:
                    image = row['img']
                
                # Save if we have both image and text
                if image is not None and text:
                    img_filename = f"synth_{split_name}_{idx:06d}.png"
                    img_path = split_dir / img_filename
                    
                    try:
                        if isinstance(image, Image.Image):
                            image.save(img_path)
                        elif hasattr(image, 'save'):
                            image.save(img_path)
                        else:
                            print(f"    Warning: Unknown image type at index {idx}")
                            continue
                        
                        # Save text file
                        txt_path = img_path.with_suffix('.txt')
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        # Add to labels
                        labels.append(f"{img_filename}\t{text}")
                        valid_samples += 1
                        
                    except Exception as e:
                        print(f"    Error saving image {idx}: {e}")
                        continue
                else:
                    if not text:
                        print(f"    Warning: No text for image {idx}")
                    if image is None:
                        print(f"    Warning: No image at index {idx}")
            
            # Save label file if we have valid samples
            if labels:
                label_file = split_dir / "labels.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(labels))
                print(f"  ✅ {split_name}: {valid_samples} valid samples")
                total_samples += valid_samples
            else:
                print(f"  ⚠️  {split_name}: No valid samples found!")
        
        # Check if we got any valid data
        if total_samples > 0:
            print(f"✅ Downloaded SynthKhmer-10k: {total_samples} total samples")
            return True
        else:
            print(f"❌ Failed: SynthKhmer-10k has 0 valid samples")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download SynthKhmer-10k: {e}")
        return False


def download_khmerfonts_previews(output_dir):
    """Download khmerfonts-info-previews dataset"""
    print("\n📥 Downloading khmerfonts-info-previews...")
    
    output_path = Path(output_dir) / "khmerfonts-info-previews"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset - NO manual URLs!
        dataset = load_dataset("seanghay/khmerfonts-info-previews")
        
        total_samples = 0
        
        for split_name in dataset.keys():
            print(f"  Processing split: {split_name}")
            split_dir = output_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            labels = []
            valid_samples = 0
            
            # Print schema
            if len(dataset[split_name]) > 0:
                print(f"  Schema: {list(dataset[split_name][0].keys())}")
            
            for idx, row in enumerate(dataset[split_name]):
                # Get text - try multiple fields
                text = None
                for field in ['text', 'sentence', 'label']:
                    if field in row and row[field]:
                        text = str(row[field]).strip()
                        if text:
                            break
                
                # Get font info if available
                font = row.get('font', f'font_{idx}')
                
                # Get image
                image = row.get('image') or row.get('img')
                
                if image is not None and text:
                    img_filename = f"font_{idx:06d}.png"
                    img_path = split_dir / img_filename
                    
                    try:
                        if isinstance(image, Image.Image):
                            image.save(img_path)
                            
                            # Save text
                            txt_path = img_path.with_suffix('.txt')
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(text)
                            
                            labels.append(f"{img_filename}\t{text}")
                            valid_samples += 1
                    except Exception as e:
                        print(f"    Error saving image {idx}: {e}")
            
            if labels:
                label_file = split_dir / "labels.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(labels))
                print(f"  ✅ {split_name}: {valid_samples} valid samples")
                total_samples += valid_samples
            else:
                print(f"  ⚠️  {split_name}: No valid samples")
        
        if total_samples > 0:
            print(f"✅ Downloaded khmerfonts: {total_samples} total samples")
            return True
        else:
            print(f"❌ Failed: khmerfonts has 0 valid samples")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download khmerfonts: {e}")
        return False


def download_khmer_dictionary(output_dir):
    """Download khmer-dictionary-44k lexicon"""
    print("\n📥 Downloading khmer-dictionary-44k...")
    
    output_path = Path(output_dir) / "khmer-dictionary-44k"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset - NO manual URLs!
        dataset = load_dataset("seanghay/khmer-dictionary-44k")
        
        all_words = set()
        
        for split_name in dataset.keys():
            print(f"  Processing split: {split_name}")
            
            # Print schema
            if len(dataset[split_name]) > 0:
                print(f"  Schema: {list(dataset[split_name][0].keys())}")
            
            for row in dataset[split_name]:
                # Try multiple field names for words
                word = None
                for field in ['word', 'text', 'entry', 'term', 'khmer', 'km', 'headword']:
                    if field in row and row[field]:
                        word = str(row[field]).strip()
                        if word:
                            all_words.add(word)
                            break
        
        if all_words:
            lexicon_file = output_path / "lexicon.txt"
            with open(lexicon_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sorted(all_words)))
            print(f"✅ Saved {len(all_words)} unique words to lexicon")
            return True
        else:
            print(f"❌ No words found in dictionary")
            return False
            
    except Exception as e:
        print(f"❌ Failed to download dictionary: {e}")
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
    
    success_count = 0
    failed_count = 0
    high_priority_success = False
    
    # High priority datasets
    if args.priority in ["high", "all"]:
        # SynthKhmer-10k (critical for training)
        if download_synthkhmer_10k(output_dir):
            success_count += 1
            high_priority_success = True
        else:
            failed_count += 1
        
        # Dictionary for lexicon
        if download_khmer_dictionary(output_dir):
            success_count += 1
        else:
            failed_count += 1
    
    # Medium priority
    if args.priority in ["medium", "all"]:
        # Font previews
        if download_khmerfonts_previews(output_dir):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Downloaded: {success_count} datasets")
    print(f"Failed: {failed_count} datasets")
    
    # Fail if high priority failed
    if args.priority == "high" and not high_priority_success:
        print("❌ Critical dataset (SynthKhmer-10k) failed!")
        return 1
    
    if success_count > 0:
        print(f"✅ Successfully downloaded {success_count} dataset(s)")
        return 0
    else:
        print("❌ All downloads failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())