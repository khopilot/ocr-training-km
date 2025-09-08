#!/usr/bin/env python3
"""Convert HF datasets to PaddleOCR format - NO manual URLs, proper schema mapping"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("Error: datasets not installed")
    sys.exit(1)


def convert_synthkhmer_from_downloaded(input_dir, output_dir):
    """Convert downloaded SynthKhmer-10k to PaddleOCR format"""
    print("\n🔄 Converting SynthKhmer-10k from downloaded files...")
    
    synthkhmer_dir = Path(input_dir) / "SynthKhmer-10k"
    if not synthkhmer_dir.exists():
        print(f"❌ SynthKhmer-10k not found in {input_dir}")
        return False
    
    # Create output directories
    rec_dir = Path(output_dir) / "recognition"
    for split in ["train", "val", "test"]:
        (rec_dir / split).mkdir(parents=True, exist_ok=True)
    
    total_converted = 0
    
    # Process each split
    for split_name in ["train", "val", "test"]:
        split_dir = synthkhmer_dir / split_name
        if not split_dir.exists():
            # Try alternative name
            split_dir = synthkhmer_dir / "train" if split_name != "train" else None
            if not split_dir or not split_dir.exists():
                print(f"  ⚠️  Split {split_name} not found")
                continue
        
        # Find images and texts
        images = sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg"))
        
        labels = []
        valid_count = 0
        
        for img_path in images:
            # Check for corresponding text file
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text:
                    # Copy image to output
                    out_split = split_name if split_name in ["train", "val", "test"] else "train"
                    out_img = rec_dir / out_split / img_path.name
                    
                    try:
                        Image.open(img_path).save(out_img)
                        labels.append(f"{img_path.name}\t{text}")
                        valid_count += 1
                    except Exception as e:
                        print(f"    Error copying {img_path.name}: {e}")
        
        # Save labels
        if labels:
            out_split = split_name if split_name in ["train", "val", "test"] else "train"
            label_file = rec_dir / out_split / "label.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(labels))
            print(f"  ✅ {split_name}: {valid_count} samples → {label_file}")
            total_converted += valid_count
    
    if total_converted > 0:
        print(f"✅ Converted SynthKhmer-10k: {total_converted} total samples")
        return True
    else:
        print(f"❌ No samples converted from SynthKhmer-10k")
        return False


def get_text_from_sample(sample):
    """Extract text from a sample with robust field checking"""
    # Method 1: Check ground_truth field (SynthKhmer-10k uses this)
    if 'ground_truth' in sample:
        gt = sample['ground_truth']
        if isinstance(gt, str):
            return gt.strip()
        elif isinstance(gt, dict):
            # Extract text from dict
            for field in ['text', 'transcription', 'label', 'sentence']:
                if field in gt and gt[field]:
                    return str(gt[field]).strip()
            # Try joining all string values
            texts = [str(v).strip() for v in gt.values() if isinstance(v, str) and v.strip()]
            if texts:
                return ' '.join(texts)
        elif isinstance(gt, list) and gt:
            # If it's a list of annotations
            texts = []
            for item in gt:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    for field in ['text', 'transcription', 'label']:
                        if field in item:
                            texts.append(str(item[field]))
            if texts:
                return ' '.join(texts)
    
    # Method 2: Direct text fields
    text_fields = ['text', 'label', 'sentence', 'transcription', 'caption', 'khmer']
    for field in text_fields:
        if field in sample and sample[field]:
            return str(sample[field]).strip()
    
    # Method 3: ID card fields (combine them)
    id_fields = ['name', 'id', 'date_of_birth', 'gender', 'address']
    id_parts = []
    for field in id_fields:
        if field in sample and sample[field]:
            id_parts.append(str(sample[field]).strip())
    if id_parts:
        return ' '.join(id_parts)
    
    return None


def convert_synthkhmer_direct(output_dir):
    """Convert SynthKhmer-10k directly from HF (fallback)"""
    print("\n🔄 Converting SynthKhmer-10k directly from HuggingFace...")
    
    try:
        # Load dataset directly - NO cache_dir!
        dataset = load_dataset("seanghay/SynthKhmer-10k")
        
        # Create output directories
        rec_dir = Path(output_dir) / "recognition"
        for split in ["train", "val", "test"]:
            (rec_dir / split).mkdir(parents=True, exist_ok=True)
        
        total_converted = 0
        
        # Map HF splits to our splits
        split_mapping = {
            "train": "train",
            "validation": "val",
            "test": "test"
        }
        
        for hf_split, our_split in split_mapping.items():
            if hf_split not in dataset:
                # Try without split
                if "train" in dataset:
                    data = dataset["train"]
                    # Split manually
                    n = len(data)
                    if our_split == "train":
                        indices = range(int(n * 0.8))
                    elif our_split == "val":
                        indices = range(int(n * 0.8), int(n * 0.9))
                    else:
                        indices = range(int(n * 0.9), n)
                else:
                    continue
            else:
                data = dataset[hf_split]
                indices = range(len(data))
            
            labels = []
            valid_count = 0
            no_text_count = 0
            
            # Print schema for debugging
            if len(data) > 0:
                print(f"  Schema for {hf_split}: {list(data[0].keys())}")
                # Print first sample to debug
                if 'ground_truth' in data[0]:
                    gt = data[0]['ground_truth']
                    print(f"  ground_truth type: {type(gt).__name__}")
                    if isinstance(gt, str):
                        print(f"  ground_truth sample: '{gt[:100]}...'")
            
            for idx in indices:
                row = data[idx]
                
                # Use robust text extraction
                text = get_text_from_sample(row)
                
                if not text:
                    no_text_count += 1
                    if no_text_count <= 3:  # Only log first few
                        print(f"    No text for image {idx}")
                    continue
                
                # Get image
                image = row.get('image')
                
                if image and text:
                    img_filename = f"synth_{our_split}_{valid_count:06d}.png"
                    img_path = rec_dir / our_split / img_filename
                    
                    try:
                        if isinstance(image, Image.Image):
                            image.save(img_path)
                            labels.append(f"{img_filename}\t{text}")
                            valid_count += 1
                            
                            # Show first 3 extracted texts
                            if valid_count <= 3:
                                print(f"    Sample {valid_count}: '{text[:50]}...'")
                    except Exception as e:
                        print(f"    Error saving image: {e}")
            
            if no_text_count > 3:
                print(f"    ... and {no_text_count - 3} more samples without text")
            
            # Save labels
            if labels:
                label_file = rec_dir / our_split / "label.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(labels))
                print(f"  ✅ {our_split}: {valid_count} samples → {label_file}")
                total_converted += valid_count
            else:
                print(f"  ❌ {our_split}: No valid samples found")
        
        if total_converted > 0:
            print(f"✅ Converted SynthKhmer-10k: {total_converted} total samples")
            return True
        else:
            print(f"❌ CRITICAL: SynthKhmer-10k produced 0 valid samples")
            print("   Check ground_truth field mapping in the dataset")
            return False
            
    except Exception as e:
        print(f"❌ Failed to convert SynthKhmer-10k: {e}")
        return False


def convert_fonts_dataset(input_dir, output_dir):
    """Convert khmerfonts dataset"""
    print("\n🔄 Converting khmerfonts-info-previews...")
    print("  ⚠️  Note: This dataset may not have text labels")
    
    # Try from downloaded files first
    fonts_dir = Path(input_dir) / "khmerfonts-info-previews"
    
    if fonts_dir.exists():
        # Convert from downloaded files
        rec_dir = Path(output_dir) / "fonts_recognition"
        rec_dir.mkdir(parents=True, exist_ok=True)
        
        total_converted = 0
        
        for split_dir in fonts_dir.iterdir():
            if split_dir.is_dir():
                images = sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg"))
                labels = []
                
                for img_path in images:
                    txt_path = img_path.with_suffix('.txt')
                    if txt_path.exists():
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if text:
                            out_img = rec_dir / img_path.name
                            try:
                                Image.open(img_path).save(out_img)
                                labels.append(f"{img_path.name}\t{text}")
                                total_converted += 1
                            except Exception as e:
                                print(f"    Error: {e}")
                
                if labels:
                    label_file = rec_dir / f"{split_dir.name}_label.txt"
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(labels))
        
        if total_converted > 0:
            print(f"✅ Converted fonts: {total_converted} samples")
            return True
    
    # Fallback to direct loading
    try:
        print("  Loading directly from HuggingFace...")
        dataset = load_dataset("seanghay/khmerfonts-info-previews")
        
        rec_dir = Path(output_dir) / "fonts_recognition"
        rec_dir.mkdir(parents=True, exist_ok=True)
        
        total_converted = 0
        all_labels = []
        
        for split_name in dataset.keys():
            data = dataset[split_name]
            
            # Print schema
            if len(data) > 0:
                print(f"  Schema: {list(data[0].keys())}")
            
            for idx, row in enumerate(data):
                # Get text - try both fields
                text = row.get('text') or row.get('sentence')
                image = row.get('image')
                
                if image and text:
                    img_filename = f"font_{total_converted:06d}.png"
                    img_path = rec_dir / img_filename
                    
                    try:
                        if isinstance(image, Image.Image):
                            image.save(img_path)
                            all_labels.append(f"{img_filename}\t{text}")
                            total_converted += 1
                    except Exception as e:
                        print(f"    Error: {e}")
        
        if all_labels:
            label_file = rec_dir / "label.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_labels))
            print(f"✅ Converted fonts: {total_converted} samples")
            return True
        else:
            print(f"❌ No fonts samples converted")
            return False
            
    except Exception as e:
        print(f"❌ Failed to convert fonts: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert HF datasets to PaddleOCR format")
    parser.add_argument("--input-dir", default="data/hf_datasets", help="Input directory")
    parser.add_argument("--output-dir", default="data/paddle_format", help="Output directory")
    parser.add_argument("--dataset", choices=["synthkhmer", "fonts", "all"], default="all")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Check input directory
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        print("Run: python ops/download_hf.py first")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    # Convert SynthKhmer
    if args.dataset in ["synthkhmer", "all"]:
        # Try from downloaded files first
        if convert_synthkhmer_from_downloaded(input_dir, output_dir):
            success_count += 1
        # Fallback to direct loading
        elif convert_synthkhmer_direct(output_dir):
            success_count += 1
        else:
            failed_count += 1
            print("❌ Failed to convert SynthKhmer-10k")
    
    # Convert fonts
    if args.dataset in ["fonts", "all"]:
        if convert_fonts_dataset(input_dir, output_dir):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Converted: {success_count} datasets")
    print(f"Failed: {failed_count} datasets")
    
    if success_count == 0:
        print("❌ No datasets converted successfully!")
        return 1
    
    print(f"✅ Conversion complete! Output in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())