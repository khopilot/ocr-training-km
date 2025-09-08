#!/usr/bin/env python3
"""Introspect HuggingFace datasets to understand their actual schema"""

import sys
from pathlib import Path
from pprint import pprint

try:
    from datasets import load_dataset
    from PIL import Image
    HAS_HF = True
except ImportError:
    print("Error: datasets not installed")
    print("Run: pip install datasets pillow")
    sys.exit(1)


def introspect_synthkhmer():
    """Introspect SynthKhmer-10k dataset schema"""
    print("\n" + "="*60)
    print("DATASET: seanghay/SynthKhmer-10k")
    print("="*60)
    
    try:
        dataset = load_dataset("seanghay/SynthKhmer-10k")
        
        # Print available splits
        print(f"\nAvailable splits: {list(dataset.keys())}")
        
        for split_name in dataset.keys():
            print(f"\n--- Split: {split_name} ---")
            split_data = dataset[split_name]
            
            # Print dataset info
            print(f"Number of samples: {len(split_data)}")
            
            if len(split_data) > 0:
                # Print features/schema
                print(f"\nFeatures/Schema:")
                if hasattr(split_data, 'features'):
                    pprint(split_data.features)
                
                # Print first 3 samples
                print(f"\nFirst 3 samples (showing types and content):")
                for i in range(min(3, len(split_data))):
                    sample = split_data[i]
                    print(f"\n  Sample {i+1}:")
                    for key, value in sample.items():
                        value_type = type(value).__name__
                        
                        # Show different info based on type
                        if isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"    {key}: ({value_type}) '{preview}'")
                        elif isinstance(value, dict):
                            print(f"    {key}: ({value_type}) Keys: {list(value.keys())}")
                            # Show dict contents
                            for k, v in value.items():
                                if isinstance(v, str):
                                    preview = v[:50] + "..." if len(v) > 50 else v
                                    print(f"      - {k}: '{preview}'")
                                else:
                                    print(f"      - {k}: {type(v).__name__}")
                        elif isinstance(value, Image.Image):
                            print(f"    {key}: ({value_type}) Size: {value.size}, Mode: {value.mode}")
                        elif isinstance(value, list):
                            print(f"    {key}: ({value_type}) Length: {len(value)}")
                            if value and len(value) > 0:
                                print(f"      First item type: {type(value[0]).__name__}")
                        else:
                            print(f"    {key}: ({value_type}) {str(value)[:100]}")
                
                # Try to extract text from ground_truth
                print(f"\n  Text extraction test:")
                sample = split_data[0]
                
                # Method 1: Direct ground_truth access
                if 'ground_truth' in sample:
                    gt = sample['ground_truth']
                    print(f"    ground_truth type: {type(gt).__name__}")
                    if isinstance(gt, str):
                        print(f"    Extracted text: '{gt}'")
                    elif isinstance(gt, dict):
                        print(f"    ground_truth dict keys: {list(gt.keys())}")
                        # Try to extract text from dict
                        for key in ['text', 'transcription', 'label', 'sentence']:
                            if key in gt:
                                print(f"    Found '{key}': '{gt[key]}'")
                                break
                    elif isinstance(gt, list):
                        print(f"    ground_truth is a list with {len(gt)} items")
                        if gt:
                            print(f"    First item: {gt[0]}")
                
                # Method 2: Check for text field
                if 'text' in sample:
                    print(f"    Direct 'text' field: '{sample['text']}'")
                    
    except Exception as e:
        print(f"Error loading SynthKhmer-10k: {e}")


def introspect_khmerfonts():
    """Introspect khmerfonts-info-previews dataset schema"""
    print("\n" + "="*60)
    print("DATASET: seanghay/khmerfonts-info-previews")
    print("="*60)
    
    try:
        dataset = load_dataset("seanghay/khmerfonts-info-previews")
        
        # Print available splits
        print(f"\nAvailable splits: {list(dataset.keys())}")
        
        for split_name in list(dataset.keys())[:1]:  # Just check first split
            print(f"\n--- Split: {split_name} ---")
            split_data = dataset[split_name]
            
            # Print dataset info
            print(f"Number of samples: {len(split_data)}")
            
            if len(split_data) > 0:
                # Print features/schema
                print(f"\nFeatures/Schema:")
                if hasattr(split_data, 'features'):
                    pprint(split_data.features)
                
                # Print first 3 samples
                print(f"\nFirst 3 samples:")
                for i in range(min(3, len(split_data))):
                    sample = split_data[i]
                    print(f"\n  Sample {i+1}:")
                    for key, value in sample.items():
                        value_type = type(value).__name__
                        if isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"    {key}: ({value_type}) '{preview}'")
                        elif isinstance(value, Image.Image):
                            print(f"    {key}: ({value_type}) Size: {value.size}")
                        else:
                            print(f"    {key}: ({value_type})")
                
                # Check if there's any text field
                print(f"\n  Text field check:")
                sample = split_data[0]
                text_fields = ['text', 'label', 'sentence', 'transcription', 'caption']
                found_text = False
                for field in text_fields:
                    if field in sample:
                        print(f"    Found '{field}': '{sample[field]}'")
                        found_text = True
                        break
                
                if not found_text:
                    print("    ⚠️  NO TEXT FIELD FOUND - Cannot use for recognition training!")
                    
    except Exception as e:
        print(f"Error loading khmerfonts: {e}")


def introspect_dictionary():
    """Introspect khmer-dictionary-44k dataset schema"""
    print("\n" + "="*60)
    print("DATASET: seanghay/khmer-dictionary-44k")
    print("="*60)
    
    try:
        dataset = load_dataset("seanghay/khmer-dictionary-44k")
        
        # Print available splits
        print(f"\nAvailable splits: {list(dataset.keys())}")
        
        for split_name in list(dataset.keys())[:1]:
            print(f"\n--- Split: {split_name} ---")
            split_data = dataset[split_name]
            
            print(f"Number of samples: {len(split_data)}")
            
            if len(split_data) > 0:
                # Print features/schema
                print(f"\nFeatures/Schema:")
                if hasattr(split_data, 'features'):
                    pprint(split_data.features)
                
                # Print first 3 samples
                print(f"\nFirst 3 samples:")
                for i in range(min(3, len(split_data))):
                    sample = split_data[i]
                    print(f"\n  Sample {i+1}:")
                    for key, value in sample.items():
                        if isinstance(value, str):
                            preview = value[:100] + "..." if len(value) > 100 else value
                            print(f"    {key}: '{preview}'")
                        else:
                            print(f"    {key}: {type(value).__name__}")
                            
    except Exception as e:
        print(f"Error loading dictionary: {e}")


def main():
    """Run introspection on all datasets"""
    print("="*60)
    print("HUGGINGFACE DATASET INTROSPECTION")
    print("="*60)
    
    # Check SynthKhmer-10k (main training data)
    introspect_synthkhmer()
    
    # Check khmerfonts
    introspect_khmerfonts()
    
    # Check dictionary
    introspect_dictionary()
    
    print("\n" + "="*60)
    print("INTROSPECTION COMPLETE")
    print("="*60)
    print("\nKey findings will help fix the conversion scripts.")


if __name__ == "__main__":
    main()