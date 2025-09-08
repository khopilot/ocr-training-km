#!/usr/bin/env python3
"""Prepare training data lists for PaddleOCR"""

import os
import json
import random
from pathlib import Path

def create_data_lists():
    """Create train_list.txt and val_list.txt for PaddleOCR training"""
    
    print("📝 Preparing data lists for PaddleOCR training...")
    
    # Data directories
    train_dir = Path("data/train")
    val_dir = Path("data/val")
    
    # Output files
    train_list = Path("data/train_list.txt")
    val_list = Path("data/val_list.txt")
    
    # Process training data
    print(f"\n Processing {train_dir}...")
    train_samples = []
    
    for img_file in train_dir.glob("*.png"):
        # Look for corresponding label file
        label_file = img_file.with_suffix(".txt")
        if label_file.exists():
            # Read label
            with open(label_file, 'r', encoding='utf-8') as f:
                label = f.read().strip()
            
            # PaddleOCR format: image_path\tlabel
            # Use relative path from project root
            rel_path = str(img_file).replace("\\", "/")
            train_samples.append(f"{rel_path}\t{label}")
        else:
            print(f"⚠️  No label for {img_file.name}")
    
    # Write training list
    if train_samples:
        with open(train_list, 'w', encoding='utf-8') as f:
            f.write("\n".join(train_samples))
        print(f"✅ Created {train_list} with {len(train_samples)} samples")
    else:
        print(f"⚠️  No training samples found")
    
    # Process validation data
    print(f"\nProcessing {val_dir}...")
    val_samples = []
    
    for img_file in val_dir.glob("*.png"):
        label_file = img_file.with_suffix(".txt")
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                label = f.read().strip()
            
            rel_path = str(img_file).replace("\\", "/")
            val_samples.append(f"{rel_path}\t{label}")
        else:
            print(f"⚠️  No label for {img_file.name}")
    
    # Write validation list
    if val_samples:
        with open(val_list, 'w', encoding='utf-8') as f:
            f.write("\n".join(val_samples))
        print(f"✅ Created {val_list} with {len(val_samples)} samples")
    else:
        print(f"⚠️  No validation samples found")
    
    # Also create a combined synthetic data list if available
    synth_dir = Path("data/synth")
    if synth_dir.exists():
        print(f"\nProcessing {synth_dir}...")
        synth_samples = []
        
        for img_file in synth_dir.glob("*.png"):
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                with open(label_file, 'r', encoding='utf-8') as f:
                    label = f.read().strip()
                
                rel_path = str(img_file).replace("\\", "/")
                synth_samples.append(f"{rel_path}\t{label}")
        
        if synth_samples:
            # Split synthetic data 80/20 for train/val
            random.shuffle(synth_samples)
            split = int(len(synth_samples) * 0.8)
            synth_train = synth_samples[:split]
            synth_val = synth_samples[split:]
            
            # Create synthetic data lists
            synth_train_list = Path("data/synth_train_list.txt")
            synth_val_list = Path("data/synth_val_list.txt")
            
            with open(synth_train_list, 'w', encoding='utf-8') as f:
                f.write("\n".join(synth_train))
            print(f"✅ Created {synth_train_list} with {len(synth_train)} samples")
            
            with open(synth_val_list, 'w', encoding='utf-8') as f:
                f.write("\n".join(synth_val))
            print(f"✅ Created {synth_val_list} with {len(synth_val)} samples")
            
            # Create combined lists
            all_train_list = Path("data/all_train_list.txt")
            all_val_list = Path("data/all_val_list.txt")
            
            with open(all_train_list, 'w', encoding='utf-8') as f:
                f.write("\n".join(train_samples + synth_train))
            print(f"✅ Created {all_train_list} with {len(train_samples) + len(synth_train)} samples (real + synthetic)")
            
            with open(all_val_list, 'w', encoding='utf-8') as f:
                f.write("\n".join(val_samples + synth_val))
            print(f"✅ Created {all_val_list} with {len(val_samples) + len(synth_val)} samples (real + synthetic)")
    
    print("\n✅ Data preparation complete!")
    print("\nYou can now use these files for training:")
    print("  - data/train_list.txt: Real training data only")
    print("  - data/val_list.txt: Real validation data only")
    print("  - data/all_train_list.txt: Real + synthetic training data")
    print("  - data/all_val_list.txt: Real + synthetic validation data")

if __name__ == "__main__":
    create_data_lists()