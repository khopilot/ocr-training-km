#!/usr/bin/env python3
"""Setup training data from HuggingFace datasets"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

def setup_hf_data():
    """Download and prepare Khmer OCR data from HuggingFace"""
    
    print("🤗 Setting up data from HuggingFace...")
    print("=" * 50)
    
    # Create data directories
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    # Your HuggingFace dataset
    dataset_name = "khopilot/khmer-ocr-train"  # Replace with your actual dataset
    
    try:
        print(f"Loading dataset: {dataset_name}")
        
        # Try to load the dataset
        # You can modify this based on your actual dataset structure
        dataset = load_dataset(dataset_name, split="train")
        
        print(f"✅ Loaded {len(dataset)} samples")
        
        # Split data: 80% train, 10% val, 10% test
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset) * 0.1)
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        # Save train data
        print("\n📝 Creating training data...")
        train_list = []
        for i, sample in enumerate(train_data):
            img_path = f"data/train/img_{i:06d}.png"
            
            # Save image
            if 'image' in sample:
                img = sample['image']
                if isinstance(img, str):  # URL
                    response = requests.get(img)
                    img = Image.open(BytesIO(response.content))
                img.save(img_path)
            
            # Save label
            if 'text' in sample:
                label_path = f"data/train/img_{i:06d}.txt"
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(sample['text'])
                
                train_list.append(f"{img_path}\t{sample['text']}")
        
        # Save validation data
        print("📝 Creating validation data...")
        val_list = []
        for i, sample in enumerate(val_data):
            img_path = f"data/val/img_{i:06d}.png"
            
            if 'image' in sample:
                img = sample['image']
                if isinstance(img, str):
                    response = requests.get(img)
                    img = Image.open(BytesIO(response.content))
                img.save(img_path)
            
            if 'text' in sample:
                label_path = f"data/val/img_{i:06d}.txt"
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(sample['text'])
                
                val_list.append(f"{img_path}\t{sample['text']}")
        
        # Create list files
        with open("data/train_list.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(train_list))
        
        with open("data/val_list.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(val_list))
        
        print(f"\n✅ Data setup complete!")
        print(f"  Training samples: {len(train_list)}")
        print(f"  Validation samples: {len(val_list)}")
        
    except Exception as e:
        print(f"⚠️  Could not load from HuggingFace: {e}")
        print("\nUsing alternative: Khmer synthetic data generation...")
        
        # Generate synthetic Khmer data as fallback
        from generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data()

if __name__ == "__main__":
    setup_hf_data()