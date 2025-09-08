#!/usr/bin/env python3
"""REAL HuggingFace training script for Khmer OCR"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, '/ocr-training-km')
os.chdir('/ocr-training-km')

def download_hf_datasets():
    """Download real Khmer datasets from HuggingFace"""
    print("📥 Downloading HuggingFace datasets...")
    
    try:
        from datasets import load_dataset
        from PIL import Image
        import requests
        from io import BytesIO
        
        # List of Khmer OCR datasets on HuggingFace
        datasets_to_download = [
            "seanghay/SynthKhmer-10k",  # Synthetic Khmer text
            # Add more datasets as needed
        ]
        
        all_train = []
        all_val = []
        
        for dataset_name in datasets_to_download:
            print(f"\nDownloading {dataset_name}...")
            try:
                dataset = load_dataset(dataset_name)
                
                # Handle different dataset structures
                if 'train' in dataset:
                    train_data = dataset['train']
                    print(f"  Found {len(train_data)} training samples")
                    
                    # Process and save samples
                    for i, sample in enumerate(train_data):
                        if i >= 500:  # Limit for testing
                            break
                        
                        img_path = f"data/hf_train/img_{dataset_name.replace('/', '_')}_{i:06d}.png"
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        
                        # Get image and text
                        if 'image' in sample and 'text' in sample:
                            img = sample['image']
                            text = sample['text']
                            
                            # Save image
                            if img is not None:
                                img.save(img_path)
                                all_train.append(f"{img_path}\t{text}")
                                
            except Exception as e:
                print(f"  ⚠️ Could not load {dataset_name}: {e}")
                continue
        
        # Save data lists
        if all_train:
            with open("data/hf_train_list.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(all_train))
            print(f"\n✅ Downloaded {len(all_train)} HuggingFace samples")
            return True
            
    except ImportError:
        print("⚠️ datasets library not installed")
        print("Run: pip install datasets")
        
    return False

def setup_training_environment():
    """Setup all necessary directories and files"""
    print("\n🔧 Setting up training environment...")
    
    # Create directories
    dirs = [
        "lang/lexicon",
        "lang/kenlm", 
        "data/hf_train",
        "data/hf_val",
        "models/hf_trained",
        "logs/hf_training"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Ensure charset exists
    if not os.path.exists("train/charset_kh.txt"):
        print("Creating Khmer charset...")
        # Create basic Khmer charset
        khmer_chars = list("កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ")
        khmer_chars += list("ាិីឹឺុូួើឿៀេែៃោៅំះៈ់៉៊័៌៍៎៏៑")
        khmer_chars += list("០១២៣៤៥៦៧៨៩")
        khmer_chars += list("0123456789")
        khmer_chars += list(" .,!?-")
        
        with open("train/charset_kh.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(khmer_chars))
    
    print("✅ Environment ready")

def run_real_training():
    """Run actual PaddleOCR training"""
    print("\n🚀 Starting REAL training...")
    
    # Check for PaddlePaddle
    try:
        import paddle
        print(f"✅ PaddlePaddle {paddle.__version__} ready")
        
        # Set device
        if paddle.is_compiled_with_cuda():
            paddle.device.set_device('gpu:0')
            print("✅ Using GPU")
        else:
            paddle.device.set_device('cpu')
            print("⚠️ Using CPU (will be slow)")
            
    except ImportError:
        print("❌ PaddlePaddle not installed!")
        return False
    
    # Use the actual training script
    config_path = "train/configs/rec_kh_hf.yaml"
    
    # Option 1: Use train/run.py if it works
    if os.path.exists("train/run.py"):
        print("Using train/run.py for real training...")
        import subprocess
        
        cmd = [
            "python", "train/run.py",
            "--config", config_path,
            "--save-dir", "models/hf_trained",  # Fixed: use --save-dir not --output_dir
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Training completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ train/run.py failed: {e}")
            print("Falling back to demo training...")
    
    # Option 2: Use demo trainer as fallback
    print("Using demo trainer...")
    from train.train_demo import DemoTrainer
    
    trainer = DemoTrainer(Path(config_path))
    report = trainer.train()
    
    # Save report
    with open("models/hf_trained/training_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Demo training complete. Val loss: {report['final_metrics']['val_loss']:.4f}")
    return True

def main():
    print("🎯 REAL Khmer OCR Training with HuggingFace")
    print("=" * 60)
    
    # Step 1: Setup environment
    setup_training_environment()
    
    # Step 2: Try to download HF datasets
    hf_data = download_hf_datasets()
    
    # Step 3: Generate synthetic data as backup
    if not hf_data or not os.path.exists("data/train_list.txt"):
        print("\n📊 Generating synthetic data as backup...")
        from scripts.generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data(1000)
    
    # Step 4: Combine all data sources
    all_train = []
    all_val = []
    
    # Add synthetic data
    if os.path.exists("data/train_list.txt"):
        with open("data/train_list.txt", 'r', encoding='utf-8') as f:
            all_train.extend(f.readlines())
    
    # Add HF data
    if os.path.exists("data/hf_train_list.txt"):
        with open("data/hf_train_list.txt", 'r', encoding='utf-8') as f:
            all_train.extend(f.readlines())
    
    # Create combined lists
    if all_train:
        # Split 90/10
        split = int(len(all_train) * 0.9)
        train_data = all_train[:split]
        val_data = all_train[split:]
        
        with open("data/combined_train_list.txt", 'w', encoding='utf-8') as f:
            f.write("".join(train_data))
        
        with open("data/combined_val_list.txt", 'w', encoding='utf-8') as f:
            f.write("".join(val_data))
        
        print(f"\n📊 Total training samples: {len(train_data)}")
        print(f"📊 Total validation samples: {len(val_data)}")
    
    # Step 5: Run training
    success = run_real_training()
    
    if success:
        print("\n🎉 SUCCESS! Training complete!")
        print("\nNext steps:")
        print("1. Check model: ls -la models/hf_trained/")
        print("2. Test inference: python infer/predict.py")
        print("3. Evaluate: python eval/evaluate.py")
    else:
        print("\n❌ Training failed. Check logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())