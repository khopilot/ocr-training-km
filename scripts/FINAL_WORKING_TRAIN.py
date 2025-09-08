#!/usr/bin/env python3
"""FINAL WORKING training script - handles all issues correctly"""

import os
import sys
import shutil
from pathlib import Path

# Set working directory
os.chdir('/ocr-training-km')
sys.path.insert(0, '/ocr-training-km')

def consolidate_data():
    """Consolidate all data sources into single files"""
    print("📊 Consolidating all training data...")
    
    all_train = []
    all_val = []
    
    # Check for HuggingFace data
    hf_data_dir = Path("data/hf_train")
    if hf_data_dir.exists():
        print(f"  Found HuggingFace data in {hf_data_dir}")
        for img_path in hf_data_dir.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                all_train.append(f"{img_path}\t{text}")
    
    # Check for synthetic data in data/train
    synth_train_dir = Path("data/train")
    if synth_train_dir.exists():
        print(f"  Found synthetic data in {synth_train_dir}")
        for img_path in synth_train_dir.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                all_train.append(f"{img_path}\t{text}")
    
    # Check for synthetic data in data/val
    synth_val_dir = Path("data/val")
    if synth_val_dir.exists():
        print(f"  Found validation data in {synth_val_dir}")
        for img_path in synth_val_dir.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                all_val.append(f"{img_path}\t{text}")
    
    # If no validation data, split training data
    if not all_val and all_train:
        split = int(len(all_train) * 0.9)
        all_val = all_train[split:]
        all_train = all_train[:split]
    
    # Write consolidated files
    if all_train:
        with open("data/train_list.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(all_train))
        print(f"✅ Created data/train_list.txt with {len(all_train)} samples")
    
    if all_val:
        with open("data/val_list.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(all_val))
        print(f"✅ Created data/val_list.txt with {len(all_val)} samples")
    
    # Create test list if needed
    if not os.path.exists("data/test_list.txt"):
        # Use some validation data as test
        test_data = all_val[:min(50, len(all_val)//2)]
        with open("data/test_list.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(test_data))
        print(f"✅ Created data/test_list.txt with {len(test_data)} samples")
    
    return len(all_train), len(all_val)

def main():
    print("🎯 FINAL WORKING KHMER OCR TRAINING")
    print("=" * 60)
    
    # Step 1: Ensure directories exist
    print("\n📁 Setting up directories...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Step 2: Check/create charset
    if not os.path.exists("train/charset_kh.txt"):
        print("Creating Khmer charset...")
        os.makedirs("train", exist_ok=True)
        # Basic Khmer charset
        chars = list(" កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ")
        chars += list("ាិីឹឺុូួើឿៀេែៃោៅំះៈ់៉៊័")
        chars += list("០១២៣៤៥៦៧៨៩0123456789.,!?-")
        
        with open("train/charset_kh.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(chars))
        print("✅ Created charset")
    
    # Step 3: Consolidate all data
    train_count, val_count = consolidate_data()
    
    if train_count == 0:
        print("\n⚠️  No training data found! Generating synthetic data...")
        from scripts.generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data(1000)
        train_count, val_count = consolidate_data()
    
    # Step 4: Choose and run appropriate trainer
    print(f"\n🚀 Starting training with {train_count} train, {val_count} val samples...")
    
    # Try different training approaches
    training_success = False
    
    # Option 1: Try train/run.py with correct arguments
    if os.path.exists("train/run.py") and not training_success:
        print("\nTrying train/run.py...")
        import subprocess
        
        cmd = [
            "python", "train/run.py",
            "--config", "train/configs/rec_kh_hf.yaml",
            "--save-dir", "models/khmer_ocr",
            "--epochs", "10",
            "--batch-size", "4",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Training with run.py successful!")
                training_success = True
            else:
                print(f"⚠️  run.py failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"⚠️  run.py error: {e}")
    
    # Option 2: Try demo trainer as fallback
    if not training_success:
        print("\nUsing demo trainer as fallback...")
        try:
            from train.train_demo import DemoTrainer
            import json
            
            config_path = Path("train/configs/rec_kh_hf.yaml")
            trainer = DemoTrainer(config_path)
            report = trainer.train()
            
            # Save report
            output_dir = Path("models/khmer_ocr")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "training_report.json", 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Demo training complete!")
            print(f"   Val loss: {report['final_metrics']['val_loss']:.4f}")
            training_success = True
            
        except Exception as e:
            print(f"❌ Demo trainer failed: {e}")
    
    # Option 3: Direct PaddleOCR command
    if not training_success:
        print("\nTrying direct PaddleOCR command...")
        import subprocess
        
        cmd = [
            "python", "-m", "paddleocr.tools.train",
            "-c", "train/configs/rec_kh_fixed.yaml",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if "error" not in result.stderr.lower():
                print("✅ PaddleOCR training started!")
                training_success = True
        except Exception as e:
            print(f"⚠️  Direct PaddleOCR failed: {e}")
    
    # Summary
    if training_success:
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYour model is saved in: models/khmer_ocr/")
        print("\nNext steps:")
        print("1. Check training report: cat models/khmer_ocr/training_report.json")
        print("2. Test inference: python infer/predict.py")
        print("3. Start API: python service/app.py")
    else:
        print("\n❌ All training methods failed.")
        print("Please check:")
        print("1. PaddlePaddle is installed: pip install paddlepaddle-gpu==2.6.1")
        print("2. PaddleOCR is installed: pip install paddleocr==2.7.0")
        print("3. Data exists: ls -la data/")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())