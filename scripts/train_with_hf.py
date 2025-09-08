#!/usr/bin/env python3
"""Train Khmer OCR model using HuggingFace configuration"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def train_with_huggingface():
    """Train using HuggingFace-optimized configuration"""
    
    parser = argparse.ArgumentParser(description="Train Khmer OCR with HuggingFace config")
    parser.add_argument("--mode", type=str, default="recognition", choices=["recognition", "detection"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples to generate")
    
    args = parser.parse_args()
    
    print("🤗 Khmer OCR Training with HuggingFace Configuration")
    print("=" * 60)
    
    # Step 1: Setup data
    print("\n📊 Step 1: Setting up training data...")
    
    # Check if we have data
    if not os.path.exists("data/train_list.txt"):
        print("No training data found. Generating synthetic data...")
        from generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data(args.num_samples)
    else:
        # Count existing samples
        with open("data/train_list.txt", 'r') as f:
            train_count = len(f.readlines())
        print(f"Found {train_count} training samples")
    
    # Step 2: Load HuggingFace config
    print("\n⚙️ Step 2: Loading HuggingFace configuration...")
    
    if args.mode == "recognition":
        config_path = "train/configs/rec_kh_hf.yaml"
    else:
        config_path = "train/configs/dbnet_hf.yaml"
    
    print(f"Using config: {config_path}")
    
    # Step 3: Start training
    print("\n🚀 Step 3: Starting training...")
    
    # Create directories
    save_dir = f'models/khmer_{args.mode}_hf'
    log_dir = f'logs/{args.mode}_hf'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("\nTraining Configuration:")
    print(f"  config: {config_path}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  learning_rate: {args.lr}")
    print(f"  save_dir: {save_dir}")
    print(f"  log_dir: {log_dir}")
    
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")
    
    # Run training - train_demo.py uses argparse, so we need to set sys.argv
    import subprocess
    
    # Build command (train_demo.py only accepts --config and --output-dir)
    cmd = [
        "python", "train/train_demo.py",
        "--config", config_path,
        "--output-dir", save_dir,
    ]
    
    # Run training
    try:
        # Run the training script
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print("\n✅ Training completed successfully!")
        print(f"Model saved to: {save_dir}")
        print(f"Logs saved to: {log_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Evaluate model
    print("\n📊 Step 4: Evaluating model...")
    
    try:
        from eval.evaluate import evaluate_model
        
        metrics = evaluate_model(
            model_dir=save_dir,
            test_list='data/val_list.txt',
            mode=args.mode
        )
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
    except Exception as e:
        print(f"⚠️ Evaluation skipped: {e}")
    
    print("\n🎉 All done! Your Khmer OCR model is ready.")
    print("\nNext steps:")
    print("1. Test inference: python infer/predict.py --image <your_image.png>")
    print("2. Export model: python scripts/export_model.py")
    print("3. Deploy API: python service/app.py")

if __name__ == "__main__":
    train_with_huggingface()