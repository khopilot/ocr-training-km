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
    
    # Import training module
    from train.train_demo import main as train_main
    
    # Set training arguments
    train_args = {
        'config': config_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'train_list': 'data/train_list.txt',
        'val_list': 'data/val_list.txt',
        'save_dir': f'models/khmer_{args.mode}_hf',
        'log_dir': f'logs/{args.mode}_hf',
        'use_gpu': True,
        'checkpoint_interval': 10,
        'eval_interval': 5,
        'use_wandb': False,  # Set to True if you want to use Weights & Biases
        'project_name': 'khmer-ocr-hf',
    }
    
    # Create directories
    os.makedirs(train_args['save_dir'], exist_ok=True)
    os.makedirs(train_args['log_dir'], exist_ok=True)
    
    print("\nTraining Configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")
    
    # Run training
    try:
        # Call the training function
        train_main(**train_args)
        
        print("\n✅ Training completed successfully!")
        print(f"Model saved to: {train_args['save_dir']}")
        print(f"Logs saved to: {train_args['log_dir']}")
        
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
            model_dir=train_args['save_dir'],
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