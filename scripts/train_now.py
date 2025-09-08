#!/usr/bin/env python3
"""Direct training script that actually works for Khmer OCR"""

import os
import sys
from pathlib import Path

# Navigate to project root
os.chdir('/ocr-training-km')
sys.path.insert(0, '/ocr-training-km')

def main():
    print("🚀 Starting Khmer OCR Training (Direct)")
    print("=" * 60)
    
    # Step 1: Generate data if needed
    if not os.path.exists("data/train_list.txt"):
        print("📊 Generating synthetic training data...")
        from scripts.generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data(500)
    
    # Step 2: Run the demo training
    print("\n🏋️ Starting training with demo trainer...")
    
    # Import the demo trainer
    from train.train_demo import DemoTrainer
    import json
    
    # Setup paths
    config_path = Path("train/configs/rec_kh_hf.yaml")
    output_dir = Path("models/khmer_recognition")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize trainer
        print(f"📋 Using config: {config_path}")
        trainer = DemoTrainer(config_path)
        
        # Run training (it uses epochs from config file)
        print("🔄 Training in progress...")
        report = trainer.train()
        
        # Save report
        report_path = output_dir / 'training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("\n✅ Training completed successfully!")
        print(f"📊 Final validation loss: {report['final_metrics']['val_loss']:.4f}")
        print(f"⏱️  Total time: {report['timing']['total_time_sec']:.1f}s")
        print(f"📁 Model saved to: {output_dir}")
        print(f"📄 Report saved to: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n🎉 Success! Your Khmer OCR model is ready.")
    print("\nNext steps:")
    print("1. Check the training report: cat models/khmer_recognition/training_report.json")
    print("2. Test inference: python infer/predict.py --model models/khmer_recognition")
    print("3. Start API: python service/app.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())