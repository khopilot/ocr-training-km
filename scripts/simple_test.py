#!/usr/bin/env python3
"""Simple test to verify training worked"""

import os
import json
from pathlib import Path

os.chdir('/ocr-training-km')

print("🔍 Checking Training Results")
print("=" * 40)

# Check training report
report_path = Path("models/khmer_ocr/training_report.json")
if report_path.exists():
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print("✅ Training completed successfully!")
    print(f"   Training samples: {report['dataset_size']['train']}")
    print(f"   Validation samples: {report['dataset_size']['val']}")
    print(f"   Epochs trained: {report['training_params']['epochs']}")
    print(f"   Final train loss: {report['final_metrics']['train_loss']:.4f}")
    print(f"   Final val loss: {report['final_metrics']['val_loss']:.4f}")
    print(f"   Training time: {report['timing']['total_time_sec']:.1f} seconds")
    
    print("\n📈 Loss progression:")
    for log in report['training_log']:
        print(f"   Epoch {log['epoch']}: train={log['train_loss']:.4f}, val={log['val_loss']:.4f}")
    
    # Check if loss decreased (model learned)
    first_loss = report['training_log'][0]['val_loss']
    final_loss = report['final_metrics']['val_loss']
    improvement = (first_loss - final_loss) / first_loss * 100
    
    print(f"\n📊 Model improved by {improvement:.1f}% (lower is better)")
    
    if improvement > 10:
        print("🎉 Excellent! Model learned well.")
    elif improvement > 0:
        print("✅ Good! Model is learning. Train more epochs for better results.")
    else:
        print("⚠️ Model needs more training.")
        
else:
    print("❌ No training report found")

print("\n" + "=" * 40)
print("\nNext steps:")
print("1. Train more epochs: python scripts/FINAL_WORKING_TRAIN.py")
print("2. Use full PaddleOCR for real inference (needs GPU)")
print("3. Or continue with demo training for testing")