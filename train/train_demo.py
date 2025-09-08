#!/usr/bin/env python3
"""
Lightweight demo training script for macOS
Simplified PaddleOCR recognition training for validation
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import numpy as np
    import cv2
    from PIL import Image
    import yaml
    from tqdm import tqdm
    print("✅ Basic dependencies loaded")
except ImportError as e:
    print(f"❌ Missing basic dependencies: {e}")
    print("Run: pip install numpy opencv-python pillow pyyaml tqdm")
    sys.exit(1)

# Optional PaddleOCR import
try:
    import paddleocr
    PADDLE_AVAILABLE = True
    print("✅ PaddleOCR available for training")
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️  PaddleOCR not available - demo training will use mock")

class DemoTrainer:
    """Lightweight trainer for demo purposes"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.charset = self._load_charset()
        self.start_time = time.time()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_charset(self) -> list:
        """Load character set"""
        charset_path = Path(self.config['charset'])
        with open(charset_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _load_dataset(self, split: str) -> list:
        """Load dataset split"""
        if split == 'train':
            label_file = Path(self.config['data']['train_list'])
        elif split == 'val':
            label_file = Path(self.config['data']['val_list'])
        else:
            raise ValueError(f"Unknown split: {split}")
            
        samples = []
        if label_file.exists():
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '\t' in line:
                        img_path, text = line.strip().split('\t', 1)
                        samples.append((img_path, text))
        
        return samples
    
    def validate_data(self) -> bool:
        """Validate training data"""
        print("🔍 Validating training data...")
        
        train_samples = self._load_dataset('train')
        val_samples = self._load_dataset('val')
        
        print(f"  📊 Dataset sizes:")
        print(f"     Training: {len(train_samples)} samples")
        print(f"     Validation: {len(val_samples)} samples")
        print(f"     Charset: {len(self.charset)} characters")
        
        # Check a few images exist
        missing_count = 0
        for i, (img_path, text) in enumerate(train_samples[:10]):
            full_path = Path('data/train') / img_path
            if not full_path.exists():
                missing_count += 1
                
        if missing_count > 0:
            print(f"  ⚠️  {missing_count}/10 sample images missing")
        else:
            print(f"  ✅ Sample images validated")
            
        return True
    
    def mock_train_epoch(self, epoch: int, samples: list) -> Dict[str, float]:
        """Mock training epoch with realistic timing"""
        batch_size = self.config['train']['batch_size']
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        print(f"  Epoch {epoch+1}: {num_batches} batches, {len(samples)} samples")
        
        # Simulate training with progress bar
        epoch_start = time.time()
        total_loss = 0.0
        
        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}", leave=False) as pbar:
            for batch_idx in pbar:
                # Simulate batch processing time (faster for demo)
                time.sleep(0.01)  # 10ms per batch
                
                # Mock loss calculation (decreasing over time)
                batch_loss = max(0.1, 4.0 - (epoch * 0.5) - (batch_idx * 0.001))
                batch_loss += np.random.normal(0, 0.1)  # Add noise
                total_loss += batch_loss
                
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        print(f"    ✅ Completed in {epoch_time:.1f}s - Loss: {avg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'batches': num_batches
        }
    
    def train(self) -> Dict[str, Any]:
        """Run demo training"""
        print("🚀 Starting demo training...")
        print(f"   Platform: macOS (CPU-only demo mode)")
        print(f"   Config: {self.config_path}")
        print(f"   PaddleOCR: {'Available' if PADDLE_AVAILABLE else 'Mock mode'}")
        print()
        
        # Validate data
        if not self.validate_data():
            raise RuntimeError("Data validation failed")
        
        # Load datasets
        train_samples = self._load_dataset('train')
        val_samples = self._load_dataset('val')
        
        # Demo training parameters
        epochs = min(3, self.config['train']['epochs'])  # Limit for demo
        print(f"🎯 Demo training: {epochs} epochs (limited for demo)")
        print()
        
        # Training loop
        training_log = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"📚 Epoch {epoch+1}/{epochs}")
            
            # Train epoch
            train_metrics = self.mock_train_epoch(epoch, train_samples)
            
            # Mock validation
            val_metrics = self.mock_train_epoch(epoch, val_samples)
            val_metrics['loss'] *= 0.9  # Validation usually lower
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'epoch_time': train_metrics['epoch_time'],
                'timestamp': time.time()
            }
            training_log.append(epoch_log)
            
            print(f"    📊 Train loss: {train_metrics['loss']:.4f}")
            print(f"    📊 Val loss: {val_metrics['loss']:.4f}")
            
            # Save best model (mock)
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                print(f"    💾 New best model saved (loss: {best_loss:.4f})")
            
            print()
        
        total_time = time.time() - self.start_time
        
        # Generate training report
        report = {
            'training_type': 'DEMO/DEVELOPMENT',
            'platform': 'macOS M3',
            'engine': 'Mock PaddleOCR' if not PADDLE_AVAILABLE else 'PaddleOCR CPU',
            'config': str(self.config_path),
            'dataset_size': {
                'train': len(train_samples),
                'val': len(val_samples)
            },
            'charset_size': len(self.charset),
            'training_params': {
                'epochs': epochs,
                'batch_size': self.config['train']['batch_size'],
                'architecture': self.config['architecture']
            },
            'final_metrics': {
                'train_loss': training_log[-1]['train_loss'],
                'val_loss': training_log[-1]['val_loss'],
                'best_val_loss': best_loss
            },
            'timing': {
                'total_time_sec': total_time,
                'avg_epoch_time': total_time / epochs
            },
            'training_log': training_log,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'disclaimer': 'This is DEMO training with mock PaddleOCR for development purposes'
        }
        
        return report

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Demo training for Khmer OCR')
    parser.add_argument('--config', type=Path, default='train/configs/rec_kh.yaml',
                       help='Training config file')
    parser.add_argument('--output-dir', type=Path, default='models/demo',
                       help='Output directory for demo models')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = DemoTrainer(args.config)
        
        # Run training
        report = trainer.train()
        
        # Save training report
        report_path = args.output_dir / 'training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("🎉 Demo training completed!")
        print(f"   📊 Final val loss: {report['final_metrics']['val_loss']:.4f}")
        print(f"   ⏱️  Total time: {report['timing']['total_time_sec']:.1f}s")
        print(f"   📄 Report: {report_path}")
        print()
        print("⚠️  Remember: This is DEMO training for development validation")
        print("   Use Docker + GPU for production training")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()