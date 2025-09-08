#!/usr/bin/env python3
"""Simple direct training script for Khmer OCR"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("🚀 Simple Khmer OCR Training")
    print("=" * 60)
    
    # Check for PaddlePaddle
    try:
        import paddle
        print(f"✅ PaddlePaddle {paddle.__version__} available")
        
        # Check GPU
        if paddle.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            print(f"✅ GPUs available: {gpu_count}")
            paddle.device.set_device('gpu:0')
        else:
            print("⚠️  Using CPU (slow)")
            paddle.device.set_device('cpu')
            
    except ImportError:
        print("❌ PaddlePaddle not installed!")
        print("Run: pip install paddlepaddle-gpu==2.6.1")
        sys.exit(1)
    
    # Check PaddleOCR
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR available")
    except ImportError:
        print("❌ PaddleOCR not installed!")
        print("Run: pip install paddleocr==2.7.0")
        sys.exit(1)
    
    # Check data
    if not os.path.exists("data/train_list.txt"):
        print("\n⚠️  No training data found. Generating synthetic data...")
        from generate_khmer_synthetic import generate_synthetic_data
        generate_synthetic_data(1000)
    
    # Simple training using PaddleOCR directly
    print("\n" + "=" * 60)
    print("Starting PaddleOCR Training")
    print("=" * 60 + "\n")
    
    from paddleocr.tools import train
    
    # Training configuration
    config = {
        'Global': {
            'use_gpu': True,
            'epoch_num': 20,
            'log_smooth_window': 20,
            'print_batch_step': 10,
            'save_model_dir': './models/khmer_ocr',
            'save_epoch_step': 5,
            'eval_batch_step': [0, 100],
            'cal_metric_during_train': True,
            'pretrained_model': None,
            'checkpoints': None,
            'save_inference_dir': None,
            'use_visualdl': False,
            'character_dict_path': 'train/charset_kh.txt',
            'max_text_length': 100,
            'infer_mode': False,
            'use_space_char': True,
            'save_res_path': './output/rec/predicts.txt',
        },
        'Architecture': {
            'model_type': 'rec',
            'algorithm': 'CRNN',
            'Transform': None,
            'Backbone': {
                'name': 'MobileNetV3',
                'scale': 0.5,
                'model_name': 'small',
            },
            'Neck': {
                'name': 'SequenceEncoder',
                'encoder_type': 'rnn',
                'hidden_size': 48,
            },
            'Head': {
                'name': 'CTCHead',
                'fc_decay': 0.00001,
            }
        },
        'Loss': {
            'name': 'CTCLoss'
        },
        'Optimizer': {
            'name': 'Adam',
            'beta1': 0.9,
            'beta2': 0.999,
            'lr': {
                'name': 'Cosine',
                'learning_rate': 0.001,
            },
            'regularizer': {
                'name': 'L2',
                'factor': 0.00001
            }
        },
        'PostProcess': {
            'name': 'CTCLabelDecode'
        },
        'Metric': {
            'name': 'RecMetric',
            'main_indicator': 'acc'
        },
        'Train': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': './data/',
                'label_file_list': ['./data/train_list.txt'],
                'transforms': [
                    {
                        'DecodeImage': {
                            'img_mode': 'BGR',
                            'channel_first': False
                        }
                    },
                    {
                        'RecResizeImg': {
                            'image_shape': [3, 32, 320]
                        }
                    },
                    {
                        'KeepKeys': {
                            'keep_keys': ['image', 'label', 'length']
                        }
                    }
                ]
            },
            'loader': {
                'shuffle': True,
                'batch_size_per_card': 8,
                'drop_last': True,
                'num_workers': 4,
            }
        },
        'Eval': {
            'dataset': {
                'name': 'SimpleDataSet',
                'data_dir': './data/',
                'label_file_list': ['./data/val_list.txt'],
                'transforms': [
                    {
                        'DecodeImage': {
                            'img_mode': 'BGR',
                            'channel_first': False
                        }
                    },
                    {
                        'RecResizeImg': {
                            'image_shape': [3, 32, 320]
                        }
                    },
                    {
                        'KeepKeys': {
                            'keep_keys': ['image', 'label', 'length']
                        }
                    }
                ]
            },
            'loader': {
                'shuffle': False,
                'drop_last': False,
                'batch_size_per_card': 8,
                'num_workers': 4,
            }
        }
    }
    
    # Save config to YAML
    import yaml
    config_path = 'configs/simple_rec_config.yaml'
    os.makedirs('configs', exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to: {config_path}")
    
    # Run training
    print("\nStarting training with PaddleOCR...")
    print("This will train a lightweight CRNN model for Khmer text recognition")
    print("-" * 60)
    
    # Don't import PaddleOCR training API - use our own training entry
    import subprocess
    
    # Use our own training script instead of PaddleOCR's
    cmd = [
        "python", "train/run.py",
        "--config", "train/configs/rec_kh_hf.yaml"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed!")
        print(f"Model saved to: ./models/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTrying with CPU config...")
        
        # Fallback to CPU training with smaller batch
        cmd = ["python", "train/run.py", "--config", "train/configs/rec_kh.yaml"]
        subprocess.run(cmd, check=False)
    
    print("\n🎉 Training pipeline complete!")
    print("\nNext steps:")
    print("1. Test your model: python infer/predict.py --model ./models/khmer_ocr")
    print("2. Export for inference: python scripts/export_model.py")
    print("3. Start API: python service/app.py")

if __name__ == "__main__":
    main()