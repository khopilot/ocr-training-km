#!/usr/bin/env python3
"""Training script with khopilot/km-tokenizer-khmer integration"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed")

try:
    import paddle
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddlePaddle not available")


class TokenizerIntegratedTrainer:
    """OCR Trainer with khopilot/km-tokenizer-khmer integration"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize tokenizer
        self.tokenizer = None
        if HAS_TRANSFORMERS:
            try:
                print("Loading khopilot/km-tokenizer-khmer...")
                self.tokenizer = AutoTokenizer.from_pretrained("khopilot/km-tokenizer-khmer")
                print(f"✅ Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
            except Exception as e:
                print(f"❌ Failed to load tokenizer: {e}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def tokenize_labels(self, label_file: Path) -> Path:
        """Tokenize labels using khopilot tokenizer"""
        if not self.tokenizer:
            print("⚠️  No tokenizer available, using original labels")
            return label_file
        
        print(f"Tokenizing {label_file}...")
        tokenized_file = label_file.parent / f"tokenized_{label_file.name}"
        
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tokenized_lines = []
        for line in lines:
            if '\t' not in line:
                continue
            
            img_path, text = line.strip().split('\t', 1)
            
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # For CTC training, we need character-level but we can use subword tokens
            # Join tokens back for now (later we can modify CTC to handle subwords)
            tokenized_text = ' '.join(tokens)
            
            tokenized_lines.append(f"{img_path}\t{tokenized_text}")
        
        with open(tokenized_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tokenized_lines))
        
        print(f"✅ Tokenized {len(tokenized_lines)} samples → {tokenized_file}")
        return tokenized_file
    
    def prepare_charset_from_tokenizer(self) -> Path:
        """Generate charset from tokenizer vocabulary"""
        if not self.tokenizer:
            return Path("train/charset_kh.txt")
        
        charset_file = Path("train/charset_tokenizer.txt")
        
        # Get all unique characters from tokenizer vocab
        chars = set()
        for token in self.tokenizer.get_vocab().keys():
            # Remove special tokens
            if token.startswith('[') and token.endswith(']'):
                continue
            if token.startswith('##'):
                token = token[2:]  # Remove ## prefix for subwords
            chars.update(token)
        
        # Add special characters
        chars.add(' ')
        chars.add('\n')
        
        # Sort and save
        sorted_chars = sorted(list(chars))
        with open(charset_file, 'w', encoding='utf-8') as f:
            for char in sorted_chars:
                if char == '\n':
                    f.write('\\n\n')
                else:
                    f.write(f"{char}\n")
        
        print(f"✅ Generated charset with {len(sorted_chars)} characters")
        return charset_file
    
    def train(self):
        """Run training with tokenizer integration"""
        print("\n🚀 Starting training with tokenizer integration")
        
        # Prepare tokenized data
        data_config = self.config.get('data', {})
        
        # Tokenize training labels
        train_file = Path(data_config.get('train_list', 'data/paddle_format/recognition/train/label.txt'))
        val_file = Path(data_config.get('val_list', 'data/paddle_format/recognition/val/label.txt'))
        test_file = Path(data_config.get('test_list', 'data/paddle_format/recognition/test/label.txt'))
        
        if self.tokenizer:
            # Use tokenized versions
            train_file = self.tokenize_labels(train_file)
            val_file = self.tokenize_labels(val_file)
            test_file = self.tokenize_labels(test_file)
            
            # Update config with tokenized paths
            self.config['data']['train_list'] = str(train_file)
            self.config['data']['val_list'] = str(val_file)
            self.config['data']['test_list'] = str(test_file)
            
            # Use tokenizer-based charset
            charset_file = self.prepare_charset_from_tokenizer()
            self.config['charset'] = str(charset_file)
        
        # Save updated config
        updated_config = self.config_path.parent / f"tokenized_{self.config_path.name}"
        with open(updated_config, 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"✅ Config updated: {updated_config}")
        
        # Run training with updated config
        if PADDLE_AVAILABLE:
            import subprocess
            cmd = [
                sys.executable, "train/run.py",
                "--config", str(updated_config)
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)
        else:
            print("❌ PaddlePaddle not available")
            print("Install with: pip install paddlepaddle-gpu==2.6.1.post120")


def main():
    parser = argparse.ArgumentParser(description="Train with tokenizer")
    parser.add_argument(
        "--config",
        default="train/configs/rec_kh_hf.yaml",
        help="Training config file"
    )
    parser.add_argument(
        "--download-tokenizer",
        action="store_true",
        help="Download tokenizer for offline use"
    )
    
    args = parser.parse_args()
    
    if args.download_tokenizer:
        if HAS_TRANSFORMERS:
            print("Downloading khopilot/km-tokenizer-khmer...")
            tokenizer = AutoTokenizer.from_pretrained("khopilot/km-tokenizer-khmer")
            save_dir = Path("lang/tokenizer/khopilot")
            save_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(save_dir)
            print(f"✅ Tokenizer saved to {save_dir}")
        else:
            print("❌ Install transformers first: pip install transformers")
        return
    
    # Run training
    trainer = TokenizerIntegratedTrainer(Path(args.config))
    trainer.train()


if __name__ == "__main__":
    main()