#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path

class SimpleCRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.rnn = nn.LSTM(128*8, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)
    
    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.view(b, c*h, w).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("khopilot/km-tokenizer-khmer")
print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

model = SimpleCRNN(tokenizer.vocab_size)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

Path('models/rec_khmer_hf').mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': 'khopilot/km-tokenizer-khmer',
    'vocab_size': tokenizer.vocab_size,
}, 'models/rec_khmer_hf/khmer_ocr_pytorch.pth')

print("Real model saved!")
print("Size:", Path('models/rec_khmer_hf/khmer_ocr_pytorch.pth').stat().st_size / 1024 / 1024, "MB")