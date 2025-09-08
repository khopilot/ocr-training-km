#!/usr/bin/env python3
"""Generate synthetic Khmer text data for OCR training"""

import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# Khmer Unicode ranges and common characters
KHMER_CONSONANTS = [chr(i) for i in range(0x1780, 0x17A3)]  # ក to អ
KHMER_VOWELS = [chr(i) for i in range(0x17B6, 0x17C6)]  # ា to ៅ
KHMER_SIGNS = [chr(i) for i in range(0x17C6, 0x17D4)]  # ំ to ៓
KHMER_DIGITS = [chr(i) for i in range(0x17E0, 0x17EA)]  # ០ to ៩

# Common Khmer words (examples)
KHMER_WORDS = [
    "សួស្តី", "អរគុណ", "សូម", "បាទ", "ចា", "ទេ", "មាន", "គ្មាន",
    "ខ្ញុំ", "អ្នក", "យើង", "គាត់", "នាង", "ពួកគេ", "នេះ", "នោះ",
    "ថ្ងៃ", "ខែ", "ឆ្នាំ", "ព្រឹក", "ល្ងាច", "យប់", "ម៉ោង", "នាទី",
    "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី",
    "កម្ពុជា", "ភ្នំពេញ", "សៀមរាប", "អង្គរវត្ត", "បាយ័ន", "ប្រាសាទ",
    "រៀន", "សាលា", "សិស្ស", "គ្រូ", "សៀវភៅ", "អាន", "សរសេរ", "និយាយ",
]

def generate_khmer_text(length="word"):
    """Generate random Khmer text"""
    if length == "word":
        return random.choice(KHMER_WORDS)
    elif length == "sentence":
        num_words = random.randint(3, 8)
        return " ".join([random.choice(KHMER_WORDS) for _ in range(num_words)])
    elif length == "char":
        # Generate random character sequence
        num_chars = random.randint(1, 5)
        chars = []
        for _ in range(num_chars):
            if random.random() < 0.6:
                chars.append(random.choice(KHMER_CONSONANTS))
            elif random.random() < 0.8:
                chars.append(random.choice(KHMER_VOWELS))
            else:
                chars.append(random.choice(KHMER_DIGITS))
        return "".join(chars)
    return random.choice(KHMER_WORDS)

def create_text_image(text, font_size=32, img_width=400, img_height=64):
    """Create an image with Khmer text"""
    # Create white background
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font that supports Khmer
    # You might need to download Khmer fonts
    try:
        # Try system fonts first
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf",
            "/usr/share/fonts/truetype/khmer/KhmerOS.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            font = ImageFont.load_default()
            
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center text
    x = (img_width - text_width) // 2
    y = (img_height - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill='black', font=font)
    
    # Add some noise/augmentation
    if random.random() < 0.3:
        # Add noise
        noise = np.random.normal(0, 10, (img_height, img_width, 3))
        img_array = np.array(img).astype(float)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    return img

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic Khmer OCR training data"""
    
    print("🎨 Generating synthetic Khmer text data...")
    
    # Create directories
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    
    train_samples = int(num_samples * train_ratio)
    val_samples = int(num_samples * val_ratio)
    test_samples = num_samples - train_samples - val_samples
    
    train_list = []
    val_list = []
    test_list = []
    
    # Generate training data
    print(f"Generating {train_samples} training samples...")
    for i in range(train_samples):
        text = generate_khmer_text(random.choice(["word", "sentence", "char"]))
        img = create_text_image(text)
        
        img_path = f"data/train/synth_{i:06d}.png"
        txt_path = f"data/train/synth_{i:06d}.txt"
        
        img.save(img_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        train_list.append(f"{img_path}\t{text}")
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{train_samples} training samples")
    
    # Generate validation data
    print(f"Generating {val_samples} validation samples...")
    for i in range(val_samples):
        text = generate_khmer_text(random.choice(["word", "sentence"]))
        img = create_text_image(text)
        
        img_path = f"data/val/synth_{i:06d}.png"
        txt_path = f"data/val/synth_{i:06d}.txt"
        
        img.save(img_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        val_list.append(f"{img_path}\t{text}")
    
    # Generate test data
    print(f"Generating {test_samples} test samples...")
    for i in range(test_samples):
        text = generate_khmer_text(random.choice(["word", "sentence"]))
        img = create_text_image(text)
        
        img_path = f"data/test/synth_{i:06d}.png"
        txt_path = f"data/test/synth_{i:06d}.txt"
        
        img.save(img_path)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        test_list.append(f"{img_path}\t{text}")
    
    # Save list files
    with open("data/train_list.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(train_list))
    
    with open("data/val_list.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(val_list))
    
    with open("data/test_list.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(test_list))
    
    print(f"\n✅ Generated {num_samples} synthetic samples!")
    print(f"  Training: {len(train_list)}")
    print(f"  Validation: {len(val_list)}")
    print(f"  Test: {len(test_list)}")
    
    return train_list, val_list, test_list

if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    generate_synthetic_data(num_samples)