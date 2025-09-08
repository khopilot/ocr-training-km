#!/usr/bin/env python3
"""Generate synthetic dataset with comprehensive Khmer charset coverage.

Creates PNG images with Khmer text designed to achieve ≥99% charset coverage.
Uses systematic generation to ensure all characters in charset_kh.txt are represented.
"""

import argparse
import random
from pathlib import Path
from typing import List, Set
import math

from PIL import Image, ImageDraw, ImageFont


def load_charset(charset_path: Path) -> List[str]:
    """Load charset from file, excluding whitespace and empty lines"""
    with open(charset_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f if line.strip()]
    return chars


def generate_charset_samples(charset: List[str], rnd: random.Random) -> List[str]:
    """Generate text samples that systematically cover all charset characters"""
    samples = []
    
    # Group characters by type for more natural combinations
    consonants = [c for c in charset if 'ក' <= c <= 'អ']
    vowels = [c for c in charset if 'ា' <= c <= 'ៅ'] 
    diacritics = [c for c in charset if 'ំ' <= c <= '៑']
    numbers = [c for c in charset if '០' <= c <= '៹']
    punctuation = [c for c in charset if c in '!"\'(),-./0123456789:;?[]']
    
    # 1. Individual character samples (ensure every char appears at least once)
    for char in charset:
        if char.strip():  # Skip whitespace
            samples.append(char)
    
    # 2. Character pairs and combinations
    for i in range(min(100, len(consonants))):
        base = rnd.choice(consonants)
        if vowels:
            vowel = rnd.choice(vowels)
            samples.append(base + vowel)
        if diacritics:
            diac = rnd.choice(diacritics)
            samples.append(base + diac)
    
    # 3. Simple syllables (consonant + vowel + optional diacritic)
    for i in range(50):
        parts = []
        if consonants:
            parts.append(rnd.choice(consonants))
        if vowels and rnd.random() > 0.3:
            parts.append(rnd.choice(vowels))
        if diacritics and rnd.random() > 0.7:
            parts.append(rnd.choice(diacritics))
        if parts:
            samples.append(''.join(parts))
    
    # 4. Number sequences
    for i in range(20):
        num_chars = rnd.choices(numbers, k=rnd.randint(2, 5))
        samples.append(''.join(num_chars))
    
    # 5. Mixed punctuation and text
    for i in range(30):
        parts = []
        if consonants and rnd.random() > 0.3:
            parts.append(rnd.choice(consonants))
        if punctuation and rnd.random() > 0.5:
            parts.append(rnd.choice(punctuation))
        if vowels and rnd.random() > 0.4:
            parts.append(rnd.choice(vowels))
        if parts:
            samples.append(''.join(parts))
    
    # 6. Longer sequences (2-4 characters)
    for i in range(100):
        char_count = rnd.randint(2, 4)
        chars_selected = rnd.choices(charset, k=char_count)
        # Filter out consecutive whitespace/newlines
        cleaned = ''.join(c for c in chars_selected if c.strip())
        if cleaned:
            samples.append(cleaned)
    
    return [s for s in samples if s.strip()]  # Remove empty samples


def make_image(text: str, out_path: Path, size=(512, 128)) -> None:
    """Generate synthetic image with text"""
    # Create white background
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a Khmer font if available, fallback to default
    font = None
    try:
        # Common Khmer fonts on different systems
        font_paths = [
            "/System/Library/Fonts/KhmerSangamMN.ttc",  # macOS
            "/usr/share/fonts/truetype/khmer/KhmerOS.ttf",  # Linux
        ]
        for font_path in font_paths:
            if Path(font_path).exists():
                font = ImageFont.truetype(font_path, 32)
                break
    except:
        pass
    
    # Position text
    text_x = 10
    text_y = size[1] // 3
    
    if font:
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    else:
        # Fallback to default font
        draw.text((text_x, text_y), text, fill=(0, 0, 0))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OCR data with full charset coverage")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/synth", help="Output directory")
    parser.add_argument("--charset", type=str, default="train/charset_kh.txt", help="Charset file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    # Load charset
    charset_path = Path(args.charset)
    if not charset_path.exists():
        print(f"Error: Charset file not found: {charset_path}")
        return
    
    charset = load_charset(charset_path)
    print(f"Loaded charset: {len(charset)} characters")
    
    rnd = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / "label.txt"

    # Generate diverse text samples targeting full charset coverage
    text_samples = generate_charset_samples(charset, rnd)
    print(f"Generated {len(text_samples)} base text samples")
    
    # Generate requested number of samples by cycling through text samples
    lines: List[str] = []
    for i in range(args.samples):
        # Cycle through text samples, adding some randomization
        text_idx = i % len(text_samples)
        text = text_samples[text_idx]
        
        # Occasionally combine multiple samples for longer text
        if rnd.random() > 0.8 and len(text_samples) > text_idx + 1:
            text += " " + text_samples[text_idx + 1]
        
        img_name = f"synth_{i:05d}.png"
        img_path = out_dir / img_name
        make_image(text, img_path)
        lines.append(f"{img_name}\t{text}\n")

    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"✓ Generated {args.samples} synthetic samples at {out_dir}")


if __name__ == "__main__":
    main()

