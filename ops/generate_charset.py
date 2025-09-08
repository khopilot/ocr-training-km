#!/usr/bin/env python3
"""Generate Khmer character set for OCR training"""

import argparse
from pathlib import Path
from typing import Set, List


def generate_khmer_charset() -> Set[str]:
    """
    Generate complete Khmer character set
    
    Returns:
        Set of Khmer characters
    """
    charset = set()
    
    # Khmer consonants (ក to អ)
    for code in range(0x1780, 0x17A3):
        charset.add(chr(code))
    
    # Khmer independent vowels
    for code in range(0x17A3, 0x17A5):
        charset.add(chr(code))
    
    # Khmer inherent vowels
    for code in range(0x17A5, 0x17B4):
        charset.add(chr(code))
    
    # Khmer dependent vowels
    for code in range(0x17B6, 0x17C6):
        charset.add(chr(code))
    
    # Khmer signs
    for code in range(0x17C6, 0x17D4):
        charset.add(chr(code))
    
    # Khmer punctuation
    charset.add(chr(0x17D4))  # KHMER SIGN KHAN
    charset.add(chr(0x17D5))  # KHMER SIGN BARIYOOSAN
    charset.add(chr(0x17D6))  # KHMER SIGN CAMNUC PII KUUH
    charset.add(chr(0x17D7))  # KHMER SIGN LEK TOO
    charset.add(chr(0x17D8))  # KHMER SIGN BEYYAL
    charset.add(chr(0x17D9))  # KHMER SIGN PHNAEK MUAN
    charset.add(chr(0x17DA))  # KHMER SIGN KOOMUUT
    
    # Khmer digits (០ to ៩)
    for code in range(0x17E0, 0x17EA):
        charset.add(chr(code))
    
    # Khmer symbols
    charset.add(chr(0x17F0))  # KHMER SYMBOL LEK ATTAK SON
    charset.add(chr(0x17F1))  # KHMER SYMBOL LEK ATTAK MUOY
    charset.add(chr(0x17F2))  # KHMER SYMBOL LEK ATTAK PII
    charset.add(chr(0x17F3))  # KHMER SYMBOL LEK ATTAK BEI
    charset.add(chr(0x17F4))  # KHMER SYMBOL LEK ATTAK BUON
    charset.add(chr(0x17F5))  # KHMER SYMBOL LEK ATTAK PRAM
    charset.add(chr(0x17F6))  # KHMER SYMBOL LEK ATTAK PRAM-MUOY
    charset.add(chr(0x17F7))  # KHMER SYMBOL LEK ATTAK PRAM-PII
    charset.add(chr(0x17F8))  # KHMER SYMBOL LEK ATTAK PRAM-BEI
    charset.add(chr(0x17F9))  # KHMER SYMBOL LEK ATTAK PRAM-BUON
    
    # Common punctuation and special characters
    charset.update([
        " ",    # Space
        ".",    # Period
        ",",    # Comma
        "!",    # Exclamation
        "?",    # Question mark
        ":",    # Colon
        ";",    # Semicolon
        "-",    # Hyphen
        "(",    # Left parenthesis
        ")",    # Right parenthesis
        "[",    # Left bracket
        "]",    # Right bracket
        "\"",   # Quote
        "'",    # Apostrophe
        "/",    # Slash
        "\n",   # Newline (for multi-line text)
    ])
    
    # Arabic numerals (often mixed with Khmer text)
    for digit in "0123456789":
        charset.add(digit)
    
    # Basic Latin letters (sometimes appear in mixed text)
    # Uncomment if needed for your use case
    # for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
    #     charset.add(c)
    
    return charset


def save_charset(charset: Set[str], output_path: Path):
    """
    Save character set to file
    
    Args:
        charset: Set of characters
        output_path: Path to output file
    """
    # Sort characters by Unicode code point
    sorted_chars = sorted(charset, key=lambda x: ord(x) if x else 0)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for char in sorted_chars:
            if char == "\n":
                f.write("\\n\n")  # Escape newline
            else:
                f.write(f"{char}\n")
    
    print(f"✅ Generated charset with {len(charset)} characters")
    print(f"   Saved to: {output_path}")


def validate_charset(charset_path: Path):
    """
    Validate and display charset statistics
    
    Args:
        charset_path: Path to charset file
    """
    with open(charset_path, "r", encoding="utf-8") as f:
        chars = [line.strip() for line in f if line.strip()]
    
    # Replace escaped newline
    chars = ["\\n" if c == "\\n" else c for c in chars]
    
    # Statistics
    stats = {
        "total": len(chars),
        "khmer_consonants": 0,
        "khmer_vowels": 0,
        "khmer_signs": 0,
        "khmer_digits": 0,
        "punctuation": 0,
        "arabic_digits": 0,
        "latin": 0,
        "other": 0,
    }
    
    for char in chars:
        if char == "\\n":
            stats["other"] += 1
        else:
            code = ord(char)
            if 0x1780 <= code <= 0x17A2:
                stats["khmer_consonants"] += 1
            elif 0x17A3 <= code <= 0x17C5:
                stats["khmer_vowels"] += 1
            elif 0x17C6 <= code <= 0x17DD:
                stats["khmer_signs"] += 1
            elif 0x17E0 <= code <= 0x17E9:
                stats["khmer_digits"] += 1
            elif char in ".,!?:;-()[]\"'/":
                stats["punctuation"] += 1
            elif char in "0123456789":
                stats["arabic_digits"] += 1
            elif char.isalpha() and char.isascii():
                stats["latin"] += 1
            else:
                stats["other"] += 1
    
    print("\n📊 Charset Statistics:")
    print(f"   Total characters: {stats['total']}")
    print(f"   Khmer consonants: {stats['khmer_consonants']}")
    print(f"   Khmer vowels: {stats['khmer_vowels']}")
    print(f"   Khmer signs: {stats['khmer_signs']}")
    print(f"   Khmer digits: {stats['khmer_digits']}")
    print(f"   Punctuation: {stats['punctuation']}")
    print(f"   Arabic digits: {stats['arabic_digits']}")
    if stats['latin'] > 0:
        print(f"   Latin letters: {stats['latin']}")
    if stats['other'] > 0:
        print(f"   Other: {stats['other']}")


def main():
    """CLI interface for charset generation"""
    parser = argparse.ArgumentParser(description="Generate Khmer character set")
    parser.add_argument(
        "--output",
        type=str,
        default="train/charset_kh.txt",
        help="Output file path"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing charset file"
    )
    parser.add_argument(
        "--include-latin",
        action="store_true",
        help="Include basic Latin letters"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    if args.validate and output_path.exists():
        validate_charset(output_path)
    else:
        # Generate charset
        charset = generate_khmer_charset()
        
        # Optionally add Latin letters
        if args.include_latin:
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
                charset.add(c)
            print("   Added Latin letters to charset")
        
        # Save to file
        save_charset(charset, output_path)
        
        # Validate
        validate_charset(output_path)


if __name__ == "__main__":
    main()