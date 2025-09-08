"""Post-processing utilities for Khmer text"""

import re
import unicodedata
from typing import List, Optional


def normalize_khmer_text(text: str) -> str:
    """
    Normalize Khmer text with proper Unicode handling
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization (NFC form)
    text = unicodedata.normalize("NFC", text)
    
    # Remove zero-width spaces and joiners
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\u200c", "")  # Zero-width non-joiner
    text = text.replace("\u200d", "")  # Zero-width joiner
    
    # Fix common diacritic ordering issues
    text = fix_diacritic_order(text)
    
    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def fix_diacritic_order(text: str) -> str:
    """
    Fix Khmer diacritic ordering according to Unicode standards
    
    Args:
        text: Input text
        
    Returns:
        Text with corrected diacritic order
    """
    # Khmer Unicode ranges
    # Consonants: U+1780 to U+17A2
    # Dependent vowels: U+17B6 to U+17C5
    # Signs: U+17C6 to U+17D3
    
    # TODO: Implement proper diacritic reordering
    # For now, just return the text as-is
    return text


def validate_khmer_chars(text: str, charset_path: Optional[str] = None) -> bool:
    """
    Validate that all characters are in the allowed Khmer charset
    
    Args:
        text: Text to validate
        charset_path: Path to charset file
        
    Returns:
        True if all characters are valid
    """
    # Load charset
    if charset_path:
        with open(charset_path, "r", encoding="utf-8") as f:
            valid_chars = set(f.read().strip())
    else:
        # Default Khmer character ranges
        valid_chars = set()
        # Add Khmer consonants
        for code in range(0x1780, 0x17A3):
            valid_chars.add(chr(code))
        # Add Khmer vowels
        for code in range(0x17B6, 0x17C6):
            valid_chars.add(chr(code))
        # Add Khmer signs
        for code in range(0x17C6, 0x17D4):
            valid_chars.add(chr(code))
        # Add Khmer digits
        for code in range(0x17E0, 0x17EA):
            valid_chars.add(chr(code))
        # Add common punctuation and spaces
        valid_chars.update([" ", ".", ",", "!", "?", "(", ")", "-", ":", ";"])
    
    # Check each character
    for char in text:
        if char not in valid_chars:
            return False
    
    return True


def split_into_words(text: str) -> List[str]:
    """
    Split Khmer text into words
    
    Note: Khmer doesn't use spaces between words, so this is a simplified approach
    
    Args:
        text: Input text
        
    Returns:
        List of words
    """
    # For now, use a simple approach based on spaces and punctuation
    # TODO: Implement proper Khmer word segmentation
    words = re.split(r"[\s\u200b]+", text)
    return [w for w in words if w]


def remove_non_khmer(text: str) -> str:
    """
    Remove non-Khmer characters from text
    
    Args:
        text: Input text
        
    Returns:
        Text with only Khmer characters
    """
    # Keep only Khmer Unicode range and basic punctuation
    khmer_pattern = r"[\u1780-\u17FF\u19E0-\u19FF\u1A00-\u1A9F\s.,!?():;-]"
    khmer_chars = re.findall(khmer_pattern, text)
    return "".join(khmer_chars)


def apply_spelling_corrections(text: str, corrections_dict: Optional[dict] = None) -> str:
    """
    Apply common spelling corrections for OCR errors
    
    Args:
        text: Input text
        corrections_dict: Dictionary of corrections
        
    Returns:
        Corrected text
    """
    if not corrections_dict:
        # Common OCR mistakes in Khmer
        corrections_dict = {
            # Add common corrections here
            # Example: "wrong" -> "correct"
        }
    
    for wrong, correct in corrections_dict.items():
        text = text.replace(wrong, correct)
    
    return text


def calculate_text_stats(text: str) -> dict:
    """
    Calculate statistics about the text
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of statistics
    """
    words = split_into_words(text)
    
    return {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "has_khmer": bool(re.search(r"[\u1780-\u17FF]", text)),
        "has_digits": bool(re.search(r"[\u17E0-\u17E9]", text)),  # Khmer digits
        "has_punctuation": bool(re.search(r"[.,!?():;-]", text)),
    }