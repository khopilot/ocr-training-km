"""Unit tests for Khmer text normalization"""

import pytest
import unicodedata
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from infer.postproc import normalize_khmer_text
from lang.tokenizers import KhmerTokenizer


class TestKhmerNormalization:
    """Test Khmer text normalization functions"""
    
    def test_zero_width_removal(self):
        """Test removal of zero-width characters"""
        text = "ខ្មែរ\u200bភាសា\u200c\u200d"  # With ZWS, ZWNJ, ZWJ
        normalized = normalize_khmer_text(text)
        assert "\u200b" not in normalized
        assert "\u200c" not in normalized
        assert "\u200d" not in normalized
        assert "ខ្មែរភាសា" in normalized
    
    def test_nfc_normalization(self):
        """Test Unicode NFC normalization"""
        # Test with decomposed characters
        text_nfd = unicodedata.normalize('NFD', "ខ្មែរ")
        text_nfc = normalize_khmer_text(text_nfd)
        assert unicodedata.is_normalized('NFC', text_nfc)
    
    def test_diacritic_preservation(self):
        """Test that diacritics are preserved"""
        text = "ស្រៈ ស្រី ក្រុម"  # Text with various diacritics
        normalized = normalize_khmer_text(text)
        
        # Count diacritics
        original_diacritics = sum(1 for c in text if 0x17B4 <= ord(c) <= 0x17D3)
        normalized_diacritics = sum(1 for c in normalized if 0x17B4 <= ord(c) <= 0x17D3)
        
        assert original_diacritics == normalized_diacritics
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization"""
        text = "ខ្មែរ   ភាសា\t\nយើង"
        normalized = normalize_khmer_text(text)
        assert "  " not in normalized
        assert "\t" not in normalized
        assert normalized == "ខ្មែរ ភាសា យើង"
    
    def test_empty_string(self):
        """Test handling of empty strings"""
        assert normalize_khmer_text("") == ""
        assert normalize_khmer_text("   ") == ""
    
    def test_mixed_script(self):
        """Test handling of mixed Khmer-English text"""
        text = "Hello ស្វាគមន៍ World"
        normalized = normalize_khmer_text(text)
        assert "Hello" in normalized
        assert "World" in normalized
        assert "ស្វាគមន៍" in normalized


class TestDiacriticReordering:
    """Test diacritic reordering logic"""
    
    def test_identify_diacritics(self):
        """Test diacritic identification"""
        from eval.cer import is_khmer_diacritic
        
        # Test known diacritics
        assert is_khmer_diacritic('\u17C6')  # KHMER SIGN NIKAHIT
        assert is_khmer_diacritic('\u17D2')  # KHMER SIGN COENG
        
        # Test non-diacritics
        assert not is_khmer_diacritic('ក')  # Consonant
        assert not is_khmer_diacritic('a')  # Latin
    
    def test_extract_diacritics(self):
        """Test diacritic extraction"""
        from eval.cer import extract_diacritics
        
        text = "ក្រុម"  # Group with coeng
        base, diacritics = extract_diacritics(text)
        
        assert len(base) < len(text)
        assert len(diacritics) > 0
        assert '\u17D2' in diacritics


class TestScoreAlignment:
    """Test score scale conversion"""
    
    def test_log10_to_ln_conversion(self):
        """Test conversion from log10 to ln"""
        import math
        
        log10_score = -2.0
        ln_score = log10_score * math.log(10)
        
        # Check conversion factor
        assert abs(ln_score - (-4.605)) < 0.01
    
    def test_score_combination(self):
        """Test combining CTC and LM scores"""
        from infer.rescoring import rescore_with_lm
        
        ctc_logprob = -5.0  # ln scale
        tokens = ["សួស្តី", "ពិភពលោក"]
        
        # Without LM
        score = rescore_with_lm(ctc_logprob, tokens, None, lam=0.3, mu=0.1)
        assert score == ctc_logprob
        
        # With empty lexicon
        score = rescore_with_lm(ctc_logprob, tokens, None, lam=0.3, mu=0.1, lexicon=set())
        assert score <= ctc_logprob  # Penalty for OOV


class TestLabelParsing:
    """Test label file parsing"""
    
    def test_parse_label_line(self):
        """Test parsing of label.txt lines"""
        line = "image001.jpg\tសួស្តី ពិភពលោក"
        parts = line.strip().split("\t", 1)
        
        assert len(parts) == 2
        assert parts[0] == "image001.jpg"
        assert parts[1] == "សួស្តី ពិភពលោក"
    
    def test_parse_empty_lines(self):
        """Test handling of empty lines"""
        lines = [
            "image1.jpg\ttext1",
            "",
            "  ",
            "image2.jpg\ttext2"
        ]
        
        valid_lines = [l for l in lines if l.strip() and "\t" in l]
        assert len(valid_lines) == 2
    
    def test_parse_unicode(self):
        """Test parsing with Unicode characters"""
        line = "ឯកសារ.jpg\tអត្ថបទខ្មែរ"
        filename, text = line.split("\t", 1)
        
        assert filename == "ឯកសារ.jpg"
        assert text == "អត្ថបទខ្មែរ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])