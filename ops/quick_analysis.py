#!/usr/bin/env python3
"""Quick charset analysis script"""

import sys
sys.path.append('.')
from ops.analyze_charset import load_charset, analyze_label_file
from pathlib import Path

# Load charset
charset = load_charset(Path('train/charset_kh.txt'))
print(f'Reference charset: {len(charset)} characters')

# Analyze new synthetic data
chars_found = analyze_label_file(Path('data/synth_new/label.txt'))
coverage_pct = len(chars_found & charset) / len(charset) * 100
missing_chars = charset - chars_found

print(f'Characters found: {len(chars_found)}')
print(f'Charset coverage: {len(chars_found & charset)}/{len(charset)} ({coverage_pct:.1f}%)')
print(f'Missing chars: {len(missing_chars)}')
if missing_chars:
    print(f'Missing: {list(missing_chars)[:20]}')

accept_threshold = 99.0
status = 'PASS' if coverage_pct >= accept_threshold else 'FAIL'
print(f'\nACCEPTANCE: {status}')
print(f'Target: >={accept_threshold}%, Actual: {coverage_pct:.1f}%')