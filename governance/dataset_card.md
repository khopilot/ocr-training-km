# Khmer OCR v1 Dataset Card

## Overview
This document provides attribution and documentation for all datasets used in the Khmer OCR v1 training pipeline.

## Primary Data Sources

### 1. SynthKhmer-10k
- **Source**: [seanghay/SynthKhmer-10k](https://huggingface.co/datasets/seanghay/SynthKhmer-10k)
- **License**: CC-BY-4.0
- **Size**: 10,000 synthetic Khmer ID document images
- **Description**: Synthetic Khmer identity documents with bounding box annotations for text elements
- **Usage**: Training DBNet detection and CTC recognition models
- **Features**:
  - Full names in Khmer script
  - Date of birth annotations
  - Gender labels
  - Unique ID numbers
  - Bounding box coordinates for each text element

### 2. khmerfonts-info-previews
- **Source**: [seanghay/khmerfonts-info-previews](https://huggingface.co/datasets/seanghay/khmerfonts-info-previews)
- **License**: CC-BY-4.0
- **Size**: 26,591 font preview images
- **Description**: Font variations dataset with 2,972 Khmer fonts × 10 sentences
- **Usage**: Improving font variation robustness in recognition model
- **Features**:
  - Wide variety of Khmer font styles
  - Consistent text samples across fonts
  - High-quality font renderings

## Language Model & Lexicon Sources

### 3. khmer-dictionary-44k
- **Source**: [seanghay/khmer-dictionary-44k](https://huggingface.co/datasets/seanghay/khmer-dictionary-44k)
- **License**: Research use only (Royal Academy of Cambodia)
- **Size**: 44,706 dictionary entries
- **Description**: Comprehensive Khmer dictionary from Royal Academy of Cambodia 2022
- **Usage**: Building KenLM language model and lexicon for post-processing
- **Features**:
  - Word definitions
  - Part of speech tags
  - Pronunciation guides
  - Example usage

### 4. google-khmer-lexicon
- **Source**: [seanghay/google-khmer-lexicon](https://huggingface.co/datasets/seanghay/google-khmer-lexicon)
- **License**: Not specified
- **Size**: 69,414 word entries
- **Description**: Google's Khmer lexicon with pronunciations
- **Usage**: Supplementing lexicon for rescoring
- **Features**:
  - Extensive vocabulary coverage
  - Pronunciation annotations

### 5. lexicon-kh
- **Source**: [seanghay/lexicon-kh](https://huggingface.co/datasets/seanghay/lexicon-kh)
- **License**: Not specified
- **Size**: 3,770 vocabulary entries
- **Description**: Additional Khmer vocabulary entries
- **Usage**: Expanding lexicon coverage

### 6. km_large_text
- **Source**: [seanghay/km_large_text](https://huggingface.co/datasets/seanghay/km_large_text)
- **License**: Not specified
- **Size**: Variable
- **Description**: Large text corpus for language modeling
- **Usage**: Training KenLM n-gram language model

## Data Processing Pipeline

### Detection Training Data
1. **SynthKhmer-10k** → PaddleOCR detection format
   - Image paths with JSON bounding box annotations
   - 80% train, 10% validation, 10% test split

### Recognition Training Data
1. **SynthKhmer-10k** → PaddleOCR recognition format
   - Image patches with text labels
2. **khmerfonts-info-previews** → Recognition augmentation
   - Font variation samples

### Language Model Data
1. **khmer-dictionary-44k** + **google-khmer-lexicon** + **lexicon-kh** → Unified lexicon
   - Deduplicated vocabulary: ~100k unique words
   - Character set extraction: ~74 Khmer characters
2. **km_large_text** → KenLM 5-gram model training corpus

## Ethical Considerations

### Synthetic Data
- SynthKhmer-10k uses synthetic profile images from thispersondoesnotexist.com
- No real personal information is included
- Backgrounds from picsum.photos (stock imagery)

### Dictionary Data
- khmer-dictionary-44k is marked for research use only
- Commercial use requires permission from Royal Academy of Cambodia

### Font Data
- Font previews downloaded from khmerfonts.info
- Fonts may have individual licensing terms

## Usage Rights

### Open Source (CC-BY-4.0)
- SynthKhmer-10k
- khmerfonts-info-previews

### Research Use Only
- khmer-dictionary-44k

### Unspecified License
- google-khmer-lexicon
- lexicon-kh
- km_large_text

## Citation

If you use these datasets, please cite:

```bibtex
@misc{seanghay2024khmerocr,
  author = {seanghay},
  title = {Khmer OCR Datasets Collection},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/seanghay}
}

@misc{khopilot2024khmerocr,
  author = {khopilot},
  title = {Khmer OCR v1: Production-grade OCR for Khmer Text},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/khopilot/khmer-ocr-v1}
}
```

## Acknowledgments

Special thanks to:
- **seanghay** for curating and sharing these valuable Khmer language datasets
- **Royal Academy of Cambodia** for the comprehensive dictionary
- **khmerfonts.info** for font resources
- The Khmer NLP community for continuous support

## Contact

For questions about dataset usage or licensing:
- GitHub: [@khopilot](https://github.com/khopilot)
- HuggingFace: [@khopilot](https://huggingface.co/khopilot)

---
*Last updated: 2024*