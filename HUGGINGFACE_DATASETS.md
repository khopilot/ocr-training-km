# HuggingFace Datasets Integration Guide

## Overview
This guide explains how to use the publicly available Khmer datasets from seanghay's HuggingFace profile for training the Khmer OCR v1 system.

## Quick Start

```bash
# 1. Setup environment
make setup

# 2. Download and prepare HuggingFace datasets
make download-hf

# 3. Train with HF datasets
make train CONFIG=train/configs/rec_kh_hf.yaml
```

## Available Datasets

### OCR Training Data
1. **SynthKhmer-10k** - 10,000 synthetic ID documents
2. **khmerfonts-info-previews** - 26,591 font variations

### Language Resources
3. **khmer-dictionary-44k** - Royal Academy dictionary
4. **google-khmer-lexicon** - 69k vocabulary entries
5. **lexicon-kh** - Additional vocabulary
6. **km_large_text** - Text corpus for language modeling

## Data Pipeline

### 1. Download Datasets
```bash
python ops/download_hf.py \
    --output-dir data/hf_datasets \
    --datasets all \
    --priority high
```

Options:
- `--datasets`: Choose specific datasets or "all"
- `--priority`: Filter by priority (high/medium/low/all)
- `--no-cache`: Force fresh download

### 2. Convert to PaddleOCR Format
```bash
python ops/convert_to_paddle.py \
    --input-dir data/hf_datasets \
    --output-dir data/paddle_format \
    --dataset all
```

This creates:
- Detection format: `data/paddle_format/detection/`
- Recognition format: `data/paddle_format/recognition/`

### 3. Build Unified Lexicon
```bash
python ops/build_lexicon.py \
    --input-dir data/hf_datasets \
    --output-dir lang/lexicon \
    --sources all
```

Outputs:
- `khmer_lexicon.txt` - Word list for post-processing
- `khmer_lexicon_kenlm.txt` - Format for KenLM training
- `charset_khmer.txt` - Extracted character set

## Training Configuration

### Detection Model (DBNet)
Use the HF-specific config:
```yaml
# train/configs/dbnet_hf.yaml
data:
  train_list: data/paddle_format/detection/train/label.txt
  val_list: data/paddle_format/detection/val/label.txt
```

### Recognition Model (CTC)
Use combined datasets:
```yaml
# train/configs/rec_kh_hf.yaml
data:
  train_lists:
    - data/paddle_format/recognition/train/label.txt
    - data/paddle_format/fonts_recognition/label.txt
```

## Directory Structure

After running `make download-hf`:
```
data/
├── hf_datasets/           # Downloaded HF datasets
│   ├── SynthKhmer-10k/
│   ├── khmerfonts-info-previews/
│   └── ...
├── paddle_format/         # Converted training data
│   ├── detection/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── recognition/
│       ├── train/
│       ├── val/
│       └── test/
└── lang/
    └── lexicon/           # Built lexicon files
        ├── khmer_lexicon.txt
        ├── charset_khmer.txt
        └── ...
```

## Training Commands

### Train Detection Model
```bash
python train/run.py \
    --config train/configs/dbnet_hf.yaml \
    --mode detection
```

### Train Recognition Model
```bash
python train/run.py \
    --config train/configs/rec_kh_hf.yaml \
    --mode recognition
```

### Combined Training
```bash
make train-hf  # Trains both models with HF data
```

## Evaluation

Evaluate on HF test sets:
```bash
python eval/harness.py \
    --model-dir models/rec_khmer_hf \
    --test-list data/paddle_format/recognition/test/label.txt
```

## License Compliance

⚠️ **Important License Information:**

- **CC-BY-4.0**: SynthKhmer-10k, khmerfonts-info-previews
- **Research Use Only**: khmer-dictionary-44k (Royal Academy)
- **Unspecified**: Other lexicon datasets

For commercial use, ensure compliance with each dataset's license.

## Performance Expectations

With HF datasets, expect:
- **Detection mAP**: 85-90% on synthetic IDs
- **Recognition CER**: 
  - Clean print: ≤3%
  - Font variations: ≤5%
  - With language model: ≤2%

## Troubleshooting

### Missing Dependencies
```bash
pip install datasets huggingface_hub pillow numpy
```

### Download Failures
- Check network connection
- Use `--no-cache` to force fresh download
- Try git clone fallback method

### Memory Issues
For large datasets:
```bash
python ops/convert_to_paddle.py \
    --max-font-samples 5000  # Limit font samples
```

## Citation

If using these datasets, please cite:
```bibtex
@misc{seanghay2024khmerocr,
  author = {seanghay},
  title = {Khmer OCR Datasets},
  year = {2024},
  publisher = {HuggingFace}
}
```

## Contact

- Issues: [GitHub](https://github.com/khopilot/khmer-ocr-v1/issues)
- Datasets: [seanghay on HuggingFace](https://huggingface.co/seanghay)