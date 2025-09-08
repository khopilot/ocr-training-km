# Datasets

This project expects PaddleOCR-style datasets:

- Recognition: `label.txt` with lines `img_path<TAB>label`
- Detection: `det_label.txt` with lines `img_path<TAB>[{"transcription":"…","points":[[x,y],…]}]`

Images are stored alongside the labels (relative paths).

## Hugging Face

Use `ops/download.py` to fetch datasets from Hugging Face into `data/raw/`.

Examples:

```bash
# From config file (data/datasets.yaml)
python ops/download.py --config data/datasets.yaml --out data/raw

# Single repo id
python ops/download.py --repo-id khopilot/khmer-ocr-recognition --out data/raw

# Force git clone fallback
python ops/download.py --repo-id khopilot/khmer-ocr-recognition --strategy git
```

Note: Large datasets should not be committed to git. Use this script to download locally or in CI.

## Splitting

If you only have a single `label.txt`, split into train/val/test via:

```bash
python ops/split.py --ratio 0.8:0.1:0.1 --raw data/raw/label.txt
```

## Validation

Validate dataset integrity and charset coverage (≥99%):

```bash
python ops/validate_dataset.py --data-dir data --charset train/charset_kh.txt --strict
```

