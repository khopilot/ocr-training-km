# CLAUDE.md — Khmer OCR v1 Project Configuration

## Project Context
**Project**: Khmer OCR v1 - Production-grade OCR system for Khmer text
**Architecture**: PaddleOCR (DBNet+CTC) with KenLM language rescoring
**Target**: CER ≤3% clean print, ≤10% degraded; latency ≤200ms/page (GPU)
**Platform**: macOS development → Linux x86_64 deployment (CUDA 12.x)

## Operating Mode
- **Style**: Small PR-sized diffs, deterministic seeds, pinned versions
- **Outputs**: Code first, then tests, then docs. No speculative dependencies
- **Security**: No external calls with secrets; keep sample data synthetic
- **Commits**: Meticulous ML engineer approach - small, reviewable commits with tests

## Repository Structure
```
kh-ocr/
  data/           # raw/, labels/, synth/ - training and evaluation data
  train/          # configs/, dbnet.yaml, rec_kh.yaml - training configurations
  infer/          # engine.py, postproc.py, rescoring.py - inference pipeline
  lang/           # km_tokenizer/, kenlm/, lexicon/ - language models
  eval/           # cer.py, harness.py, samples/ - evaluation tools
  service/        # app.py, routers/, schemas.py, metrics.py - FastAPI service
  ops/            # download.py, split.py, synth.py - data operations
  governance/     # license_ledger.csv, dataset_card.md, manifest.json
  pyproject.toml  # Dependencies with pinned versions
  Makefile        # setup, train, infer, serve, bench targets
```

## Technical Stack
### Core Dependencies
- **OCR Engine**: PaddleOCR 2.7.0 with PaddlePaddle GPU 2.6.1
- **Language Model**: KenLM 5-gram with khopilot/km-tokenizer-khmer
- **API Framework**: FastAPI 0.112.0 with Pydantic 2.8.2
- **Runtime**: Python 3.11, CUDA 12.x, optional TensorRT 9.x
- **Observability**: Prometheus metrics, trace IDs, structured logging

### Development Tools
- **Testing**: pytest with CER evaluation harness
- **Formatting**: Ruff (line-length=100)
- **Pre-commit**: Security checks, no credentials
- **Versioning**: Semantic versioning with SHA256 hashes in manifest.json

## Implementation Guidelines

### Character Recognition
- Use CTC at character level (not subwords)
- Generate charset_kh.txt with Khmer letters + diacritics
- Implement Unicode normalization and diacritic rules in postproc.py

### Language Rescoring
```python
# Combine CTC logprob + λ·KenLM + μ·lexicon penalties
def rescore(ctc_logprob, tokens, kenlm, lam, mu, lexicon):
    lm_lp = kenlm.score(" ".join(tokens))
    lex_pen = 0.0 if all(t in lexicon for t in tokens) else -1.0
    return ctc_logprob + lam * lm_lp + mu * lex_pen
```

### API Contract
```python
class OCRResponse(BaseModel):
    text: str
    lines: List[Line]  # bbox, text, conf
    version: dict
    metrics: dict
    pii_report: dict | None = None
```

## Guardrails
- Never change API schemas without updating schemas.py and manifest.json
- Run pre-commit before any commit
- All models/artifacts must have SHA256 in manifest.json
- Test CER on both clean and degraded samples
- Monitor p95 latency ≤200ms/page (GPU batch)

## Acceptance Criteria
1. **CER Performance**: ≤3% clean print, ≤10% degraded
2. **Latency**: p95 ≤200ms/page on GPU batch
3. **Reproducibility**: Full manifest with hashes
4. **API**: Validated contract with metrics
5. **Ablations**: CTC → +KenLM → +KenLM+lexicon with CER deltas

## Development Workflow
1. Use Makefile targets: `make setup`, `make train`, `make eval`, `make serve`
2. Run evaluation harness after each model change
3. Generate manifest.json before releases
4. Cross-check with Tesseract on 5% samples (optional)
5. Log model versions and hashes per request

## Next Steps (v1.1 backlog)
- Complex layouts (tables/columns)
- PII/DLP blocking mode
- TensorRT FP16/INT8 optimization
- Active learning for hard samples

## Commands & Scripts
- `make setup`: Initialize virtual environment
- `make download`: Fetch training data
- `make train`: Train DBNet + Recognition models
- `make infer`: Run inference on samples
- `make serve`: Start FastAPI server
- `make eval`: Run CER evaluation harness
- `make bench`: Performance benchmarking

## Contact & Support
- GitHub: [@khopilot](https://github.com/khopilot)
- HuggingFace: [@khopilot](https://huggingface.co/khopilot)