# claude.md — Khmer OCR Build Plan (Claude Code)

> Goal: enable Claude Code to scaffold, implement, test, and package a **Khmer OCR v1** that meets production acceptance. Keep scope tight; defer complex layout/handwriting to later.

---

## 0) Operating mode for Claude Code

- **Style**: small PR-sized diffs, deterministic seeds, pinned versions.
- **Outputs**: code first, then tests, then docs. No speculative dependencies.
- **Constraints**: Linux x86\_64, CUDA 12.x, Python 3.11, optional TensorRT 9.x.
- **Security**: no external calls with secrets; keep sample data synthetic.

### Guardrails

- Do not change API schemas without updating `schemas.py` and bumping `manifest.json` semver.
- Never commit credentials; run `pre-commit` locally.
- All models/artifacts must have SHA256 in `manifest.json`.

---

## 1) System prompt for Claude Code

Use the following instruction as the system message before coding:

```text
You are a meticulous ML engineer. Build a Khmer OCR pipeline with PaddleOCR (DBNet+CTC), plus a language rescoring layer (KenLM 5-gram) powered by the SentencePiece tokenizer `khopilot/km-tokenizer-khmer`. Keep CTC at character level; do not use subwords inside the CTC head. Ship a REST API with observability. Target acceptance: CER ≤3% clean print, ≤10% degraded; latency ≤200 ms/page p95 (GPU batch). Produce small, reviewable commits with tests.
```

---

## 2) Repository plan

```
kh-ocr/
  data/           # raw/, labels/, synth/
  train/          # configs/, dbnet.yaml, rec_kh.yaml
  infer/          # engine.py, postproc.py, rescoring.py
  lang/           # km_tokenizer/, kenlm/, lexicon/
  eval/           # cer.py, harness.py, samples/
  service/        # app.py, routers/, schemas.py, metrics.py, Dockerfile
  ops/            # download.py, split.py, synth.py, manifests.py
  governance/     # license_ledger.csv, dataset_card.md, manifest.json
  pyproject.toml  # pinned deps
  Makefile        # setup, train, infer, serve, bench
  .pre-commit-config.yaml
```

---

## 3) Dependency spec (pyproject excerpt)

```toml
[project]
name = "kh-ocr"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "paddlepaddle-gpu==2.6.1; platform_system=='Linux'",
  "paddleocr==2.7.0",
  "onnxruntime-gpu==1.18.1",
  "kenlm @ git+https://github.com/kpu/kenlm.git#subdirectory=python",
  "sentencepiece==0.1.99",
  "fastapi==0.112.0",
  "uvicorn[standard]==0.30.5",
  "numpy==1.26.4", "pandas==2.2.2",
  "prometheus-client==0.20.0",
  "pydantic==2.8.2",
  "python-multipart==0.0.9",
]
[tool.ruff]
line-length = 100
```

---

## 4) Initial tasks (Claude Code checklist)

1. **Scaffold repo**: create directories, `pyproject.toml`, `Makefile`, pre-commit.
2. **Charset**: generate `train/charset_kh.txt` (Khmer letters + diacritics).
3. **Configs**: add `train/configs/dbnet.yaml` and `rec_kh.yaml` (CTC char-level).
4. **Training script**: glue around PaddleOCR to launch train from YAML; log CER.
5. **Tokenizer & LM**: script to tokenize corpora with `khopilot/km-tokenizer-khmer`; train KenLM 5-gram; save `.arpa` + binary.
6. **Inference engine**: implement `infer/engine.py` (DBNet→Rec→Normalize) and `postproc.py` (Unicode/diacritics rules).
7. **Rescoring**: implement `infer/rescoring.py` combining CTC logprob + λ·KenLM + μ·lexicon penalties.
8. **Tesseract cross-check**: optional `ops/tess_check.py` on 5% sample; produce triage report.
9. **API**: build FastAPI `POST /ocr`; add Prom metrics and trace id.
10. **Eval harness**: implement `eval/cer.py`, `eval/harness.py`, plus synthetic stress set.
11. **Manifest**: generate `governance/manifest.json` with hashes and versions.

---

## 5) Makefile targets

```make
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

download:
	python ops/download.py --out data/raw

train:
	python train/run.py --dbnet train/configs/dbnet.yaml --rec train/configs/rec_kh.yaml

infer:
	python infer/engine.py --images samples/ --out out/

serve:
	uvicorn service.app:app --host 0.0.0.0 --port 8000

eval:
	python eval/harness.py --test data/test --report eval/report.json

bench:
	python eval/bench.py --corpus eval/samples --batch 8
```

---

## 6) Code seeds

### 6.1 `infer/rescoring.py` (skeleton)

```python
from typing import List

def rescore(ctc_logprob: float, tokens: List[str], kenlm, lam: float, mu: float, lexicon) -> float:
    lm_lp = kenlm.score(" ".join(tokens))
    lex_pen = 0.0 if all(t in lexicon for t in tokens) else -1.0
    return ctc_logprob + lam * lm_lp + mu * lex_pen
```

### 6.2 `service/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Tuple

class Line(BaseModel):
    bbox: Tuple[int, int, int, int]
    text: str
    conf: float

class OCRResponse(BaseModel):
    text: str
    lines: List[Line]
    version: dict
    metrics: dict
    pii_report: dict | None = None
```

### 6.3 `service/app.py` (excerpt)

```python
from fastapi import FastAPI, UploadFile, File
from .schemas import OCRResponse

app = FastAPI()

@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    # TODO: call engine, build response
    return OCRResponse(text="", lines=[], version={"rec": "v1.0.0"}, metrics={"lat_ms": 0})
```

---

## 7) Evaluation & acceptance

- Report **CER** by bucket: clean vs degraded; dpi; contrast; font size.
- **Ablations**: CTC only → +KenLM → +KenLM+lexicon; each must include CER deltas.
- **Acceptance gates** (block release): CER clean ≤3%, degraded ≤10%; p95 latency ≤200 ms/page (GPU); reproducible manifest; API contract validated.

---

## 8) Observability plan

- Prom metrics: `requests_total`, `latency_ms_bucket`, `errors_total`, `pages_processed`.
- Add `X-Request-ID`; sample 1% payloads to S3 for QA with encryption at rest.
- Log model versions and hashes per request.

---

## 9) Governance & safety

- License ledger CSV for all datasets; quarantine unknown sources.
- PII/DLP report-only in v1; upgrade to blocking in v1.1.
- Delete-cascade hook to purge cached pages/outputs by `job_id`.

---

## 10) Milestone review checklist

-

---

## 11) Next steps (v1.1 backlog)

- Complex layouts (tables/columns), table-structure extraction.
- PII/DLP blocking; DPIA note.
- TRT FP16/INT8 build; autotune batcher.
- Active learning loop for hard samples; weekly font expansion.

---

## 12) References for Claude Code

- PaddleOCR docs (DBNet/Rec CTC)
- KenLM python bindings
- `khopilot/km-tokenizer-khmer` (SentencePiece) for LM/tokenization only (not CTC)
- FastAPI + pydantic v2

