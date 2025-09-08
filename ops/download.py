#!/usr/bin/env python3
"""Download OCR datasets from Hugging Face into data/raw/.

Supports two strategies:
  1) huggingface_hub.snapshot_download (preferred)
  2) git clone fallback (if hub API unavailable)

Also performs a light post-download pass to locate PaddleOCR-style
label files and place a convenience copy at data/raw/<name>/label.txt
or det_label.txt when found.

Note: This script does not run in this sandboxed environment due to
network restrictions. Use it locally or in CI with network enabled.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class HFSpec:
    repo_id: str  # e.g., khopilot/khmer-ocr-recognition
    revision: Optional[str] = None  # commit/tag/branch
    subdir: Optional[str] = None  # optional subdirectory within repo


def have_hf_hub() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


def snapshot_download(spec: HFSpec, target_dir: Path) -> Path:
    """Download a repo snapshot via huggingface_hub."""
    from huggingface_hub import snapshot_download

    local_dir = target_dir / spec.repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)

    repo_type = "dataset"  # we expect datasets
    revision = spec.revision or "main"
    print(f"Downloading {spec.repo_id}@{revision} → {local_dir}")

    snapshot_path = snapshot_download(
        repo_id=spec.repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    return Path(snapshot_path)


def git_clone(spec: HFSpec, target_dir: Path) -> Path:
    """Fallback: clone dataset repo via git."""
    url = f"https://huggingface.co/datasets/{spec.repo_id}"
    local_dir = target_dir / spec.repo_id.replace("/", "__")
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"Reusing existing repo clone at {local_dir}")
        return local_dir

    print(f"Cloning {url} → {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    rev = spec.revision or "main"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", rev, url, str(local_dir)], check=True)
    return local_dir


def find_and_copy_labels(root: Path, out_dir: Path) -> None:
    """Find PaddleOCR label files and copy a convenience copy to out_dir.

    - recognition: label.txt
    - detection: det_label.txt (customary naming)
    """
    rec_found = None
    det_found = None
    for p in root.rglob("label.txt"):
        rec_found = p
        break
    for p in root.rglob("det_label.txt"):
        det_found = p
        break

    out_dir.mkdir(parents=True, exist_ok=True)
    if rec_found:
        dest = out_dir / "label.txt"
        if dest.resolve() != rec_found.resolve():
            shutil.copyfile(rec_found, dest)
        print(f"✓ recognition labels: {dest}")
    if det_found:
        dest = out_dir / "det_label.txt"
        if dest.resolve() != det_found.resolve():
            shutil.copyfile(det_found, dest)
        print(f"✓ detection labels: {dest}")


def parse_specs_from_config(config_path: Optional[Path]) -> List[HFSpec]:
    if not config_path:
        return []
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    specs: List[HFSpec] = []
    for item in cfg.get("huggingface_datasets", []):
        specs.append(HFSpec(
            repo_id=item.get("repo_id"),
            revision=item.get("revision"),
            subdir=item.get("subdir"),
        ))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OCR datasets from Hugging Face")
    parser.add_argument("--repo-id", action="append", help="HF dataset repo id (can repeat)")
    parser.add_argument("--revision", help="Revision/tag/commit", default=None)
    parser.add_argument("--subdir", help="Subdirectory inside repo", default=None)
    parser.add_argument("--config", type=Path, help="YAML config listing datasets", default=Path("data/datasets.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--strategy", choices=["hub", "git", "auto"], default="auto")
    args = parser.parse_args()

    target_root = args.out
    target_root.mkdir(parents=True, exist_ok=True)

    # Collect specs
    specs: List[HFSpec] = []
    specs.extend(parse_specs_from_config(args.config) or [])
    if args.repo_id:
        for rid in args.repo_id:
            specs.append(HFSpec(repo_id=rid, revision=args.revision, subdir=args.subdir))

    if not specs:
        print("No datasets specified. Provide --repo-id or a data/datasets.yaml.")
        sys.exit(0)

    use_hub = (args.strategy == "hub") or (args.strategy == "auto" and have_hf_hub())

    for spec in specs:
        try:
            repo_dir = snapshot_download(spec, target_root) if use_hub else git_clone(spec, target_root)
            # If subdir specified, use it as root for labels
            root_for_labels = repo_dir / spec.subdir if spec.subdir else repo_dir
            dataset_out = target_root / spec.repo_id.replace("/", "__")
            find_and_copy_labels(root_for_labels, dataset_out)
        except Exception as e:
            print(f"✗ Failed to fetch {spec.repo_id}: {e}")
            continue

    print("Done.")


if __name__ == "__main__":
    main()

