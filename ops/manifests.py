#!/usr/bin/env python3
"""Manifest generation and validation for Khmer OCR with provenance tracking"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_git_info(project_root: Path) -> Dict[str, str]:
    """
    Get git repository information
    
    Args:
        project_root: Root directory of project
        
    Returns:
        Dictionary with git info
    """
    git_info = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
        "remote": "unknown"
    }
    
    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()[:8]  # Short hash
        
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            git_info["dirty"] = bool(result.stdout.strip())
        
        # Get remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            git_info["remote"] = result.stdout.strip()
    
    except Exception as e:
        print(f"Warning: Could not get git info: {e}")
    
    return git_info


def get_system_info() -> Dict[str, Any]:
    """
    Get system and environment information
    
    Returns:
        Dictionary with system info
    """
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def get_dataset_checksums(data_dir: Path) -> Dict[str, str]:
    """
    Calculate checksums for dataset files
    
    Args:
        data_dir: Data directory
        
    Returns:
        Dictionary of file checksums
    """
    checksums = {}
    
    if not data_dir.exists():
        return checksums
    
    # Check for label files
    for label_file in data_dir.glob("**/label.txt"):
        rel_path = label_file.relative_to(data_dir)
        checksums[str(rel_path)] = calculate_file_hash(label_file)[:16]  # Short hash
    
    # Check for image directories (just count, not hash all images)
    for subset in ["train", "val", "test"]:
        subset_dir = data_dir / subset
        if subset_dir.exists():
            image_count = len(list(subset_dir.glob("*.jpg")) + list(subset_dir.glob("*.png")))
            if image_count > 0:
                checksums[f"{subset}_images"] = f"{image_count} files"
    
    return checksums


def track_model_lineage(
    model_dir: Path,
    training_logs: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Track model training lineage
    
    Args:
        model_dir: Model directory
        training_logs: Path to training logs
        
    Returns:
        Dictionary with lineage info
    """
    lineage = {
        "training_date": "unknown",
        "base_model": "unknown",
        "dataset_version": "unknown",
        "hyperparameters": {},
        "metrics": {}
    }
    
    # Check for training config
    config_files = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.json"))
    if config_files:
        # Use modification time as proxy for training date
        newest_config = max(config_files, key=lambda p: p.stat().st_mtime)
        lineage["training_date"] = datetime.fromtimestamp(
            newest_config.stat().st_mtime
        ).isoformat()
        
        # Try to load config
        try:
            with open(newest_config, "r") as f:
                if newest_config.suffix == ".json":
                    config = json.load(f)
                else:
                    import yaml
                    config = yaml.safe_load(f)
                
                lineage["hyperparameters"] = config.get("hyperparameters", {})
                lineage["base_model"] = config.get("base_model", "unknown")
        except:
            pass
    
    # Check for training metrics
    metrics_file = model_dir / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                lineage["metrics"] = json.load(f)
        except:
            pass
    
    return lineage


def track_training_runs(project_root: Path) -> List[Dict[str, Any]]:
    """Track training run history and metadata"""
    runs = []
    
    # Check for training logs
    log_dirs = [
        project_root / "models",
        project_root / "train" / "logs",
        project_root / "logs",
    ]
    
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        
        # Look for training artifacts
        for run_dir in log_dir.glob("*"):
            if not run_dir.is_dir():
                continue
            
            run_info = {
                "run_id": run_dir.name,
                "path": str(run_dir.relative_to(project_root)),
                "start_time": "unknown",
                "end_time": "unknown",
                "status": "unknown",
                "model_type": "unknown",
                "metrics": {},
                "artifacts": [],
            }
            
            # Determine run type and collect artifacts
            artifacts = list(run_dir.glob("*"))
            run_info["artifacts"] = [f.name for f in artifacts if f.is_file()]
            
            # Check for specific model types
            if any(f.name.endswith(".pdparams") for f in artifacts):
                run_info["model_type"] = "paddle"
            elif any(f.name.endswith(".onnx") for f in artifacts):
                run_info["model_type"] = "onnx"
            elif any(f.name.endswith(".arpa") for f in artifacts):
                run_info["model_type"] = "kenlm"
            elif any(f.name.endswith(".model") for f in artifacts):
                run_info["model_type"] = "sentencepiece"
            
            # Get timestamps from artifacts
            if artifacts:
                timestamps = [f.stat().st_mtime for f in artifacts if f.is_file()]
                if timestamps:
                    run_info["start_time"] = datetime.fromtimestamp(min(timestamps)).isoformat()
                    run_info["end_time"] = datetime.fromtimestamp(max(timestamps)).isoformat()
            
            # Check for metrics files
            metrics_files = list(run_dir.glob("*metrics*")) + list(run_dir.glob("*results*"))
            for metrics_file in metrics_files:
                if metrics_file.suffix in [".json", ".yaml", ".yml"]:
                    try:
                        with open(metrics_file, "r") as f:
                            if metrics_file.suffix == ".json":
                                run_info["metrics"].update(json.load(f))
                            else:
                                import yaml
                                run_info["metrics"].update(yaml.safe_load(f))
                    except Exception as e:
                        print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            
            # Determine status
            if run_info["artifacts"]:
                if any("error" in name.lower() for name in run_info["artifacts"]):
                    run_info["status"] = "failed"
                elif any(name.endswith((".pdparams", ".onnx", ".arpa", ".model")) for name in run_info["artifacts"]):
                    run_info["status"] = "completed"
                else:
                    run_info["status"] = "incomplete"
            
            # Only include runs with meaningful artifacts
            if run_info["artifacts"] or run_info["metrics"]:
                runs.append(run_info)
    
    # Sort by start time (newest first)
    runs.sort(key=lambda r: r.get("start_time", "0000"), reverse=True)
    
    return runs[:10]  # Keep only last 10 runs


def generate_manifest(project_root: Path) -> Dict[str, Any]:
    """
    Generate manifest for the project
    
    Args:
        project_root: Root directory of project
        
    Returns:
        Manifest dictionary
    """
    manifest = {
        "version": "0.1.0",
        "name": "kh-ocr",
        "description": "Khmer OCR system with PaddleOCR and KenLM",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "provenance": {
            "git": get_git_info(project_root),
            "system": get_system_info(),
            "created_by": "ops/manifests.py",
        },
        "models": {},
        "configs": {},
        "datasets": {},
        "tokenizers": {},
        "training_runs": [],
        "requirements": {
            "python": ">=3.11",
            "cuda": "12.x",
            "paddlepaddle": "2.6.1",
            "paddleocr": "2.7.0",
            "kenlm": "optional",
            "sentencepiece": "0.1.99",
            "transformers": "optional (>=4.30.0)",
            "onnxruntime": "optional (>=1.12.0)",
            "pytesseract": "optional (>=0.3.10)",
        },
        "performance": {
            "cer_clean_target": 0.03,
            "cer_degraded_target": 0.10,
            "latency_p95_target_ms": 200,
            "platform_specific": {
                "cpu": {"latency_p95_target_ms": 200},
                "gpu": {"latency_p95_target_ms": 50},
            },
        },
    }
    
    # Hash model files if they exist
    models_dir = project_root / "models"
    if models_dir.exists():
        # Track model lineage
        lineage = track_model_lineage(models_dir)
        
        for model_file in models_dir.glob("*.pdparams"):
            manifest["models"][model_file.name] = {
                "path": str(model_file.relative_to(project_root)),
                "sha256": calculate_file_hash(model_file),
                "size_bytes": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "lineage": lineage,
            }
        
        for model_file in models_dir.glob("*.pdmodel"):
            manifest["models"][model_file.name] = {
                "path": str(model_file.relative_to(project_root)),
                "sha256": calculate_file_hash(model_file),
                "size_bytes": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            }
        
        for model_file in models_dir.glob("*.onnx"):
            manifest["models"][model_file.name] = {
                "path": str(model_file.relative_to(project_root)),
                "sha256": calculate_file_hash(model_file),
                "size_bytes": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "format": "onnx",
            }
    
    # Hash config files
    train_dir = project_root / "train"
    if train_dir.exists():
        for config_file in train_dir.glob("**/*.yaml"):
            manifest["configs"][config_file.name] = {
                "path": str(config_file.relative_to(project_root)),
                "sha256": calculate_file_hash(config_file),
            }
        
        charset_file = train_dir / "charset_kh.txt"
        if charset_file.exists():
            manifest["configs"]["charset_kh.txt"] = {
                "path": str(charset_file.relative_to(project_root)),
                "sha256": calculate_file_hash(charset_file),
                "num_chars": len(open(charset_file, "r", encoding="utf-8").readlines()),
            }
    
    # Language model files
    lang_dir = project_root / "lang"
    if lang_dir.exists():
        for lm_file in lang_dir.glob("**/*.arpa"):
            manifest["models"][lm_file.name] = {
                "path": str(lm_file.relative_to(project_root)),
                "sha256": calculate_file_hash(lm_file),
                "size_bytes": lm_file.stat().st_size,
                "type": "kenlm",
                "modified": datetime.fromtimestamp(lm_file.stat().st_mtime).isoformat(),
            }
        
        # Check for SentencePiece models
        for sp_file in lang_dir.glob("**/*.model"):
            manifest["models"][sp_file.name] = {
                "path": str(sp_file.relative_to(project_root)),
                "sha256": calculate_file_hash(sp_file),
                "size_bytes": sp_file.stat().st_size,
                "type": "sentencepiece",
                "modified": datetime.fromtimestamp(sp_file.stat().st_mtime).isoformat(),
            }
        
        # Check for HuggingFace tokenizers
        for tokenizer_dir in lang_dir.glob("**/tokenizer/*/"):
            if tokenizer_dir.is_dir():
                config_file = tokenizer_dir / "tokenizer_config.json"
                if config_file.exists():
                    try:
                        with open(config_file, "r") as f:
                            config = json.load(f)
                        
                        manifest["tokenizers"][tokenizer_dir.name] = {
                            "path": str(tokenizer_dir.relative_to(project_root)),
                            "type": "huggingface",
                            "model_type": config.get("tokenizer_class", "unknown"),
                            "vocab_size": config.get("vocab_size", 0),
                            "model_max_length": config.get("model_max_length", 0),
                            "files": [f.name for f in tokenizer_dir.glob("*") if f.is_file()],
                            "modified": datetime.fromtimestamp(config_file.stat().st_mtime).isoformat(),
                        }
                    except Exception as e:
                        print(f"Warning: Could not read tokenizer config {config_file}: {e}")
    
    # Add dataset checksums
    data_dir = project_root / "data"
    if data_dir.exists():
        manifest["datasets"] = get_dataset_checksums(data_dir)
    
    # Add evaluation results if available
    eval_report = project_root / "eval" / "report.json"
    if eval_report.exists():
        with open(eval_report, "r") as f:
            report = json.load(f)
            manifest["evaluation"] = {
                "cer_clean": report.get("cer_clean"),
                "cer_degraded": report.get("cer_degraded"),
                "timestamp": report.get("timestamp"),
                "acceptance_criteria": report.get("acceptance_criteria", {}),
            }
    
    # Add dependency versions
    try:
        import paddle
        manifest["provenance"]["paddle_version"] = paddle.__version__
    except:
        pass
    
    try:
        import paddleocr
        manifest["provenance"]["paddleocr_version"] = paddleocr.__version__
    except:
        pass
    
    try:
        from transformers import __version__ as transformers_version
        manifest["provenance"]["transformers_version"] = transformers_version
    except:
        pass
    
    try:
        import onnxruntime
        manifest["provenance"]["onnxruntime_version"] = onnxruntime.__version__
    except:
        pass
    
    # Track training runs from logs
    manifest["training_runs"] = track_training_runs(project_root)
    
    return manifest


def validate_manifest(manifest_path: Path) -> bool:
    """
    Validate manifest file
    
    Args:
        manifest_path: Path to manifest.json
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Check required fields
        required_fields = ["version", "name", "timestamp", "models", "configs"]
        for field in required_fields:
            if field not in manifest:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate training runs if present
        if "training_runs" in manifest:
            for i, run in enumerate(manifest["training_runs"]):
                if "run_id" not in run:
                    print(f"⚠️  Training run {i} missing run_id")
                if "status" not in run:
                    print(f"⚠️  Training run {i} missing status")
                elif run["status"] == "failed":
                    print(f"⚠️  Training run {run.get('run_id', i)} failed")
        
        # Check tokenizer configurations
        if "tokenizers" in manifest:
            for tok_name, tok_info in manifest["tokenizers"].items():
                if tok_info.get("type") == "huggingface" and tok_info.get("vocab_size", 0) == 0:
                    print(f"⚠️  Tokenizer {tok_name} has zero vocab size")
        
        # Validate version format
        version = manifest["version"]
        if not version or not version[0].isdigit():
            print(f"❌ Invalid version format: {version}")
            return False
        
        # Check model hashes
        project_root = manifest_path.parent.parent
        for model_name, model_info in manifest.get("models", {}).items():
            if "sha256" not in model_info:
                print(f"⚠️ Missing SHA256 for model: {model_name}")
            
            # Verify file exists if path is provided
            if "path" in model_info:
                model_path = project_root / model_info["path"]
                if not model_path.exists():
                    print(f"⚠️ Model file not found: {model_info['path']}")
                else:
                    # Verify hash matches
                    actual_hash = calculate_file_hash(model_path)
                    if actual_hash != model_info.get("sha256"):
                        print(f"❌ Hash mismatch for {model_name}")
                        print(f"   Expected: {model_info.get('sha256')}")
                        print(f"   Actual: {actual_hash}")
                        return False
        
        # Check performance targets
        if "performance" in manifest:
            perf = manifest["performance"]
            if perf.get("cer_clean_target", 1.0) > 0.03:
                print(f"⚠️ CER clean target above 3%: {perf['cer_clean_target']}")
            if perf.get("cer_degraded_target", 1.0) > 0.10:
                print(f"⚠️ CER degraded target above 10%: {perf['cer_degraded_target']}")
        
        print(f"✅ Manifest validation passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in manifest: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating manifest: {e}")
        return False


def update_manifest(project_root: Path):
    """
    Update or create manifest file
    
    Args:
        project_root: Project root directory
    """
    governance_dir = project_root / "governance"
    governance_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = governance_dir / "manifest.json"
    
    # Generate new manifest
    manifest = generate_manifest(project_root)
    
    # Save to file
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Manifest updated: {manifest_path}")
    
    # Display summary
    print(f"\n📋 Manifest Summary:")
    print(f"   Version: {manifest['version']}")
    print(f"   Models: {len(manifest.get('models', {}))}")
    print(f"   Configs: {len(manifest.get('configs', {}))}")
    print(f"   Tokenizers: {len(manifest.get('tokenizers', {}))}")
    print(f"   Training runs: {len(manifest.get('training_runs', []))}")
    
    # Show recent training runs
    if manifest.get("training_runs"):
        print(f"\n🏃 Recent Training Runs:")
        for run in manifest["training_runs"][:3]:  # Show last 3
            status_emoji = "✅" if run.get("status") == "completed" else "⚠️ " if run.get("status") == "failed" else "🔄"
            print(f"   {status_emoji} {run.get('run_id', 'unknown')} ({run.get('model_type', 'unknown')}) - {run.get('status', 'unknown')}")
    
    # Show tokenizer info
    if manifest.get("tokenizers"):
        print(f"\n🔤 Tokenizers:")
        for tok_name, tok_info in manifest["tokenizers"].items():
            vocab_size = tok_info.get("vocab_size", 0)
            tok_type = tok_info.get("type", "unknown")
            print(f"   📝 {tok_name} ({tok_type}) - {vocab_size} vocab")
    
    if "evaluation" in manifest:
        eval_info = manifest["evaluation"]
        if eval_info.get("cer_clean") is not None:
            print(f"   CER (clean): {eval_info['cer_clean']:.2f}%")
        if eval_info.get("cer_degraded") is not None:
            print(f"   CER (degraded): {eval_info['cer_degraded']:.2f}%")


def main():
    """CLI interface for manifest management"""
    parser = argparse.ArgumentParser(description="Manage project manifest")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update or create manifest"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing manifest"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="governance/manifest.json",
        help="Path to manifest file"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.path:
        manifest_path = Path(args.path)
        if manifest_path.is_absolute():
            project_root = manifest_path.parent.parent
        else:
            project_root = Path.cwd()
            manifest_path = project_root / args.path
    else:
        project_root = Path.cwd()
        manifest_path = project_root / "governance" / "manifest.json"
    
    if args.update:
        update_manifest(project_root)
    elif args.validate:
        if not manifest_path.exists():
            print(f"❌ Manifest not found: {manifest_path}")
            print("   Run with --update to create it")
            sys.exit(1)
        
        if not validate_manifest(manifest_path):
            sys.exit(1)
    else:
        # Default to update
        update_manifest(project_root)


if __name__ == "__main__":
    main()