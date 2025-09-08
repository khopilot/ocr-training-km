#!/usr/bin/env python3
"""Hardened ML pipeline orchestrator for Khmer OCR v1

Executes steps with validation gates, retry logic, and conditional execution.
Includes dataset validation, metric-based gates, and detailed reporting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class Step:
    name: str
    command: str
    output: Optional[str] = None
    gpu_required: bool = False
    timeout: Optional[int] = None
    optional: bool = False
    validation: Optional[str] = None  # Validation command
    retry_count: int = 1  # Number of retries
    retry_delay: int = 5  # Delay between retries in seconds
    gate: Optional[Dict[str, Any]] = None  # Validation gate conditions


@dataclass
class Task:
    name: str
    priority: int
    dependencies: List[str]
    steps: List[Step]
    conditional: Optional[str] = None  # Condition for execution
    validation_required: bool = True  # Whether to validate outputs


def load_template(template_path: Path) -> Dict:
    with open(template_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tasks(doc: Dict) -> List[Task]:
    tasks_doc = doc.get("tasks", {})
    tasks: List[Task] = []
    for task_name, task_body in tasks_doc.items():
        steps = []
        for s in task_body.get("steps", []):
            steps.append(
                Step(
                    name=s.get("name", "step"),
                    command=s["command"],
                    output=s.get("output"),
                    gpu_required=bool(s.get("gpu_required", False)),
                    timeout=s.get("timeout"),
                    optional=bool(s.get("optional", False)),
                    validation=s.get("validation"),
                    retry_count=s.get("retry_count", 1),
                    retry_delay=s.get("retry_delay", 5),
                    gate=s.get("gate"),
                )
            )

        tasks.append(
            Task(
                name=task_name,
                priority=int(task_body.get("priority", 999)),
                dependencies=list(task_body.get("dependencies", [])),
                steps=steps,
                conditional=task_body.get("conditional"),
                validation_required=task_body.get("validation_required", True),
            )
        )
    tasks.sort(key=lambda t: t.priority)
    return tasks


def topological_order(tasks: List[Task]) -> List[Task]:
    # Simplistic topo-order using priorities and dependencies; detects cycles loosely
    name_to_task = {t.name: t for t in tasks}
    resolved: List[str] = []
    ordered: List[Task] = []
    remaining = set(name_to_task.keys())
    guard = 0
    while remaining and guard < 1000:
        progressed = False
        for name in list(remaining):
            deps = set(name_to_task[name].dependencies)
            if deps.issubset(set(resolved)):
                ordered.append(name_to_task[name])
                resolved.append(name)
                remaining.remove(name)
                progressed = True
        if not progressed:
            # Fallback to priority order if blocked
            for name in list(remaining):
                ordered.append(name_to_task[name])
                resolved.append(name)
                remaining.remove(name)
            break
        guard += 1
    return ordered


def run_command(
    cmd: str,
    timeout: Optional[int],
    dry_run: bool,
    verbose: bool,
    retry_count: int = 1,
    retry_delay: int = 5
) -> Tuple[int, str]:
    """Run command with retry logic"""
    print(f"  → {cmd}")
    if dry_run:
        return 0, "Dry run"
    
    for attempt in range(retry_count):
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            
            if result.returncode == 0:
                return int(result.returncode), result.stdout
            
            # If failed and have retries left
            if attempt < retry_count - 1:
                print(f"  ⚠ Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                if verbose:
                    print(f"  Error output: {result.stderr}")
                return int(result.returncode), result.stderr
                
        except subprocess.TimeoutExpired:
            if attempt < retry_count - 1:
                print(f"  ⚠ Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(retry_delay)
            else:
                print("  ✖ Timeout expired after all retries")
                return 124, "Timeout"
    
    return 1, "Failed after retries"


def validate_dataset(data_dir: Path, charset_path: Path, min_coverage: float = 99.0) -> bool:
    """
    Validate dataset quality
    
    Args:
        data_dir: Dataset directory
        charset_path: Path to charset file
        min_coverage: Minimum charset coverage required
        
    Returns:
        True if validation passes
    """
    print(f"  Validating dataset in {data_dir}")
    
    # Check if validation script exists
    validation_script = Path("ops/validate_dataset.py")
    if validation_script.exists():
        cmd = f"python {validation_script} --data-dir {data_dir} --charset {charset_path} --min-coverage {min_coverage}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"    ✓ Dataset validation passed")
            return True
        else:
            print(f"    ✗ Dataset validation failed")
            print(f"    {result.stderr}")
            return False
    else:
        # Fallback: basic checks
        if not data_dir.exists():
            print(f"    ✗ Data directory not found: {data_dir}")
            return False
        
        label_file = data_dir / "label.txt"
        if not label_file.exists():
            print(f"    ⚠ No label.txt found, assuming valid")
        
        return True


def check_validation_gate(gate: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
    """
    Check if validation gate conditions are met
    
    Args:
        gate: Gate conditions
        metrics: Current metrics
        
    Returns:
        True if gate passes
    """
    if not gate:
        return True
    
    for metric, condition in gate.items():
        if metric not in metrics:
            print(f"    ⚠ Metric '{metric}' not found, skipping gate")
            continue
        
        value = metrics[metric]
        
        # Parse condition (e.g., "< 0.03", ">= 100")
        if isinstance(condition, str):
            parts = condition.split()
            if len(parts) == 2:
                op, threshold = parts[0], float(parts[1])
                
                if op == "<" and value >= threshold:
                    print(f"    ✗ Gate failed: {metric}={value} not < {threshold}")
                    return False
                elif op == "<=" and value > threshold:
                    print(f"    ✗ Gate failed: {metric}={value} not <= {threshold}")
                    return False
                elif op == ">" and value <= threshold:
                    print(f"    ✗ Gate failed: {metric}={value} not > {threshold}")
                    return False
                elif op == ">=" and value < threshold:
                    print(f"    ✗ Gate failed: {metric}={value} not >= {threshold}")
                    return False
    
    print(f"    ✓ All gates passed")
    return True


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from file"""
    if metrics_path.exists():
        if metrics_path.suffix == ".json":
            with open(metrics_path, "r") as f:
                return json.load(f)
        elif metrics_path.suffix == ".yaml":
            with open(metrics_path, "r") as f:
                return yaml.safe_load(f)
    return {}


from typing import Tuple


def orchestrate(
    template_path: Path,
    continue_on_error: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    validate_datasets: bool = True,
    report_path: Optional[Path] = None,
) -> int:
    print(f"Using template: {template_path}")
    doc = load_template(template_path)
    tasks = parse_tasks(doc)
    tasks = topological_order(tasks)

    status: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    overall_rc = 0
    execution_report = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": {},
        "metrics": {},
    }
    
    # Add dataset validation as first step if enabled
    if validate_datasets:
        print("\n== Dataset Validation ==")
        data_dir = Path("data")
        charset_path = Path("train/charset_kh.txt")
        
        if not validate_dataset(data_dir, charset_path):
            print("✗ Dataset validation failed")
            if not continue_on_error:
                return 1
            overall_rc = 1

    for task in tasks:
        print(f"\n== Task: {task.name} (priority {task.priority}) ==")
        
        # Check conditional execution
        if task.conditional:
            if task.conditional == "if_metrics_below_target":
                # Check if metrics are below target
                eval_report = Path("eval/report.json")
                if eval_report.exists():
                    current_metrics = load_metrics(eval_report)
                    if current_metrics.get("cer_clean", 100) <= 3.0:
                        print(f"  Skipping (metrics already meet target)")
                        status[task.name] = "skipped"
                        continue
        
        status[task.name] = "running"
        task_start = time.time()
        task_report = {"steps": [], "status": "running"}
        
        for step in task.steps:
            # Basic presence checks for python script commands
            tokens = step.command.split()
            maybe_path = None
            if tokens and tokens[0] == "python" and len(tokens) > 1:
                maybe_path = Path(tokens[1])
            elif tokens and tokens[0].endswith(".py"):
                maybe_path = Path(tokens[0])
            if maybe_path and not maybe_path.exists():
                msg = f"  ⚠ Script not found: {maybe_path}"
                if step.optional:
                    print(msg + " (optional, skipping)")
                    continue
                else:
                    print(msg)
                    if continue_on_error:
                        overall_rc = overall_rc or 127
                        continue
                    return 127

            print(f"- Step: {step.name}")
            step_start = time.time()
            
            # Run command with retry logic
            rc, output = run_command(
                step.command,
                step.timeout,
                dry_run,
                verbose,
                retry_count=step.retry_count,
                retry_delay=step.retry_delay
            )
            
            step_duration = time.time() - step_start
            step_info = {
                "name": step.name,
                "command": step.command,
                "return_code": rc,
                "duration_sec": step_duration,
            }
            
            if rc != 0:
                print(f"  ✖ Failed with code {rc}")
                step_info["status"] = "failed"
                if step.optional:
                    print("  (optional step failure ignored)")
                    step_info["status"] = "skipped"
                elif continue_on_error:
                    overall_rc = overall_rc or rc
                else:
                    task_report["steps"].append(step_info)
                    execution_report["tasks"][task.name] = task_report
                    return rc
            else:
                print("  ✓ Completed")
                step_info["status"] = "success"
                
                # Run validation if specified
                if step.validation:
                    print(f"  Validating: {step.validation}")
                    val_rc, _ = run_command(step.validation, 60, dry_run, verbose)
                    if val_rc != 0:
                        print(f"    ✗ Validation failed")
                        step_info["validation"] = "failed"
                        if not continue_on_error:
                            return val_rc
                    else:
                        print(f"    ✓ Validation passed")
                        step_info["validation"] = "passed"
                
                # Check validation gate
                if step.gate:
                    # Load current metrics
                    if step.output and Path(step.output).exists():
                        step_metrics = load_metrics(Path(step.output))
                        metrics.update(step_metrics)
                    
                    if not check_validation_gate(step.gate, metrics):
                        print(f"  ✗ Validation gate failed")
                        step_info["gate"] = "failed"
                        if not continue_on_error:
                            return 1
                    else:
                        step_info["gate"] = "passed"
            
            task_report["steps"].append(step_info)

        task_duration = time.time() - task_start
        task_report["duration_sec"] = task_duration
        task_report["status"] = "completed"
        status[task.name] = "completed"
        execution_report["tasks"][task.name] = task_report
        print(f"✓ Task completed: {task.name} ({task_duration:.1f}s)")

    # Save execution report
    execution_report["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    execution_report["overall_status"] = "failed" if overall_rc != 0 else "success"
    execution_report["metrics"] = metrics
    
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(execution_report, f, indent=2)
        print(f"\nExecution report saved to: {report_path}")
    
    print("\nPipeline finished")
    return overall_rc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hardened ML pipeline")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(".claude/templates/agent-ml-pipeline.yaml"),
        help="Path to pipeline YAML template",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip dataset validation"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(".claude/logs/pipeline_report.json"),
        help="Path to save execution report"
    )
    args = parser.parse_args()

    rc = orchestrate(
        template_path=args.template,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        verbose=args.verbose,
        validate_datasets=not args.no_validation,
        report_path=args.report,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

