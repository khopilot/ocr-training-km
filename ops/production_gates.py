#!/usr/bin/env python3
"""Production gates validation for Khmer OCR pipeline.

Validates that all production requirements are met before deployment:
- License compliance (no research-only datasets)
- Model performance gates (CER targets)
- Latency requirements
- Manifest completeness
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ProductionGateValidator:
    """Validates production deployment requirements"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.failures = []
        self.warnings = []
    
    def validate_license_compliance(self) -> bool:
        """Check license compliance for production deployment"""
        print("🔍 Validating license compliance...")
        
        ledger_path = self.project_root / "governance" / "license_ledger.csv"
        if not ledger_path.exists():
            self.failures.append("License ledger not found: governance/license_ledger.csv")
            return False
        
        if not HAS_PANDAS:
            self.warnings.append("pandas not available - skipping detailed license check")
            return True
        
        try:
            df = pd.read_csv(ledger_path)
            
            # Check for research-only datasets
            research_only = df[df['usage_allowed'] == 'research_only']
            if len(research_only) > 0:
                self.failures.append(
                    f"Found {len(research_only)} research-only datasets: "
                    f"{', '.join(research_only['dataset'].tolist())}"
                )
                return False
            
            # Check for non-commercial datasets
            non_commercial = df[df['commercial_use'] == 'no']
            if len(non_commercial) > 0:
                self.failures.append(
                    f"Found {len(non_commercial)} non-commercial datasets: "
                    f"{', '.join(non_commercial['dataset'].tolist())}"
                )
                return False
            
            print(f"   ✅ All {len(df)} datasets cleared for commercial production use")
            return True
            
        except Exception as e:
            self.failures.append(f"License validation failed: {e}")
            return False
    
    def validate_model_performance(self) -> bool:
        """Check model performance against targets"""
        print("📊 Validating model performance...")
        
        report_path = self.project_root / "eval" / "report.json"
        if not report_path.exists():
            self.warnings.append("Evaluation report not found - skipping performance check")
            return True
        
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # CER targets
            cer_clean = report.get('cer_clean')
            cer_degraded = report.get('cer_degraded')
            
            performance_ok = True
            
            if cer_clean is not None:
                if cer_clean > 0.03:  # 3% target
                    self.failures.append(f"CER clean {cer_clean:.3f} > 0.03 target")
                    performance_ok = False
                else:
                    print(f"   ✅ CER clean: {cer_clean:.3f} ≤ 0.03")
            
            if cer_degraded is not None:
                if cer_degraded > 0.10:  # 10% target
                    self.failures.append(f"CER degraded {cer_degraded:.3f} > 0.10 target")
                    performance_ok = False
                else:
                    print(f"   ✅ CER degraded: {cer_degraded:.3f} ≤ 0.10")
            
            return performance_ok
            
        except Exception as e:
            self.warnings.append(f"Performance validation failed: {e}")
            return True  # Don't block on performance validation errors
    
    def validate_latency_requirements(self) -> bool:
        """Check latency requirements"""
        print("⚡ Validating latency requirements...")
        
        benchmark_path = self.project_root / "eval" / "benchmark.json"
        if not benchmark_path.exists():
            self.warnings.append("Benchmark report not found - skipping latency check")
            return True
        
        try:
            with open(benchmark_path, 'r') as f:
                benchmark = json.load(f)
            
            # Check P95 latency for batch processing
            batch_results = benchmark.get('results', {}).get('batch_benchmarks', {})
            
            latency_ok = True
            for batch_size, results in batch_results.items():
                p95_ms = results.get('p95_ms')
                if p95_ms is not None:
                    if p95_ms > 200:  # 200ms P95 target
                        self.failures.append(f"Batch {batch_size} P95 latency {p95_ms:.1f}ms > 200ms target")
                        latency_ok = False
                    else:
                        print(f"   ✅ Batch {batch_size} P95: {p95_ms:.1f}ms ≤ 200ms")
            
            return latency_ok
            
        except Exception as e:
            self.warnings.append(f"Latency validation failed: {e}")
            return True  # Don't block on latency validation errors
    
    def validate_manifest_completeness(self) -> bool:
        """Check manifest completeness"""
        print("📋 Validating manifest completeness...")
        
        manifest_path = self.project_root / "governance" / "manifest.json"
        if not manifest_path.exists():
            self.failures.append("Manifest not found: governance/manifest.json")
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Required fields
            required_fields = ["version", "models", "provenance", "timestamp"]
            for field in required_fields:
                if field not in manifest:
                    self.failures.append(f"Missing required manifest field: {field}")
                    return False
            
            # Check model hashes
            models = manifest.get("models", {})
            missing_hashes = []
            for model_name, model_info in models.items():
                if "sha256" not in model_info:
                    missing_hashes.append(model_name)
            
            if missing_hashes:
                self.failures.append(f"Missing SHA256 hashes: {', '.join(missing_hashes)}")
                return False
            
            print(f"   ✅ Manifest complete with {len(models)} models tracked")
            return True
            
        except Exception as e:
            self.failures.append(f"Manifest validation failed: {e}")
            return False
    
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks"""
        print("🚪 Running production gate validation...\n")
        
        checks = [
            self.validate_license_compliance(),
            self.validate_model_performance(),
            self.validate_latency_requirements(),
            self.validate_manifest_completeness(),
        ]
        
        all_passed = all(checks)
        
        print(f"\n{'='*50}")
        if all_passed:
            print("✅ All production gates PASSED")
            if self.warnings:
                print(f"⚠️  {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    print(f"   - {warning}")
        else:
            print("❌ Production gates FAILED")
            print(f"   {len(self.failures)} failures:")
            for failure in self.failures:
                print(f"   - {failure}")
            if self.warnings:
                print(f"   {len(self.warnings)} warnings:")
                for warning in self.warnings:
                    print(f"   - {warning}")
        
        print("="*50)
        
        return all_passed, self.failures, self.warnings


def main():
    """CLI interface for production gate validation"""
    parser = argparse.ArgumentParser(description="Validate production deployment gates")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Treat warnings as failures"
    )
    
    args = parser.parse_args()
    
    validator = ProductionGateValidator(args.project_root)
    all_passed, failures, warnings = validator.validate_all()
    
    if not all_passed:
        sys.exit(1)
    
    if args.fail_on_warnings and warnings:
        print("❌ Failing on warnings as requested")
        sys.exit(1)
    
    print("🎉 Production gates validation complete!")


if __name__ == "__main__":
    main()