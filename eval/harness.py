"""Evaluation harness for Khmer OCR system with real inference"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .cer import calculate_cer, calculate_wer, evaluate_batch, get_error_analysis

# Import the real OCR engine
sys.path.append(str(Path(__file__).parent.parent))
from infer.engine import OCREngine


class EvaluationHarness:
    """Evaluation harness with real OCR engine integration"""
    
    def __init__(
        self, 
        test_dir: Path, 
        model_dir: Optional[Path] = None,
        use_gpu: bool = False,
        lm_path: Optional[Path] = None,
        ablation_mode: str = "full"
    ):
        """
        Initialize evaluation harness with real OCR engine
        
        Args:
            test_dir: Directory containing test data
            model_dir: Directory containing models
            use_gpu: Whether to use GPU for inference
            lm_path: Path to language model
            ablation_mode: Ablation mode (ctc_only, ctc_lm, full)
        """
        self.test_dir = Path(test_dir)
        self.model_dir = model_dir or Path("models")
        self.use_gpu = use_gpu
        self.ablation_mode = ablation_mode
        self.results = []
        self.latencies = []
        
        # Initialize real OCR engine
        lm_path = lm_path or Path("lang/kenlm/khmer_5gram.arpa")
        self.engine = OCREngine(
            model_dir=self.model_dir,
            use_gpu=self.use_gpu,
            lm_path=lm_path if lm_path.exists() else None,
            charset_path=Path("train/charset_kh.txt")
        )
    
    def load_test_data(self) -> List[Tuple[Path, str]]:
        """
        Load test data pairs (image_path, ground_truth_text)
        
        Returns:
            List of test samples
        """
        test_samples = []
        
        # Look for label file
        label_file = self.test_dir / "label.txt"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        image_name, text = line.strip().split("\t", 1)
                        image_path = self.test_dir / image_name
                        if image_path.exists():
                            test_samples.append((image_path, text))
        else:
            # Create placeholder test data
            print("Warning: No label.txt found, using placeholder data")
            test_samples = [
                (Path("sample1.jpg"), "សួស្តី ពិភពលោក"),
                (Path("sample2.jpg"), "ប្រទេសកម្ពុជា"),
                (Path("sample3.jpg"), "ភាសាខ្មែរ"),
            ]
        
        return test_samples
    
    def run_inference(self, image_path: Path) -> Tuple[str, float]:
        """
        Run real OCR inference using the engine
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (predicted text, latency in ms)
        """
        start_time = time.time()
        
        # Check if image exists, if not create placeholder
        if not image_path.exists():
            # For testing, create a placeholder image if needed
            image_path = self.test_dir / image_path.name
            if not image_path.exists():
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (400, 100), color='white')
                draw = ImageDraw.Draw(img)
                draw.text((10, 30), "Test", fill='black')
                image_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(image_path)
        
        # Run real inference with ablation mode
        result = self.engine.process_image(
            image_path, 
            enable_lm=(self.ablation_mode != "ctc_only"),
            ablation_mode=self.ablation_mode
        )
        
        latency_ms = (time.time() - start_time) * 1000
        self.latencies.append(latency_ms)
        
        return result["text"], latency_ms
    
    def evaluate_sample(
        self,
        image_path: Path,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single sample with real inference
        
        Args:
            image_path: Path to image
            ground_truth: Ground truth text
            
        Returns:
            Evaluation metrics
        """
        # Run real inference
        prediction, latency_ms = self.run_inference(image_path)
        
        # Calculate metrics
        cer = calculate_cer(ground_truth, prediction)
        wer = calculate_wer(ground_truth, prediction)
        
        return {
            "image": str(image_path),
            "ground_truth": ground_truth,
            "prediction": prediction,
            "cer": cer,
            "wer": wer,
            "latency_ms": latency_ms,
            "ablation_mode": self.ablation_mode,
            "gt_length": len(ground_truth),
            "pred_length": len(prediction),
        }
    
    def evaluate_dataset(
        self,
        dataset_type: str = "all"
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset
        
        Args:
            dataset_type: Type of dataset (all, clean, degraded)
            
        Returns:
            Aggregated metrics
        """
        # Load test data
        test_samples = self.load_test_data()
        
        if not test_samples:
            print("No test samples found")
            return {}
        
        print(f"Evaluating {len(test_samples)} samples...")
        
        # Evaluate each sample
        results = []
        for image_path, ground_truth in tqdm(test_samples):
            result = self.evaluate_sample(image_path, ground_truth)
            results.append(result)
            self.results.append(result)
        
        # Aggregate metrics with latency percentiles
        latencies = [r["latency_ms"] for r in results]
        aggregated = {
            "dataset_type": dataset_type,
            "num_samples": len(results),
            "ablation_mode": self.ablation_mode,
            "cer_mean": np.mean([r["cer"] for r in results]),
            "cer_std": np.std([r["cer"] for r in results]),
            "cer_min": np.min([r["cer"] for r in results]),
            "cer_max": np.max([r["cer"] for r in results]),
            "wer_mean": np.mean([r["wer"] for r in results]),
            "wer_std": np.std([r["wer"] for r in results]),
            "latency_mean_ms": np.mean(latencies),
            "latency_p50_ms": np.percentile(latencies, 50),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "samples_below_3_cer": sum(1 for r in results if r["cer"] < 3.0),
            "samples_below_10_cer": sum(1 for r in results if r["cer"] < 10.0),
        }
        
        # Calculate pass rates
        aggregated["pass_rate_3pct"] = (
            aggregated["samples_below_3_cer"] / aggregated["num_samples"] * 100
        )
        aggregated["pass_rate_10pct"] = (
            aggregated["samples_below_10_cer"] / aggregated["num_samples"] * 100
        )
        
        return aggregated
    
    def generate_report(self, output_path: Path):
        """
        Generate evaluation report
        
        Args:
            output_path: Path to save report
        """
        # Evaluate different dataset types
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": "0.1.0-placeholder",
            "test_directory": str(self.test_dir),
        }
        
        # Run evaluations
        if (self.test_dir / "clean").exists():
            self.test_dir = self.test_dir / "clean"
            report["clean"] = self.evaluate_dataset("clean")
        
        if (self.test_dir / "degraded").exists():
            self.test_dir = self.test_dir / "degraded"
            report["degraded"] = self.evaluate_dataset("degraded")
        
        # Default evaluation
        if "clean" not in report and "degraded" not in report:
            report["all"] = self.evaluate_dataset("all")
        
        # Add summary
        if "clean" in report:
            report["cer_clean"] = report["clean"]["cer_mean"]
        if "degraded" in report:
            report["cer_degraded"] = report["degraded"]["cer_mean"]
        
        # Check acceptance criteria with latency
        clean_latency_p95 = report.get("clean", {}).get("latency_p95_ms", float('inf'))
        degraded_latency_p95 = report.get("degraded", {}).get("latency_p95_ms", float('inf'))
        overall_latency_p95 = report.get("all", {}).get("latency_p95_ms", clean_latency_p95)
        
        report["acceptance_criteria"] = {
            "cer_clean_target": 3.0,
            "cer_clean_pass": report.get("cer_clean", 100) <= 3.0,
            "cer_degraded_target": 10.0,
            "cer_degraded_pass": report.get("cer_degraded", 100) <= 10.0,
            "latency_p95_target_ms": 200,
            "latency_p95_pass": overall_latency_p95 <= 200,
        }
        
        # Determine overall pass/fail
        all_pass = (
            report["acceptance_criteria"]["cer_clean_pass"] and
            report["acceptance_criteria"]["cer_degraded_pass"] and
            report["acceptance_criteria"]["latency_p95_pass"]
        )
        report["acceptance_criteria"]["all_pass"] = all_pass
        
        # Save report
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        if "clean" in report:
            print(f"Clean dataset CER: {report['clean']['cer_mean']:.2f}%")
            print(f"  Pass rate (<3%): {report['clean']['pass_rate_3pct']:.1f}%")
        
        if "degraded" in report:
            print(f"Degraded dataset CER: {report['degraded']['cer_mean']:.2f}%")
            print(f"  Pass rate (<10%): {report['degraded']['pass_rate_10pct']:.1f}%")
        
        if "all" in report:
            print(f"Overall CER: {report['all']['cer_mean']:.2f}%")
        
        print("\nAcceptance Criteria:")
        for key, value in report["acceptance_criteria"].items():
            if key.endswith("_pass"):
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"  {key}: {status}")
        
        print(f"\nReport saved to: {output_path}")


def main():
    """CLI interface for evaluation harness with ablation support"""
    parser = argparse.ArgumentParser(description="Khmer OCR evaluation harness")
    parser.add_argument(
        "--test",
        type=str,
        default="data/test",
        help="Test data directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Model directory"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="eval/report.json",
        help="Output report path"
    )
    parser.add_argument(
        "--ablation",
        choices=["ctc_only", "ctc_lm", "full"],
        default="full",
        help="Ablation mode for evaluation"
    )
    parser.add_argument(
        "--lm-path",
        type=str,
        help="Path to language model"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if acceptance criteria not met"
    )
    
    args = parser.parse_args()
    
    # Check if test directory exists
    test_dir = Path(args.test)
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist")
        print("Creating placeholder test directory...")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder label file
        label_file = test_dir / "label.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("sample1.jpg\tសួស្តី ពិភពលោក\n")
            f.write("sample2.jpg\tប្រទេសកម្ពុជា\n")
            f.write("sample3.jpg\tភាសាខ្មែរ\n")
    
    # Initialize harness with real engine
    lm_path = Path(args.lm_path) if args.lm_path else None
    harness = EvaluationHarness(
        test_dir, 
        Path(args.model_dir),
        use_gpu=args.gpu,
        lm_path=lm_path,
        ablation_mode=args.ablation
    )
    
    # Generate report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    harness.generate_report(report_path)
    
    # Check if we should enforce acceptance criteria
    if args.strict:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        if not report.get("acceptance_criteria", {}).get("all_pass", False):
            print("\n❌ FAILED: Acceptance criteria not met")
            sys.exit(1)
        else:
            print("\n✅ PASSED: All acceptance criteria met")
            sys.exit(0)


if __name__ == "__main__":
    main()