#!/usr/bin/env python3
"""Latency benchmarking for Khmer OCR pipeline

Tests end-to-end, detection-only, and recognition-only performance
with platform-aware thresholds (200ms CPU, 50ms GPU).
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Import OCR engine
import sys
sys.path.append(str(Path(__file__).parent.parent))
from infer.engine import OCREngine


class LatencyBenchmark:
    """Benchmark OCR pipeline latency"""
    
    def __init__(
        self,
        model_dir: Path,
        use_gpu: bool = False,
        use_onnx: bool = False
    ):
        """
        Initialize benchmark
        
        Args:
            model_dir: Directory containing models
            use_gpu: Whether to use GPU
            use_onnx: Whether to use ONNX models
        """
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self.use_onnx = use_onnx
        
        # Platform detection
        self.platform = platform.system()
        self.processor = platform.processor()
        
        # Set platform-aware thresholds
        if self.platform == "Darwin":  # macOS
            self.threshold_p95 = 200  # ms for CPU
            self.threshold_p50 = 100
        elif self.use_gpu:
            self.threshold_p95 = 50   # ms for GPU
            self.threshold_p50 = 25
        else:
            self.threshold_p95 = 200  # ms for CPU
            self.threshold_p50 = 100
        
        # Initialize engine
        print(f"Initializing OCR engine (GPU: {use_gpu}, ONNX: {use_onnx})")
        self.engine = OCREngine(
            model_dir=model_dir,
            use_gpu=use_gpu
        )
        
        self.results = {}
    
    def generate_test_images(
        self,
        sizes: List[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Generate test images of various sizes
        
        Args:
            sizes: List of (width, height) tuples
            
        Returns:
            List of test images
        """
        if sizes is None:
            sizes = [
                (320, 48),    # Small text line
                (640, 100),   # Medium text block
                (1280, 200),  # Large text block
                (1920, 1080), # Full page
            ]
        
        images = []
        for width, height in sizes:
            # Create synthetic image with text-like patterns
            img = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Add some horizontal lines to simulate text
            for y in range(20, height - 20, 30):
                img[y:y+2, 20:-20] = 0
            
            images.append(img)
            
        return images
    
    def benchmark_batch(
        self,
        batch_sizes: List[int] = [1, 4, 16, 32],
        warmup_runs: int = 10,
        test_runs: int = 100
    ) -> Dict[str, any]:
        """
        Benchmark different batch sizes
        
        Args:
            batch_sizes: Batch sizes to test
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
            
        Returns:
            Benchmark results
        """
        print(f"\n{'='*60}")
        print("BATCH SIZE BENCHMARKING")
        print(f"{'='*60}")
        
        results = {}
        test_images = self.generate_test_images()
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            
            # Prepare batch
            if batch_size == 1:
                batch = test_images[0]
            else:
                # Repeat images to match batch size
                batch = test_images[:min(batch_size, len(test_images))]
                while len(batch) < batch_size:
                    batch.extend(test_images[:min(batch_size - len(batch), len(test_images))])
            
            # Warmup
            print(f"  Warmup ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                if batch_size == 1:
                    _ = self._process_image(batch)
                else:
                    for img in batch:
                        _ = self._process_image(img)
            
            # Benchmark
            print(f"  Testing ({test_runs} runs)...")
            times = []
            for _ in range(test_runs):
                start = time.perf_counter()
                
                if batch_size == 1:
                    _ = self._process_image(batch)
                else:
                    for img in batch:
                        _ = self._process_image(img)
                
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed / batch_size)  # Per-image time
            
            # Calculate statistics
            times = np.array(times)
            stats = {
                "batch_size": batch_size,
                "mean_ms": np.mean(times),
                "std_ms": np.std(times),
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "p50_ms": np.percentile(times, 50),
                "p95_ms": np.percentile(times, 95),
                "p99_ms": np.percentile(times, 99),
                "throughput": 1000 / np.mean(times),  # images/sec
                "meets_threshold": np.percentile(times, 95) <= self.threshold_p95
            }
            
            results[f"batch_{batch_size}"] = stats
            
            # Print results
            print(f"  Results:")
            print(f"    Mean: {stats['mean_ms']:.2f} ms/image")
            print(f"    Std: {stats['std_ms']:.2f} ms")
            print(f"    P50: {stats['p50_ms']:.2f} ms")
            print(f"    P95: {stats['p95_ms']:.2f} ms")
            print(f"    P99: {stats['p99_ms']:.2f} ms")
            print(f"    Throughput: {stats['throughput']:.1f} images/sec")
            
            status = "✓ PASS" if stats['meets_threshold'] else "✗ FAIL"
            print(f"    Threshold ({self.threshold_p95}ms): {status}")
        
        return results
    
    def benchmark_components(
        self,
        warmup_runs: int = 10,
        test_runs: int = 100
    ) -> Dict[str, any]:
        """
        Benchmark individual pipeline components
        
        Args:
            warmup_runs: Number of warmup iterations
            test_runs: Number of test iterations
            
        Returns:
            Component-wise benchmark results
        """
        print(f"\n{'='*60}")
        print("COMPONENT BENCHMARKING")
        print(f"{'='*60}")
        
        test_image = self.generate_test_images([(640, 100)])[0]
        components = {}
        
        # End-to-end timing
        print("\nEnd-to-end pipeline:")
        e2e_times = []
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self._process_image(test_image)
        
        # Test
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = self._process_image(test_image)
            e2e_times.append((time.perf_counter() - start) * 1000)
        
        e2e_times = np.array(e2e_times)
        components["end_to_end"] = {
            "mean_ms": np.mean(e2e_times),
            "p50_ms": np.percentile(e2e_times, 50),
            "p95_ms": np.percentile(e2e_times, 95),
            "p99_ms": np.percentile(e2e_times, 99),
        }
        
        print(f"  Mean: {components['end_to_end']['mean_ms']:.2f} ms")
        print(f"  P95: {components['end_to_end']['p95_ms']:.2f} ms")
        
        # Detection-only timing (if possible)
        if hasattr(self.engine, 'detect_only'):
            print("\nDetection only:")
            det_times = []
            
            for _ in range(warmup_runs):
                _ = self.engine.detect_only(test_image)
            
            for _ in range(test_runs):
                start = time.perf_counter()
                _ = self.engine.detect_only(test_image)
                det_times.append((time.perf_counter() - start) * 1000)
            
            det_times = np.array(det_times)
            components["detection"] = {
                "mean_ms": np.mean(det_times),
                "p50_ms": np.percentile(det_times, 50),
                "p95_ms": np.percentile(det_times, 95),
                "p99_ms": np.percentile(det_times, 99),
            }
            
            print(f"  Mean: {components['detection']['mean_ms']:.2f} ms")
            print(f"  P95: {components['detection']['p95_ms']:.2f} ms")
        
        # Recognition-only timing (if possible)
        if hasattr(self.engine, 'recognize_only'):
            print("\nRecognition only:")
            rec_times = []
            
            # Use a pre-cropped text line
            text_line = self.generate_test_images([(320, 48)])[0]
            
            for _ in range(warmup_runs):
                _ = self.engine.recognize_only(text_line)
            
            for _ in range(test_runs):
                start = time.perf_counter()
                _ = self.engine.recognize_only(text_line)
                rec_times.append((time.perf_counter() - start) * 1000)
            
            rec_times = np.array(rec_times)
            components["recognition"] = {
                "mean_ms": np.mean(rec_times),
                "p50_ms": np.percentile(rec_times, 50),
                "p95_ms": np.percentile(rec_times, 95),
                "p99_ms": np.percentile(rec_times, 99),
            }
            
            print(f"  Mean: {components['recognition']['mean_ms']:.2f} ms")
            print(f"  P95: {components['recognition']['p95_ms']:.2f} ms")
        
        # Post-processing timing
        print("\nPost-processing (LM rescoring):")
        dummy_text = "សួស្តី ពិភពលោក " * 10
        pp_times = []
        
        for _ in range(test_runs):
            start = time.perf_counter()
            _ = self.engine.postprocess(dummy_text) if hasattr(self.engine, 'postprocess') else dummy_text
            pp_times.append((time.perf_counter() - start) * 1000)
        
        pp_times = np.array(pp_times)
        components["postprocessing"] = {
            "mean_ms": np.mean(pp_times),
            "p50_ms": np.percentile(pp_times, 50),
            "p95_ms": np.percentile(pp_times, 95),
            "p99_ms": np.percentile(pp_times, 99),
        }
        
        print(f"  Mean: {components['postprocessing']['mean_ms']:.2f} ms")
        print(f"  P95: {components['postprocessing']['p95_ms']:.2f} ms")
        
        return components
    
    def _process_image(self, image: np.ndarray) -> str:
        """Process single image through OCR pipeline"""
        # Save to temp file (engine expects file path)
        temp_path = Path("/tmp/benchmark_test.jpg")
        if PIL_AVAILABLE:
            Image.fromarray(image).save(temp_path)
        else:
            # Fallback: just create empty file
            temp_path.touch()
        
        result = self.engine.process_image(temp_path)
        return result.get("text", "")
    
    def plot_results(
        self,
        output_dir: Path
    ):
        """
        Generate latency distribution plots
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.results:
            print("No results to plot")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Batch size vs latency
        if "batch_benchmarks" in self.results:
            batch_data = self.results["batch_benchmarks"]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Latency by batch size
            batch_sizes = []
            mean_latencies = []
            p95_latencies = []
            
            for key, stats in batch_data.items():
                batch_sizes.append(stats["batch_size"])
                mean_latencies.append(stats["mean_ms"])
                p95_latencies.append(stats["p95_ms"])
            
            ax1.plot(batch_sizes, mean_latencies, 'o-', label='Mean')
            ax1.plot(batch_sizes, p95_latencies, 's-', label='P95')
            ax1.axhline(y=self.threshold_p95, color='r', linestyle='--', label='Threshold')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Latency vs Batch Size')
            ax1.legend()
            ax1.grid(True)
            
            # Throughput by batch size
            throughputs = [batch_data[f"batch_{bs}"]["throughput"] for bs in batch_sizes]
            ax2.bar(batch_sizes, throughputs)
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (images/sec)')
            ax2.set_title('Throughput vs Batch Size')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / "batch_performance.png")
            plt.close()
        
        # 2. Component breakdown
        if "component_benchmarks" in self.results:
            comp_data = self.results["component_benchmarks"]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            components = list(comp_data.keys())
            mean_times = [comp_data[c]["mean_ms"] for c in components]
            p95_times = [comp_data[c]["p95_ms"] for c in components]
            
            x = np.arange(len(components))
            width = 0.35
            
            ax.bar(x - width/2, mean_times, width, label='Mean', alpha=0.8)
            ax.bar(x + width/2, p95_times, width, label='P95', alpha=0.8)
            
            ax.set_xlabel('Component')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Component-wise Latency Breakdown')
            ax.set_xticks(x)
            ax.set_xticklabels(components, rotation=45, ha='right')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / "component_breakdown.png")
            plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def generate_report(
        self,
        output_path: Path
    ):
        """
        Generate benchmark report
        
        Args:
            output_path: Path to save report
        """
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": {
                "system": self.platform,
                "processor": self.processor,
                "gpu": self.use_gpu,
                "onnx": self.use_onnx,
            },
            "thresholds": {
                "p50_target_ms": self.threshold_p50,
                "p95_target_ms": self.threshold_p95,
            },
            "results": self.results,
            "summary": {}
        }
        
        # Add summary
        if "batch_benchmarks" in self.results:
            batch_1 = self.results["batch_benchmarks"].get("batch_1", {})
            report["summary"]["batch_1_p95"] = batch_1.get("p95_ms")
            report["summary"]["batch_1_throughput"] = batch_1.get("throughput")
            report["summary"]["meets_threshold"] = batch_1.get("meets_threshold", False)
        
        if "component_benchmarks" in self.results:
            e2e = self.results["component_benchmarks"].get("end_to_end", {})
            report["summary"]["end_to_end_p95"] = e2e.get("p95_ms")
        
        # Overall pass/fail
        all_pass = all([
            report["summary"].get("meets_threshold", False),
            report["summary"].get("end_to_end_p95", float('inf')) <= self.threshold_p95
        ])
        report["summary"]["all_pass"] = all_pass
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Platform: {self.platform} ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"P95 Target: {self.threshold_p95} ms")
        
        if "summary" in report:
            s = report["summary"]
            if "batch_1_p95" in s:
                status = "✓ PASS" if s.get("meets_threshold") else "✗ FAIL"
                print(f"Batch-1 P95: {s['batch_1_p95']:.2f} ms [{status}]")
            if "batch_1_throughput" in s:
                print(f"Throughput: {s['batch_1_throughput']:.1f} images/sec")
            if "end_to_end_p95" in s:
                print(f"End-to-end P95: {s['end_to_end_p95']:.2f} ms")
        
        overall = "✓ ALL TESTS PASSED" if all_pass else "✗ SOME TESTS FAILED"
        print(f"\n{overall}")
        print(f"\nReport saved to: {output_path}")


def main():
    """CLI for latency benchmarking"""
    parser = argparse.ArgumentParser(description="Benchmark OCR latency")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Model directory"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX models"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup runs"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Test runs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/benchmark_report.json"),
        help="Output report path"
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("eval/plots"),
        help="Directory for plots"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = LatencyBenchmark(
        model_dir=args.model_dir,
        use_gpu=args.gpu,
        use_onnx=args.onnx
    )
    
    # Run benchmarks
    print("Starting latency benchmarks...")
    
    # Batch size benchmarks
    batch_results = benchmark.benchmark_batch(
        batch_sizes=args.batch_sizes,
        warmup_runs=args.warmup,
        test_runs=args.runs
    )
    benchmark.results["batch_benchmarks"] = batch_results
    
    # Component benchmarks
    component_results = benchmark.benchmark_components(
        warmup_runs=args.warmup,
        test_runs=args.runs
    )
    benchmark.results["component_benchmarks"] = component_results
    
    # Generate plots
    benchmark.plot_results(args.plot_dir)
    
    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.generate_report(args.output)


if __name__ == "__main__":
    main()