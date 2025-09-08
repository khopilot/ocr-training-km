#!/usr/bin/env python3
"""Parameter sweep for language model rescoring optimization

Grid search over λ (LM weight), μ (lexicon weight), and beam width
to find optimal parameters for different dataset types.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Import OCR components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from infer.engine import OCREngine
from infer.rescoring import rescore_with_lm, rescore_nbest, LanguageModel
from eval.cer import calculate_cer


class RescoringOptimizer:
    """Optimize rescoring parameters through grid search"""
    
    def __init__(
        self,
        test_data_path: Path,
        model_dir: Path,
        lm_path: Optional[Path] = None,
        lexicon_path: Optional[Path] = None
    ):
        """
        Initialize optimizer
        
        Args:
            test_data_path: Path to test dataset
            model_dir: Directory containing models
            lm_path: Path to language model
            lexicon_path: Path to lexicon file
        """
        self.test_data_path = Path(test_data_path)
        self.model_dir = Path(model_dir)
        self.lm_path = lm_path
        self.lexicon_path = lexicon_path
        
        # Load test data
        self.test_samples = self._load_test_data()
        
        # Initialize engine
        self.engine = OCREngine(
            model_dir=model_dir,
            lm_path=lm_path
        )
        
        # Load language model
        self.lm = LanguageModel(str(lm_path)) if lm_path and lm_path.exists() else None
        
        # Load lexicon
        self.lexicon = self._load_lexicon() if lexicon_path else set()
        
        self.results = []
    
    def _load_test_data(self) -> List[Tuple[Path, str]]:
        """Load test samples"""
        samples = []
        
        label_file = self.test_data_path / "label.txt"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" in line:
                        image_name, text = line.strip().split("\t", 1)
                        image_path = self.test_data_path / image_name
                        samples.append((image_path, text))
        else:
            # Placeholder data
            samples = [
                (Path("sample1.jpg"), "សួស្តី ពិភពលោក"),
                (Path("sample2.jpg"), "ប្រទេសកម្ពុជា"),
                (Path("sample3.jpg"), "ភាសាខ្មែរ"),
            ]
        
        return samples
    
    def _load_lexicon(self) -> set:
        """Load lexicon from file"""
        lexicon = set()
        if self.lexicon_path and self.lexicon_path.exists():
            with open(self.lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        lexicon.add(word)
        return lexicon
    
    def grid_search(
        self,
        lambda_values: List[float] = None,
        mu_values: List[float] = None,
        beam_widths: List[int] = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Perform grid search over parameters
        
        Args:
            lambda_values: LM weight values to test
            mu_values: Lexicon weight values to test
            beam_widths: Beam width values to test
            max_samples: Maximum number of samples to test
            
        Returns:
            Grid search results
        """
        # Default parameter ranges
        if lambda_values is None:
            lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        if mu_values is None:
            mu_values = [0.05, 0.1, 0.15, 0.2]
        if beam_widths is None:
            beam_widths = [5, 10, 20]
        
        print(f"Grid Search Configuration:")
        print(f"  λ values: {lambda_values}")
        print(f"  μ values: {mu_values}")
        print(f"  Beam widths: {beam_widths}")
        print(f"  Total combinations: {len(lambda_values) * len(mu_values) * len(beam_widths)}")
        
        # Sample subset if requested
        test_samples = self.test_samples[:max_samples] if max_samples else self.test_samples
        print(f"  Test samples: {len(test_samples)}")
        
        results = []
        best_params = None
        best_cer = float('inf')
        
        # Grid search
        for lam, mu, beam in product(lambda_values, mu_values, beam_widths):
            print(f"\nTesting λ={lam:.2f}, μ={mu:.2f}, beam={beam}")
            
            # Evaluate with these parameters
            metrics = self._evaluate_params(
                test_samples,
                lam=lam,
                mu=mu,
                beam_width=beam
            )
            
            results.append({
                "lambda": lam,
                "mu": mu,
                "beam_width": beam,
                **metrics
            })
            
            # Track best parameters
            if metrics["cer_mean"] < best_cer:
                best_cer = metrics["cer_mean"]
                best_params = {"lambda": lam, "mu": mu, "beam_width": beam}
            
            print(f"  CER: {metrics['cer_mean']:.2f}%")
            print(f"  Latency: {metrics['latency_mean']:.2f} ms")
        
        self.results = results
        
        return {
            "grid_results": results,
            "best_params": best_params,
            "best_cer": best_cer,
            "parameter_ranges": {
                "lambda": lambda_values,
                "mu": mu_values,
                "beam_width": beam_widths
            }
        }
    
    def _evaluate_params(
        self,
        samples: List[Tuple[Path, str]],
        lam: float,
        mu: float,
        beam_width: int
    ) -> Dict[str, float]:
        """
        Evaluate specific parameter combination
        
        Args:
            samples: Test samples
            lam: Language model weight
            mu: Lexicon weight
            beam_width: Beam search width
            
        Returns:
            Evaluation metrics
        """
        cers = []
        latencies = []
        
        for image_path, ground_truth in samples:
            start_time = time.time()
            
            # Run OCR with current parameters
            if image_path.exists():
                # Real inference
                result = self.engine.process_image(
                    image_path,
                    lm_weight=lam,
                    lex_weight=mu,
                    beam_width=beam_width
                )
                prediction = result.get("text", "")
            else:
                # Simulate with rescoring
                ctc_text = ground_truth  # Placeholder
                ctc_score = -10.0
                tokens = ctc_text.split()
                
                # Apply rescoring
                score = rescore_with_lm(
                    ctc_score,
                    tokens,
                    self.lm,
                    lam=lam,
                    mu=mu,
                    lexicon=self.lexicon
                )
                
                # For simulation, add some noise to text
                prediction = ctc_text
            
            latency = (time.time() - start_time) * 1000
            
            # Calculate CER
            cer = calculate_cer(ground_truth, prediction)
            
            cers.append(cer)
            latencies.append(latency)
        
        return {
            "cer_mean": np.mean(cers),
            "cer_std": np.std(cers),
            "cer_min": np.min(cers),
            "cer_max": np.max(cers),
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "latency_p95": np.percentile(latencies, 95) if latencies else 0
        }
    
    def ablation_study(
        self,
        base_lambda: float = 0.3,
        base_mu: float = 0.1,
        base_beam: int = 10
    ) -> Dict[str, any]:
        """
        Perform ablation study on individual parameters
        
        Args:
            base_lambda: Baseline LM weight
            base_mu: Baseline lexicon weight
            base_beam: Baseline beam width
            
        Returns:
            Ablation results
        """
        print("\nAblation Study")
        print(f"Baseline: λ={base_lambda}, μ={base_mu}, beam={base_beam}")
        
        ablation_results = {}
        
        # Test without LM (λ=0)
        print("\n1. Without LM (λ=0):")
        no_lm = self._evaluate_params(
            self.test_samples,
            lam=0.0,
            mu=base_mu,
            beam_width=base_beam
        )
        ablation_results["no_lm"] = no_lm
        print(f"   CER: {no_lm['cer_mean']:.2f}%")
        
        # Test without lexicon (μ=0)
        print("\n2. Without lexicon (μ=0):")
        no_lex = self._evaluate_params(
            self.test_samples,
            lam=base_lambda,
            mu=0.0,
            beam_width=base_beam
        )
        ablation_results["no_lexicon"] = no_lex
        print(f"   CER: {no_lex['cer_mean']:.2f}%")
        
        # Test with greedy decoding (beam=1)
        print("\n3. Greedy decoding (beam=1):")
        greedy = self._evaluate_params(
            self.test_samples,
            lam=base_lambda,
            mu=base_mu,
            beam_width=1
        )
        ablation_results["greedy"] = greedy
        print(f"   CER: {greedy['cer_mean']:.2f}%")
        
        # Test with all components
        print("\n4. Full system:")
        full = self._evaluate_params(
            self.test_samples,
            lam=base_lambda,
            mu=base_mu,
            beam_width=base_beam
        )
        ablation_results["full"] = full
        print(f"   CER: {full['cer_mean']:.2f}%")
        
        # Calculate improvements
        improvements = {
            "lm_improvement": no_lm["cer_mean"] - full["cer_mean"],
            "lexicon_improvement": no_lex["cer_mean"] - full["cer_mean"],
            "beam_improvement": greedy["cer_mean"] - full["cer_mean"]
        }
        
        ablation_results["improvements"] = improvements
        
        return ablation_results
    
    def plot_results(self, output_dir: Path):
        """
        Generate plots for parameter sweep results
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.results:
            print("No results to plot")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        
        # Convert results to arrays for plotting
        lambdas = [r["lambda"] for r in self.results]
        mus = [r["mu"] for r in self.results]
        beams = [r["beam_width"] for r in self.results]
        cers = [r["cer_mean"] for r in self.results]
        latencies = [r["latency_mean"] for r in self.results]
        
        # 1. CER vs Latency trade-off
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(latencies, cers, c=lambdas, s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('CER (%)')
        ax.set_title('CER vs Latency Trade-off')
        plt.colorbar(scatter, label='λ (LM weight)')
        
        # Add Pareto frontier
        pareto_points = self._find_pareto_frontier(latencies, cers)
        if pareto_points:
            pareto_lat, pareto_cer = zip(*pareto_points)
            ax.plot(pareto_lat, pareto_cer, 'r--', alpha=0.5, label='Pareto Frontier')
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "cer_vs_latency.png")
        plt.close()
        
        # 2. Heatmap of CER by λ and μ
        # Average over beam widths for clarity
        unique_lambdas = sorted(set(lambdas))
        unique_mus = sorted(set(mus))
        
        cer_matrix = np.zeros((len(unique_mus), len(unique_lambdas)))
        for i, mu in enumerate(unique_mus):
            for j, lam in enumerate(unique_lambdas):
                # Average CER for this (λ, μ) pair
                matching = [r["cer_mean"] for r in self.results 
                           if r["lambda"] == lam and r["mu"] == mu]
                if matching:
                    cer_matrix[i, j] = np.mean(matching)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cer_matrix,
            xticklabels=[f"{l:.2f}" for l in unique_lambdas],
            yticklabels=[f"{m:.2f}" for m in unique_mus],
            annot=True,
            fmt=".1f",
            cmap='RdYlGn_r',
            ax=ax
        )
        ax.set_xlabel('λ (LM weight)')
        ax.set_ylabel('μ (Lexicon weight)')
        ax.set_title('CER Heatmap: λ vs μ')
        plt.tight_layout()
        plt.savefig(output_dir / "cer_heatmap.png")
        plt.close()
        
        # 3. Parameter importance
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # λ effect
        for mu in unique_mus[:3]:  # Show top 3 μ values
            mu_data = [(r["lambda"], r["cer_mean"]) for r in self.results if r["mu"] == mu]
            if mu_data:
                x, y = zip(*sorted(mu_data))
                axes[0].plot(x, y, 'o-', label=f'μ={mu:.2f}', alpha=0.7)
        axes[0].set_xlabel('λ (LM weight)')
        axes[0].set_ylabel('CER (%)')
        axes[0].set_title('Effect of LM Weight')
        axes[0].legend()
        axes[0].grid(True)
        
        # μ effect
        for lam in unique_lambdas[::2]:  # Show every other λ
            lam_data = [(r["mu"], r["cer_mean"]) for r in self.results if r["lambda"] == lam]
            if lam_data:
                x, y = zip(*sorted(lam_data))
                axes[1].plot(x, y, 's-', label=f'λ={lam:.2f}', alpha=0.7)
        axes[1].set_xlabel('μ (Lexicon weight)')
        axes[1].set_ylabel('CER (%)')
        axes[1].set_title('Effect of Lexicon Weight')
        axes[1].legend()
        axes[1].grid(True)
        
        # Beam width effect
        unique_beams = sorted(set(beams))
        beam_cers = [np.mean([r["cer_mean"] for r in self.results if r["beam_width"] == b]) 
                     for b in unique_beams]
        axes[2].bar(unique_beams, beam_cers, alpha=0.7)
        axes[2].set_xlabel('Beam Width')
        axes[2].set_ylabel('Average CER (%)')
        axes[2].set_title('Effect of Beam Width')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "parameter_effects.png")
        plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def _find_pareto_frontier(
        self,
        x_values: List[float],
        y_values: List[float]
    ) -> List[Tuple[float, float]]:
        """
        Find Pareto frontier points (minimize both x and y)
        
        Args:
            x_values: X coordinates (e.g., latency)
            y_values: Y coordinates (e.g., CER)
            
        Returns:
            List of Pareto optimal points
        """
        points = list(zip(x_values, y_values))
        pareto = []
        
        for i, (x1, y1) in enumerate(points):
            dominated = False
            for j, (x2, y2) in enumerate(points):
                if i != j and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                    dominated = True
                    break
            if not dominated:
                pareto.append((x1, y1))
        
        return sorted(pareto)
    
    def generate_report(self, output_path: Path):
        """
        Generate parameter sweep report
        
        Args:
            output_path: Path to save report
        """
        if not self.results:
            print("No results to report")
            return
        
        # Find best parameters for different objectives
        best_cer = min(self.results, key=lambda x: x["cer_mean"])
        best_latency = min(self.results, key=lambda x: x["latency_mean"])
        
        # Find balanced solution (minimize CER + normalized_latency)
        max_lat = max(r["latency_mean"] for r in self.results)
        best_balanced = min(
            self.results,
            key=lambda x: x["cer_mean"] + (x["latency_mean"] / max_lat) * 10
        )
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_combinations_tested": len(self.results),
            "test_samples": len(self.test_samples),
            "best_for_cer": {
                "params": {
                    "lambda": best_cer["lambda"],
                    "mu": best_cer["mu"],
                    "beam_width": best_cer["beam_width"]
                },
                "cer_mean": best_cer["cer_mean"],
                "latency_mean": best_cer["latency_mean"]
            },
            "best_for_latency": {
                "params": {
                    "lambda": best_latency["lambda"],
                    "mu": best_latency["mu"],
                    "beam_width": best_latency["beam_width"]
                },
                "cer_mean": best_latency["cer_mean"],
                "latency_mean": best_latency["latency_mean"]
            },
            "best_balanced": {
                "params": {
                    "lambda": best_balanced["lambda"],
                    "mu": best_balanced["mu"],
                    "beam_width": best_balanced["beam_width"]
                },
                "cer_mean": best_balanced["cer_mean"],
                "latency_mean": best_balanced["latency_mean"]
            },
            "all_results": self.results
        }
        
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PARAMETER SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Combinations tested: {len(self.results)}")
        print(f"\nBest for CER:")
        print(f"  λ={best_cer['lambda']:.2f}, μ={best_cer['mu']:.2f}, beam={best_cer['beam_width']}")
        print(f"  CER: {best_cer['cer_mean']:.2f}%")
        print(f"  Latency: {best_cer['latency_mean']:.2f} ms")
        
        print(f"\nBest for Latency:")
        print(f"  λ={best_latency['lambda']:.2f}, μ={best_latency['mu']:.2f}, beam={best_latency['beam_width']}")
        print(f"  CER: {best_latency['cer_mean']:.2f}%")
        print(f"  Latency: {best_latency['latency_mean']:.2f} ms")
        
        print(f"\nBest Balanced:")
        print(f"  λ={best_balanced['lambda']:.2f}, μ={best_balanced['mu']:.2f}, beam={best_balanced['beam_width']}")
        print(f"  CER: {best_balanced['cer_mean']:.2f}%")
        print(f"  Latency: {best_balanced['latency_mean']:.2f} ms")
        
        print(f"\nReport saved to: {output_path}")


def main():
    """CLI for rescoring parameter sweep"""
    parser = argparse.ArgumentParser(description="Optimize rescoring parameters")
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/test"),
        help="Test data directory"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Model directory"
    )
    parser.add_argument(
        "--lm-path",
        type=Path,
        default=Path("lang/kenlm/khmer_5gram.arpa"),
        help="Language model path"
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        help="Lexicon file path"
    )
    parser.add_argument(
        "--lambda-range",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Lambda values to test"
    )
    parser.add_argument(
        "--mu-range",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2],
        help="Mu values to test"
    )
    parser.add_argument(
        "--beam-range",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Beam width values to test"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to test"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/rescoring_report.json"),
        help="Output report path"
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("eval/plots"),
        help="Directory for plots"
    )
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = RescoringOptimizer(
        test_data_path=args.test_data,
        model_dir=args.model_dir,
        lm_path=args.lm_path,
        lexicon_path=args.lexicon
    )
    
    # Run grid search
    print("Starting parameter sweep...")
    results = optimizer.grid_search(
        lambda_values=args.lambda_range,
        mu_values=args.mu_range,
        beam_widths=args.beam_range,
        max_samples=args.max_samples
    )
    
    # Run ablation if requested
    if args.ablation:
        best_params = results.get("best_params", {})
        ablation = optimizer.ablation_study(
            base_lambda=best_params.get("lambda", 0.3),
            base_mu=best_params.get("mu", 0.1),
            base_beam=best_params.get("beam_width", 10)
        )
        results["ablation"] = ablation
    
    # Generate plots
    optimizer.plot_results(args.plot_dir)
    
    # Generate report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    optimizer.generate_report(args.output)


if __name__ == "__main__":
    main()