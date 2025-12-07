"""
Simplified Coefficient Analysis - Only Score vs Coefficient Plots

Creates one fine-grained (0.1 increment) score vs coefficient plot per concept.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

# Assuming these imports from your codebase
from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import SteeringConfig, generate_with_steering, generate_baseline
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics, compute_repetition_ratio


class CoefficientAnalyzer:
    """Analyzes the effect of steering coefficients on model outputs."""
    
    def __init__(self, model, tokenizer, steering_vectors: Dict[str, torch.Tensor],
                 optimal_layers: Dict[str, int], prompts: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors = steering_vectors
        self.optimal_layers = optimal_layers
        self.prompts = prompts
        self.quality_metrics = QualityMetrics()
        
    def analyze_coefficient_range(
        self,
        concept: str,
        coef_range: List[float],
        n_generations: int = 5
    ) -> Dict:
        """
        Analyze a range of coefficients for a single concept.
        """
        # Validate concept exists
        if concept not in self.steering_vectors:
            raise KeyError(f"Concept '{concept}' not found in steering vectors. "
                          f"Available concepts: {list(self.steering_vectors.keys())}")
        
        if concept not in self.optimal_layers:
            raise KeyError(f"Concept '{concept}' not found in optimal layers. "
                          f"Available concepts: {list(self.optimal_layers.keys())}")
        
        # Initialize evaluator
        evaluator = MultiAttributeEvaluator([concept])
        vec = self.steering_vectors[concept]
        layer = self.optimal_layers[concept]
        
        results = {
            "concept": concept,
            "layer": layer,
            "coefficient_range": coef_range,
            "data": []
        }
        
        print(f"\nAnalyzing '{concept}' across {len(coef_range)} coefficient values")
        print(f"Range: [{min(coef_range):.2f}, {max(coef_range):.2f}]")
        
        for coef in tqdm(coef_range, desc="Coefficients"):
            scores = []
            perplexities = []
            repetition_rates = []
            text_lengths = []
            texts = []
            
            print(f"\n  Testing coefficient {coef:.2f}...", end="", flush=True)
            
            for prompt in self.prompts:
                for gen_idx in range(n_generations):
                    try:
                        # Generate text
                        if coef == 0.0:
                            text = generate_baseline(
                                self.model, self.tokenizer, prompt,
                                max_new_tokens=50,
                                temperature=0.7,
                                device=self.model.device
                            )
                        else:
                            config = SteeringConfig(vector=vec, layer=layer, coefficient=coef)
                            text = generate_with_steering(
                                self.model, self.tokenizer, prompt, config,
                                max_new_tokens=50,
                                temperature=0.7,
                                device=self.model.device
                            )
                        
                        texts.append(text)
                        
                        # Evaluate
                        score = evaluator.classifiers[concept].score(text)
                        scores.append(score)
                        
                        ppl = self.quality_metrics.perplexity_calc.compute(text)
                        perplexities.append(ppl)
                        
                        rep = compute_repetition_ratio(text)
                        repetition_rates.append(rep)
                        
                        text_lengths.append(len(text.split()))
                        
                        # Progress indicator
                        if (gen_idx + 1) % 5 == 0:
                            print(".", end="", flush=True)
                    
                    except Exception as e:
                        print(f"\n    Warning: Generation failed: {e}")
                        scores.append(0.0)
                        perplexities.append(1000.0)
                        repetition_rates.append(1.0)
                        text_lengths.append(0)
                        texts.append("")
            
            print(f" done! (avg score: {np.mean(scores):.3f})", flush=True)
            
            # Compute statistics
            result = {
                "coefficient": float(coef),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores)),
                "min_score": float(np.min(scores)),
                "max_score": float(np.max(scores)),
                "q25_score": float(np.percentile(scores, 25)),
                "q75_score": float(np.percentile(scores, 75)),
                "mean_perplexity": float(np.mean(perplexities)),
                "std_perplexity": float(np.std(perplexities)),
                "median_perplexity": float(np.median(perplexities)),
                "mean_repetition": float(np.mean(repetition_rates)),
                "std_repetition": float(np.std(repetition_rates)),
                "mean_length": float(np.mean(text_lengths)),
                "success_rate": float(np.mean([s > 0.5 for s in scores])),
                "strong_success_rate": float(np.mean([s > 0.7 for s in scores])),
                "quality_score": float(np.mean(scores) / (1 + np.log(max(np.mean(perplexities), 1.0)))),
                "all_scores": scores,
                "sample_texts": texts[:3]
            }
            
            results["data"].append(result)
        
        # Find optimal coefficient
        valid_results = [r for r in results["data"] if r["mean_perplexity"] < 100]
        if valid_results:
            optimal = max(valid_results, key=lambda x: x["quality_score"])
            results["optimal_coefficient"] = optimal["coefficient"]
            results["optimal_score"] = optimal["mean_score"]
            results["optimal_perplexity"] = optimal["mean_perplexity"]
        
        return results


class CoefficientVisualizer:
    """Creates score vs coefficient visualizations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures" / "coefficients"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print(f"📊 Visualizer initialized. Figures will be saved to: {self.figures_dir}")
    
    def plot_score_vs_coefficient(self, results: Dict, save_suffix: str = ""):
        """
        Plot: Score vs coefficient with error bars and confidence bands.
        """
        data = results["data"]
        concept = results["concept"]
        
        coefficients = [d["coefficient"] for d in data]
        mean_scores = [d["mean_score"] for d in data]
        std_scores = [d["std_score"] for d in data]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Main line
        ax.plot(coefficients, mean_scores, 'o-', linewidth=2, markersize=8,
                label='Mean Score', color='#2E86AB', alpha=0.9)
        
        # Error bars
        ax.errorbar(coefficients, mean_scores, yerr=std_scores,
                   fmt='none', ecolor='#A23B72', alpha=0.3, capsize=5)
        
        # Percentile bands
        if "q25_score" in data[0]:
            q25 = [d["q25_score"] for d in data]
            q75 = [d["q75_score"] for d in data]
            ax.fill_between(coefficients, q25, q75, alpha=0.2, 
                           color='#2E86AB', label='25-75 Percentile')
        
        # Mark baseline (coefficient = 0)
        baseline_idx = None
        for i, coef in enumerate(coefficients):
            if abs(coef) < 1e-6:
                baseline_idx = i
                break
        
        if baseline_idx is not None:
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
            ax.plot(0, mean_scores[baseline_idx], 'r*', markersize=15, 
                   label=f'Baseline Score: {mean_scores[baseline_idx]:.3f}')
        
        # Mark optimal coefficient if available
        if "optimal_coefficient" in results:
            opt_coef = results["optimal_coefficient"]
            opt_score = results["optimal_score"]
            ax.axvline(x=opt_coef, color='green', linestyle='--', alpha=0.5)
            ax.plot(opt_coef, opt_score, 'g*', markersize=15,
                   label=f'Optimal: {opt_coef:.2f} (score={opt_score:.3f})')
        
        ax.set_xlabel('Steering Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Concept Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Score vs Coefficient: {concept}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        filename = f"score_vs_coef_{concept}{save_suffix}.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if filepath.exists():
            print(f"  ✓ Saved: {filename} ({filepath.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ✗ Failed to save: {filename}")


def convert_to_native(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def main():
    """Run coefficient analysis - ONLY fine-grained score vs coefficient plots."""
    import argparse
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/coefficient_analysis")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer samples")
    
    # Handle Jupyter/Colab kernel arguments
    if any('kernel' in arg or '.json' in arg for arg in sys.argv):
        sys.argv = [arg for arg in sys.argv if not (arg.startswith('-f') or '.json' in arg)]
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}")
    
    print("="*70)
    print("COEFFICIENT ANALYSIS - SCORE VS COEFFICIENT ONLY")
    print("="*70)
    
    # Load optimal layers
    optimal_layers_path = week1_dir / "optimal_layers.json"
    if not optimal_layers_path.exists():
        raise FileNotFoundError(f"Optimal layers not found at {optimal_layers_path}")
    
    with open(optimal_layers_path) as f:
        optimal_layers = json.load(f)
    print(f"\n✓ Loaded optimal layers: {optimal_layers}")
    
    # Use all concepts from optimal_layers unless specified
    concepts = args.concepts or list(optimal_layers.keys())
    layers = cfg.model.steering_layers
    
    print(f"\nAnalyzing concepts: {concepts}")
    print(f"Output: {output_dir}")
    
    # Clear GPU memory and load model
    print("\nClearing GPU memory...")
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load steering vectors
    print("Loading steering vectors...")
    
    vectors_dir = week1_dir / "vectors"
    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")
    
    print(f"  Looking in: {vectors_dir}")
    
    available_files = list(vectors_dir.glob("*.pt"))
    print(f"  Found {len(available_files)} vector files")
    
    # Load vectors manually with map_location
    steering_vectors_by_layer = {}
    for concept in concepts:
        steering_vectors_by_layer[concept] = {}
        for layer in layers:
            # Try both naming conventions
            vec_path1 = week1_dir / "vectors" / f"{concept}_layer_{layer}.pt"
            vec_path2 = week1_dir / "vectors" / f"{concept}_layer{layer}.pt"
            
            vec_path = vec_path1 if vec_path1.exists() else vec_path2
            
            if vec_path.exists():
                try:
                    vec = torch.load(vec_path, map_location='cpu')
                    steering_vectors_by_layer[concept][layer] = vec
                except Exception as load_error:
                    print(f"    ✗ Failed to load {concept} layer {layer}: {load_error}")
    
    print(f"\n  Loaded vectors for concepts: {list(steering_vectors_by_layer.keys())}")
    
    steering_vectors = {
        c: steering_vectors_by_layer[c][optimal_layers[c]].to(model.device)
        for c in concepts if c in steering_vectors_by_layer and optimal_layers[c] in steering_vectors_by_layer[c]
    }
    
    print(f"✓ Loaded {len(steering_vectors)} steering vectors")
    print(f"Available concepts: {list(steering_vectors.keys())}")
    
    # Filter concepts to only those with loaded vectors
    concepts = [c for c in concepts if c in steering_vectors]
    
    if not concepts:
        raise ValueError("No steering vectors loaded successfully. Check your week1_dir and concept names.")
    
    print(f"✓ Using concepts: {concepts}")
    
    # Get prompts
    prompts = get_test_prompts()
    if args.quick:
        prompts = prompts[:2]
        n_gen = 2
    else:
        prompts = prompts[:5]
        n_gen = 3
    
    # Initialize analyzer and visualizer
    analyzer = CoefficientAnalyzer(model, tokenizer, steering_vectors, optimal_layers, prompts)
    visualizer = CoefficientVisualizer(output_dir)
    
    all_results = {}
    
    # Analysis for each concept - ONLY fine-grained score vs coefficient
    for concept in concepts:
        print(f"\n{'='*70}")
        print(f"ANALYZING: {concept}")
        print(f"{'='*70}")
        
        # Fine-grained analysis (increments of 0.1) - ONLY THIS
        print("\nFine-grained analysis (0.1 increments)")
        fine_range = np.arange(-2.0, 3.1, 0.1).tolist()
        fine_results = analyzer.analyze_coefficient_range(concept, fine_range, n_gen)
        
        print("\nGenerating visualization...")
        visualizer.plot_score_vs_coefficient(fine_results, save_suffix="_fine_0.1")
        
        all_results[concept] = {"fine": fine_results}
        
        print(f"\n✓ Completed analysis for {concept}")
    
    # Save results
    all_results = convert_to_native(all_results)
    
    try:
        with open(output_dir / "coefficient_analysis_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {output_dir / 'coefficient_analysis_results.json'}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save results: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {output_dir / 'figures' / 'coefficients'}")
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for concept in concepts:
        print(f"\n{concept.upper()}:")
        fine_data = all_results[concept]["fine"]["data"]
        
        # Find coefficient with highest score
        max_score_result = max(fine_data, key=lambda x: x["mean_score"])
        print(f"  Max score: {max_score_result['mean_score']:.3f} at coefficient {max_score_result['coefficient']:.2f}")
        
        # Find optimal (balancing score and quality)
        if "optimal_coefficient" in all_results[concept]["fine"]:
            opt_coef = all_results[concept]["fine"]["optimal_coefficient"]
            opt_score = all_results[concept]["fine"]["optimal_score"]
            print(f"  Optimal (quality-adjusted): {opt_score:.3f} at coefficient {opt_coef:.2f}")
        
        # Baseline comparison
        baseline_result = [d for d in fine_data if abs(d["coefficient"]) < 0.05][0]
        print(f"  Baseline score: {baseline_result['mean_score']:.3f}")
        print(f"  Improvement: {(max_score_result['mean_score'] - baseline_result['mean_score']):.3f}")
    
    return all_results


if __name__ == "__main__":
    main()