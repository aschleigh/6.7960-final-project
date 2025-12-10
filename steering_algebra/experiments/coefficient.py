"""
Coefficient Analysis: Score vs Coefficient & Optimization Stats.

Fixes:
1. Dynamic Perplexity Threshold (Relative to Baseline) to prevent false failures.
2. Wider sweep range (-2 to +3) to capture full behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import sys

# Ensure project root is in path for imports
sys.path.append(".") 

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import SteeringConfig, generate_with_steering, generate_baseline
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics


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
        # Check keys
        if concept not in self.steering_vectors:
            raise KeyError(f"Concept '{concept}' missing from vectors.")
        
        evaluator = MultiAttributeEvaluator([concept], device="cpu") # Force CPU to avoid OOM
        vec = self.steering_vectors[concept]
        layer = self.optimal_layers[concept]
        
        results = {
            "concept": concept,
            "layer": layer,
            "coefficient_range": coef_range,
            "data": []
        }
        
        print(f"\nAnalyzing '{concept}' (Layer {layer})")
        
        # 1. Establish Baseline PPL first (to set threshold)
        baseline_ppls = []
        for prompt in self.prompts[:3]:
            txt = generate_baseline(self.model, self.tokenizer, prompt, max_new_tokens=30)
            baseline_ppls.append(self.quality_metrics.perplexity_calc.compute(txt))
        
        avg_base_ppl = np.mean(baseline_ppls)
        ppl_threshold = max(100.0, avg_base_ppl * 3.0) # Allow 3x degradation or at least 100
        print(f"  Baseline PPL: {avg_base_ppl:.1f} | Threshold: {ppl_threshold:.1f}")

        # 2. Run Sweep
        for coef in tqdm(coef_range, desc="Sweeping"):
            scores = []
            perplexities = []
            
            for prompt in self.prompts:
                for gen_idx in range(n_generations):
                    try:
                        # Generate
                        if abs(coef) < 1e-6:
                            text = generate_baseline(
                                self.model, self.tokenizer, prompt,
                                max_new_tokens=40,
                                temperature=0.7,
                                device=self.model.device
                            )
                        else:
                            config = SteeringConfig(vector=vec, layer=layer, coefficient=coef)
                            text = generate_with_steering(
                                self.model, self.tokenizer, prompt, config,
                                max_new_tokens=40,
                                temperature=0.7,
                                device=self.model.device
                            )
                        
                        # Score
                        score = evaluator.classifiers[concept].score(text)
                        scores.append(score)
                        
                        ppl = self.quality_metrics.perplexity_calc.compute(text)
                        perplexities.append(ppl)
                    
                    except Exception as e:
                        # Generation failure (OOM, etc)
                        scores.append(0.0)
                        perplexities.append(9999.0)
            
            mean_score = float(np.mean(scores))
            mean_ppl = float(np.mean(perplexities))
            
            # Quality Score: Reward Score, Penalize Log(PPL)
            # We use a softer penalty: Score / log10(PPL)
            # If Score=0.9, PPL=10 -> 0.9 / 1.0 = 0.9
            # If Score=0.9, PPL=100 -> 0.9 / 2.0 = 0.45
            q_score = mean_score / np.log10(max(mean_ppl, 10.0)) 
            
            results["data"].append({
                "coefficient": float(coef),
                "mean_score": mean_score,
                "std_score": float(np.std(scores)),
                "mean_perplexity": mean_ppl,
                "quality_score": q_score
            })
        
        # 3. Find Optimal (Dynamic Thresholding)
        # Filter out "Exploded" models
        valid_candidates = [r for r in results["data"] if r["mean_perplexity"] < ppl_threshold]
        
        if not valid_candidates:
            # If everything exploded, pick the one with lowest PPL
            print("  WARNING: All coefficients exceeded PPL threshold. Picking safest.")
            optimal = min(results["data"], key=lambda x: x["mean_perplexity"])
        else:
            # Pick highest Quality Score
            optimal = max(valid_candidates, key=lambda x: x["quality_score"])
            
        results["optimal_coefficient"] = optimal["coefficient"]
        results["optimal_score"] = optimal["mean_score"]
        
        return results


class CoefficientVisualizer:
    def __init__(self, output_dir: Path):
        self.figures_dir = output_dir / "figures" / "coefficients"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_score_vs_coefficient(self, results: Dict, save_suffix: str = ""):
        data = results["data"]
        concept = results["concept"]
        
        coefs = [d["coefficient"] for d in data]
        scores = [d["mean_score"] for d in data]
        stds = [d["std_score"] for d in data]
        ppls = [d["mean_perplexity"] for d in data]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Score (Left Axis)
        color = 'tab:blue'
        ax1.set_xlabel('Coefficient')
        ax1.set_ylabel('Concept Score', color=color, fontweight='bold')
        ax1.plot(coefs, scores, color=color, marker='o', label="Score")
        ax1.fill_between(coefs, np.array(scores)-np.array(stds), np.array(scores)+np.array(stds), color=color, alpha=0.2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-0.05, 1.05)
        
        # Plot PPL (Right Axis)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Perplexity (Log Scale)', color=color, fontweight='bold')
        ax2.plot(coefs, ppls, color=color, linestyle='--', alpha=0.5, label="Perplexity")
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Mark Optimal
        opt = results.get("optimal_coefficient", 0)
        ax1.axvline(opt, color='green', linestyle=':', linewidth=2, label=f"Optimal ({opt})")
        
        plt.title(f"Steering Dynamics: {concept}")
        fig.tight_layout()
        plt.savefig(self.figures_dir / f"coef_{concept}{save_suffix}.png")
        plt.close()


def convert_to_native(obj):
    if isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_native(i) for i in obj]
    if isinstance(obj, (np.int64, np.int32)): return int(obj)
    if isinstance(obj, (np.float64, np.float32)): return float(obj)
    return obj


def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/coefficient_analysis")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    week1_dir = Path(args.week1_dir)
    
    # Load optimal layers
    with open(week1_dir / "optimal_layers.json") as f:
        optimal_layers = json.load(f)
    
    concepts = args.concepts or list(optimal_layers.keys())
    
    # Load Model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load Vectors
    print("Loading vectors...")
    steering_vectors = {}
    for c in concepts:
        # Robust loading: try multiple naming conventions
        path1 = week1_dir / "vectors" / f"{c}_layer{optimal_layers[c]}.pt"
        path2 = week1_dir / "vectors" / f"{c}_layer_{optimal_layers[c]}.pt"
        path = path1 if path1.exists() else path2
        
        if path.exists():
            steering_vectors[c] = torch.load(path, map_location=model.device)
            
    # Filter concepts
    concepts = list(steering_vectors.keys())
    print(f"Analyzing {len(concepts)} concepts: {concepts}")
    
    prompts = get_test_prompts()
    if args.quick:
        prompts = prompts[:2]
        n_gen = 2
    else:
        prompts = prompts[:5]
        n_gen = 3
        
    analyzer = CoefficientAnalyzer(model, tokenizer, steering_vectors, optimal_layers, prompts)
    visualizer = CoefficientVisualizer(output_dir)
    
    all_results = {}
    
    # --- RUN ANALYSIS ---
    # Sweep from -2.0 to +3.0 to catch everything
    fine_range = np.arange(-2.0, 3.1, 0.25).tolist()
    
    for concept in concepts:
        res = analyzer.analyze_coefficient_range(concept, fine_range, n_gen)
        visualizer.plot_score_vs_coefficient(res)
        all_results[concept] = res

    # --- SUMMARY STATISTICS ---
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"{'CONCEPT':<15} | {'OPT':<6} | {'OPT SCORE':<10} | {'1.0 SCORE':<10} | {'GAIN':<10}")
    print("-" * 70)
    
    summary_data = []
    
    for concept, res in all_results.items():
        data = res["data"]
        
        # 1. Optimal
        opt_coef = res["optimal_coefficient"]
        opt_score = res["optimal_score"]
        
        # 2. Scaled (1.0) - Find closest
        scaled_res = min(data, key=lambda x: abs(x["coefficient"] - 1.0))
        scaled_score = scaled_res["mean_score"]
        
        # 3. Gain
        gain = opt_score - scaled_score
        
        print(f"{concept:<15} | {opt_coef:<6.2f} | {opt_score:<10.3f} | {scaled_score:<10.3f} | {gain:<+10.3f}")
        
        summary_data.append({
            "concept": concept,
            "optimal_coefficient": opt_coef,
            "gain": gain
        })

    with open(output_dir / "coefficient_results.json", "w") as f:
        json.dump(convert_to_native(all_results), f, indent=2)

if __name__ == "__main__":
    main()