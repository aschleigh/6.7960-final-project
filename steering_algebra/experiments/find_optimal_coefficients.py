"""
Coefficient Sweep Experiment: Determine Optimal Steering Strength.

Goal: Find the coefficient that maximizes the target concept score
      WITHOUT destroying text quality (perplexity).

Fixed Layer: 16 (as requested)
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib
# Force headless backend for servers (prevents display errors)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import SteeringConfig, generate_with_steering
from evaluation.classifiers import AttributeClassifier
from evaluation.metrics import QualityMetrics
from extraction.extract_vectors import load_cached_vectors

def convert_to_native(obj):
    """
    Recursively convert numpy types to native Python types.
    This prevents JSON serialization errors (Circular reference / TypeError).
    """
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float_)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native(obj.tolist())
    return obj

def sweep_coefficients(
    model, tokenizer, vector, concept, layer, prompts, coefficients
):
    """
    Test a range of coefficients for a single concept.
    Returns: list of results dicts.
    """
    classifier = AttributeClassifier(concept)
    metrics_calc = QualityMetrics(device=model.device.type)
    
    results = []
    
    print(f"\nSweeping {concept} (Layer {layer})...")
    
    # 1. Get Baseline (Coef=0)
    baseline_texts = []
    for p in prompts:
        baseline_texts.append(generate_with_steering(model, tokenizer, p, []))
    
    # Use list comprehension for scoring (Robustness fix)
    base_scores = [classifier.score(t) for t in baseline_texts]
    base_score = np.mean(base_scores)
    
    # Handle infinite perplexity in baseline gracefully
    base_eval = metrics_calc.evaluate(baseline_texts)
    base_ppl = base_eval["perplexity_mean"] if base_eval["perplexity_mean"] < 1000 else 100.0
    
    print(f"  Baseline: Score={base_score:.2f}, PPL={base_ppl:.1f}")
    
    # 2. Test Coefficients
    for coef in coefficients:
        config = SteeringConfig(vector=vector, layer=layer, coefficient=coef)
        texts = []
        
        for p in prompts:
            # Generate 1 sample per prompt per coef
            text = generate_with_steering(model, tokenizer, p, config)
            texts.append(text)
        
        # Evaluate
        current_scores = [classifier.score(t) for t in texts]
        avg_score = np.mean(current_scores)
        
        quality = metrics_calc.evaluate(texts)
        avg_ppl = quality["perplexity_mean"]
        
        # Validity Heuristic:
        # PPL shouldn't be > 100 absolute, or > 5x baseline relative
        threshold = max(100, base_ppl * 5)
        valid = avg_ppl < threshold
        
        results.append({
            "coefficient": coef,
            "score": avg_score,
            "score_gain": avg_score - base_score,
            "perplexity": avg_ppl,
            "is_valid": valid
        })
        
        status = "OK" if valid else "BAD"
        print(f"  Coef {coef:.1f}: Score={avg_score:.2f} (PPL={avg_ppl:.1f}) [{status}]")
        
    return results

def plot_sweep(concept, results, save_path):
    """Plot Score vs. Perplexity tradeoff for a single concept."""
    coefs = [r["coefficient"] for r in results]
    scores = [r["score"] for r in results]
    ppls = [r["perplexity"] for r in results]
    
    # Setup Figure with Twin Axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title(f"Coefficient Sweep: {concept} (Layer 16)")
    
    # Axis 1: Concept Score (Blue)
    color_score = 'tab:blue'
    ax1.set_xlabel('Steering Coefficient')
    ax1.set_ylabel('Concept Classifier Score', color=color_score, fontsize=12)
    ax1.plot(coefs, scores, color=color_score, marker='o', linewidth=2, label="Score")
    ax1.tick_params(axis='y', labelcolor=color_score)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Axis 2: Perplexity (Red)
    ax2 = ax1.twinx()
    color_ppl = 'tab:red'
    ax2.set_ylabel('Perplexity (Lower is Better)', color=color_ppl, fontsize=12)
    ax2.plot(coefs, ppls, color=color_ppl, linestyle='--', marker='x', linewidth=2, label="Perplexity")
    ax2.tick_params(axis='y', labelcolor=color_ppl)
    
    # Cap PPL view so the graph is readable even if PPL explodes
    max_plot_ppl = 200 
    # But if data stays low, zoom in
    if max(ppls) < 100: max_plot_ppl = 100
    ax2.set_ylim(0, max_plot_ppl)
    
    # Annotate "Optimal" Region
    valid_scores = [r["score"] for r in results if r["is_valid"]]
    if valid_scores:
        best_score = max(valid_scores)
        # Find index of best valid score
        best_result = max([r for r in results if r["is_valid"]], key=lambda x: x["score"])
        best_coef = best_result["coefficient"]
        
        ax1.axvline(best_coef, color='green', linestyle=':', alpha=0.8, label="Optimal")
        ax1.text(best_coef, 0.05, f" Optimal ({best_coef})", color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Saved to {save_path}")

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--output_dir", default="outputs/coefficient_sweep")
    parser.add_argument("--layer", type=int, default=16, help="Target layer (default 16)")
    # Sweep range
    parser.add_argument("--min_c", type=float, default=0.5)
    parser.add_argument("--max_c", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=0.5)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load resources
    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load vectors
    concepts = [c for c in cfg.model.concept_coefficients]
        
    vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    
    # Filter for target layer
    vectors = {}
    for c, layer_map in vectors_by_layer.items():
        if args.layer in layer_map:
            vectors[c] = layer_map[args.layer].to(model.device)
        else:
            print(f"Warning: No vector for {c} at layer {args.layer}")

    prompts = get_test_prompts()[:5] # Test on 5 prompts
    coefficients = np.arange(args.min_c, args.max_c + 0.01, args.step).tolist()
    
    optimal_coefs = {}
    all_results = {}
    
    for concept, vec in vectors.items():
        sweep_data = sweep_coefficients(
            model, tokenizer, vec, concept, args.layer, prompts, coefficients
        )
        all_results[concept] = sweep_data
        
        # Determine Optimal: Max Score where PPL is "valid"
        valid_runs = [r for r in sweep_data if r["is_valid"]]
        if valid_runs:
            best_run = max(valid_runs, key=lambda x: x["score"])
            optimal_coefs[concept] = best_run["coefficient"]
            print(f"-> Selected Optimal: {best_run['coefficient']}")
        else:
            print("-> Warning: All coefficients degraded quality too much. Defaulting to 1.0")
            optimal_coefs[concept] = 1.0
            
        # Plot Individual Concept
        plot_sweep(concept, sweep_data, output_dir / f"{concept}_layer{args.layer}_sweep.png")

    # Save Results
    with open(output_dir / "optimal_coefficients.json", "w") as f:
        # Convert optimal_coefs (just in case they are numpy floats)
        json.dump(convert_to_native(optimal_coefs), f, indent=2)
        
    with open(output_dir / "full_sweep_data.json", "w") as f:
        # Recursively clean the entire dictionary BEFORE dumping
        # This is safer than relying on 'default' callback
        clean_results = convert_to_native(all_results)
        json.dump(clean_results, f, indent=2)

    print(f"\nDone! Optimal coefficients saved to {output_dir / 'optimal_coefficients.json'}")

if __name__ == "__main__":
    main()