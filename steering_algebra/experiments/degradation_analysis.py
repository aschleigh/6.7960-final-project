"""
Degradation Analysis: Measuring Destructive Interference directly.

Hypothesis:
    Instead of checking if A and B *both* work (Joint Success), 
    we check how much B *hurts* A (Degradation).
    
    Metric: Score_Drop = Score(A | A_only) - Score(A | A + B)
    Prediction: High Logit Overlap -> Large Score Drop.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import SteeringConfig, generate_with_steering
from evaluation.classifiers import MultiAttributeEvaluator
from extraction.extract_vectors import load_cached_vectors


def compute_logit_overlap(vec_a, vec_b, W_U, top_k=500):
    """Compute overlap of top tokens (Resource Contention)."""
    target_device = W_U.device
    target_dtype = W_U.dtype
    va = vec_a.to(device=target_device, dtype=target_dtype)
    vb = vec_b.to(device=target_device, dtype=target_dtype)
    
    # Project to logits
    logits_a = va @ W_U
    logits_b = vb @ W_U
    
    # Identify top indices (mask)
    top_a = torch.topk(logits_a.abs(), top_k).indices
    top_b = torch.topk(logits_b.abs(), top_k).indices
    
    # Jaccard Sim of indices
    set_a = set(top_a.cpu().tolist())
    set_b = set(top_b.cpu().tolist())
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    return intersection / union if union > 0 else 0.0


def run_degradation_test(
    model, tokenizer, steering_vectors, concepts, W_U,
    layer, prompts, n_gen=1, coef=2.0
) -> List[Dict]:
    
    results = []
    evaluator_cache = {}
    
    # Generate list of pairs
    pairs = []
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i != j: # Test A vs B AND B vs A (Asymmetric)
                pairs.append((c1, c2))
    
    print(f"Testing degradation on {len(pairs)} directed pairs...")
    
    for concept_target, concept_interferer in tqdm(pairs):
        
        # 1. Setup Evaluator for TARGET only
        if concept_target not in evaluator_cache:
            evaluator_cache[concept_target] = MultiAttributeEvaluator([concept_target], device="cpu")
        evaluator = evaluator_cache[concept_target]
        
        vec_target = steering_vectors[concept_target]
        vec_interferer = steering_vectors[concept_interferer]
        
        # 2. Compute Overlap (The Predictor)
        overlap = compute_logit_overlap(vec_target, vec_interferer, W_U)
        
        # 3. Run Baselines vs Interference
        scores_baseline = []
        scores_interfered = []
        
        for prompt in prompts:
            for _ in range(n_gen):
                # A. Baseline: Just Target
                config_base = SteeringConfig(vector=vec_target, layer=layer, coefficient=coef)
                text_base = generate_with_steering(model, tokenizer, prompt, config_base)
                scores_baseline.append(evaluator.evaluate(text_base, [concept_target])[concept_target])
                
                # B. Interference: Target + Interferer
                config_int = [
                    SteeringConfig(vector=vec_target, layer=layer, coefficient=coef),
                    SteeringConfig(vector=vec_interferer, layer=layer, coefficient=coef)
                ]
                text_int = generate_with_steering(model, tokenizer, prompt, config_int)
                scores_interfered.append(evaluator.evaluate(text_int, [concept_target])[concept_target])
        
        # 4. Calculate Drop
        mean_base = np.mean(scores_baseline)
        mean_int = np.mean(scores_interfered)
        score_drop = mean_base - mean_int # Positive = Degradation
        
        results.append({
            "target": concept_target,
            "interferer": concept_interferer,
            "logit_overlap": overlap,
            "baseline_score": mean_base,
            "interfered_score": mean_int,
            "score_drop": score_drop
        })
        
    return results


def analyze_degradation(results: List[Dict], save_dir: Path):
    df = pd.DataFrame(results)
    
    # Filter out cases where baseline was already bad (can't drop if it's 0)
    df_clean = df[df["baseline_score"] > 0.3]
    
    if len(df_clean) < 5:
        print("Not enough successful baselines to analyze.")
        return

    r, p = pearsonr(df_clean["logit_overlap"], df_clean["score_drop"])
    
    print("\n" + "="*60)
    print("DEGRADATION ANALYSIS")
    print("="*60)
    print(f"Pairs Analyzed: {len(df_clean)}")
    print(f"Correlation (Overlap vs. Drop): r = {r:.3f} (p={p:.3f})")
    print("-" * 60)
    
    if r > 0.3:
        print(">> RESULT: Strong positive trend. Overlap causes degradation!")
    else:
        print(">> RESULT: Trend is weak.")
    print("="*60)
    
    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.regplot(x="logit_overlap", y="score_drop", data=df_clean, 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    plt.title(f"Resource Contention: Overlap vs. Performance Drop\n(r={r:.2f})")
    plt.xlabel("Logit Overlap (Jaccard Index of Top-500 Tokens)")
    plt.ylabel("Performance Drop (Baseline - Interfered)")
    plt.ylim(-0.2, 1.0) # Drop can be max 1.0
    
    plt.savefig(save_dir / "degradation_trend.png")
    print(f"Saved plot to {save_dir}")

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
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--output_dir", default="outputs/degradation_analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--coef", type=float, default=2.0)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
    # Get Head
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach()
    elif hasattr(model, "get_output_embeddings"):
        W_U = model.get_output_embeddings().weight.detach()
    else:
        raise ValueError("Head not found")
        
    # Check head shape
    # We need [hidden, vocab] for vec @ W
    # Usually weight is [vocab, hidden], so we transpose
    if W_U.shape[0] != 4096: # Assuming 7B model hidden dim
         W_U = W_U.T

    with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
        concepts = json.load(f)["concepts"]
        
    vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    vectors = {c: v[cfg.model.default_layer].to(model.device) for c, v in vectors_by_layer.items()}
    
    prompts = get_test_prompts()[:2] if args.quick else get_test_prompts()[:5]
    n_gen = 2 if args.quick else 3
    
    results = run_degradation_test(
        model, tokenizer, vectors, concepts, W_U,
        cfg.model.default_layer, prompts, n_gen, args.coef
    )
    
    analyze_degradation(results, output_dir)
    
    with open(output_dir / "degradation_results.json", "w") as f:
        json.dump(convert_to_native(results), f, indent=2)

if __name__ == "__main__":
    main()