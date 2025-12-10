"""
Dynamic Active Subspace Analysis: The "Semantic Crash" Test.

Hypothesis:
    We combine Dynamic Trajectory Analysis (Layer-wise tracking) with 
    Active Subspace Analysis (High-K Logit Projection).
    
    We test if vectors 'converge' semantically as they propagate depth-wise.
    We sweep K values (1k, 5k, 10k, 20k) to find the "Long Tail" interference.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import gc

from config import cfg
from extraction.hooks import ActivationHooks
from data.contrastive_pairs import get_contrastive_pairs
from data.prompts import get_test_prompts
from evaluation.classifiers import MultiAttributeEvaluator
from steering.apply_steering import generate_with_steering, SteeringConfig
from extraction.extract_vectors import load_cached_vectors

# ==========================================
# 1. DYNAMIC SEMANTIC EXTRACTION
# ==========================================

def get_layerwise_mean_activations(
    model, tokenizer, concept, start_layer, n_layers, n_samples=30
):
    """
    Extracts the MEAN activation vector for a concept at ALL layers.
    Unlike PCA, this represents the "Center of Mass" of the concept trajectory.
    """
    pairs = get_contrastive_pairs(concept, n_samples)
    prompts = [p[0] for p in pairs]
    
    hooks = ActivationHooks(model)
    trace_layers = list(range(start_layer, n_layers))
    
    # Accumulators
    layer_sums = {l: torch.zeros(model.config.hidden_size).float() for l in trace_layers}
    layer_counts = {l: 0 for l in trace_layers}
    
    batch_size = 4
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with hooks.extraction_context(trace_layers):
            with torch.no_grad():
                model(**inputs)
            
            seq_lens = inputs.attention_mask.sum(dim=1) - 1
            for l in trace_layers:
                # [batch, seq, hidden]
                acts = hooks.cache[f"layer_{l}_residual"]
                for b, length in enumerate(seq_lens):
                    # Grab last token, move to CPU
                    vec = acts[b, length, :].cpu().float()
                    layer_sums[l] += vec
                    layer_counts[l] += 1
        
        hooks.clear_hooks()
        torch.cuda.empty_cache()

    # Compute Means
    layer_means = {}
    for l in trace_layers:
        if layer_counts[l] > 0:
            layer_means[l] = layer_sums[l] / layer_counts[l]
            
    return layer_means


def compute_dynamic_overlap_profile(
    traj_a, traj_b, layers, unembedding_matrix, k_sweep=[1000, 5000, 10000]
):
    """
    Computes the overlap profile across layers for multiple K values.
    Uses "Signed Centered Cosine" on the Active Subspace.
    
    Returns:
        Dict {k: { "max_overlap": float, "profile": List[float] }}
    """
    target_device = unembedding_matrix.device
    target_dtype = unembedding_matrix.dtype
    
    results = {k: {"profile": []} for k in k_sweep}
    
    for l in layers:
        # Move vectors to GPU for projection
        va = traj_a[l].to(device=target_device, dtype=target_dtype)
        vb = traj_b[l].to(device=target_device, dtype=target_dtype)
        
        # 1. Logit Lens Project
        logits_a = va @ unembedding_matrix
        logits_b = vb @ unembedding_matrix
        
        # 2. Center (Remove "The" Noise)
        logits_a = logits_a - logits_a.mean()
        logits_b = logits_b - logits_b.mean()
        
        # 3. Sweep K (Active Subspace)
        for k in k_sweep:
            # Identify Active Indices (Union of Top-K magnitude)
            idx_a = torch.topk(logits_a.abs(), k).indices
            idx_b = torch.topk(logits_b.abs(), k).indices
            mask = torch.unique(torch.cat([idx_a, idx_b]))
            
            # Slice
            filt_a = logits_a[mask]
            filt_b = logits_b[mask]
            
            # Signed Cosine
            sim = torch.nn.functional.cosine_similarity(
                filt_a.unsqueeze(0), 
                filt_b.unsqueeze(0)
            ).item()
            
            results[k]["profile"].append(sim)
            
    # Find Peaks
    for k in k_sweep:
        profile = results[k]["profile"]
        # We care about the MAXIMUM semantic alignment/conflict
        # (Using abs max to capture strong negative conflicts too)
        max_val = max(profile, key=abs) 
        results[k]["max_overlap"] = max_val
        results[k]["crash_layer"] = layers[profile.index(max_val)]
        
    return results


# ==========================================
# 2. EXPERIMENT LOOP
# ==========================================

def run_experiment(
    model, tokenizer, steering_vectors, concepts, start_layer, n_layers, 
    n_gen=2, k_sweep=[1000, 5000, 10000, 20000]
):
    # 1. Map Trajectories (The "Movie")
    print(f"\n{'='*40}\nPHASE 1: Mapping Dynamic Semantic Trajectories\n{'='*40}")
    trajectories = {}
    for concept in tqdm(concepts, desc="Extracting"):
        trajectories[concept] = get_layerwise_mean_activations(
            model, tokenizer, concept, 
            start_layer=start_layer, 
            n_layers=n_layers
        )

    # 2. Steering Test (The "Outcome")
    print(f"\n{'='*40}\nPHASE 2: Steering & Collision Detection\n{'='*40}")
    
    pairs = []
    for i, c1 in enumerate(concepts):
        for j in range(i+1, len(concepts)):
            pairs.append((concepts[i], concepts[j]))
            
    # Head for projection
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach()
        if W_U.shape[0] != model.config.hidden_size: W_U = W_U.T
    elif hasattr(model, "get_output_embeddings"):
        W_U = model.get_output_embeddings().weight.detach().T
    else:
        raise ValueError("Head not found")

    evaluator_cache = {}
    prompts = get_test_prompts()[:2]
    experiment_data = []
    
    # Need to verify layers list matches extraction
    layers_to_scan = list(range(start_layer, n_layers))

    for c1, c2 in tqdm(pairs, desc="Testing Pairs"):
        # --- A. Measure Success ---
        pair_key = tuple(sorted([c1, c2]))
        if pair_key not in evaluator_cache:
            try:
                evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2], device="cpu")
            except:
                evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2])
        
        vec_a = steering_vectors[c1]
        vec_b = steering_vectors[c2]
        
        success_count = 0
        total = 0
        
        for prompt in prompts:
            for _ in range(n_gen):
                config = [
                    SteeringConfig(vector=vec_a, layer=start_layer, coefficient=1.5),
                    SteeringConfig(vector=vec_b, layer=start_layer, coefficient=1.5)
                ]
                text = generate_with_steering(model, tokenizer, prompt, config)
                scores = evaluator_cache[pair_key].evaluate(text, [c1, c2])
                
                if scores[c1] > 0.5 and scores[c2] > 0.5:
                    success_count += 1
                total += 1
        
        success_rate = success_count / total
        
        # --- B. Measure Dynamic Overlap ---
        overlap_data = compute_dynamic_overlap_profile(
            trajectories[c1], trajectories[c2], layers_to_scan, W_U, k_sweep
        )
        
        # Flatten for DataFrame
        row = {
            "concept_a": c1, "concept_b": c2, "success_rate": success_rate
        }
        for k in k_sweep:
            row[f"max_overlap_k{k}"] = overlap_data[k]["max_overlap"]
            row[f"profile_k{k}"] = overlap_data[k]["profile"]
            
        experiment_data.append(row)
        
    return experiment_data


# ==========================================
# 3. ANALYSIS
# ==========================================

def analyze_results(results: List[Dict], save_dir: Path, start_layer: int):
    df = pd.DataFrame(results)
    if len(df) < 5: return

    print("\n" + "="*60)
    print("DYNAMIC ACTIVE SUBSPACE SUMMARY")
    print("="*60)
    
    # Check correlations for each K
    best_k = None
    best_r = 0
    
    cols = [c for c in df.columns if "max_overlap_k" in c]
    for col in cols:
        k_val = col.split("k")[-1]
        r, p = pearsonr(df[col], df["success_rate"])
        print(f"K={k_val:<6} | Correlation (r): {r:.3f} (p={p:.3f})")
        
        if abs(r) > abs(best_r):
            best_r = r
            best_k = col
            
    print("-" * 60)
    print(f">> BEST PREDICTOR: {best_k} (r={best_r:.3f})")
    
    if abs(best_r) > 0.3:
        print(">> RESULT: Clear trend found! Dynamic convergence predicts outcome.")
    else:
        print(">> RESULT: Still noisy. The relationship is elusive.")
    print("="*60)
    
    # Plot 1: Best K Correlation
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.regplot(x=best_k, y="success_rate", data=df, line_kws={"color": "red"})
    plt.title(f"Dynamic Convergence ({best_k}) vs Success")
    plt.ylabel("Joint Success Rate")
    plt.xlabel("Max Signed Cosine (Active Subspace)")
    plt.savefig(save_dir / "dynamic_correlation.png")
    
    # Plot 2: Evolution Profiles (The "Movie")
    # Show how the overlap changes per layer for Success vs Failure
    plt.figure(figsize=(12, 7))
    df_sorted = df.sort_values("success_rate", ascending=False)
    
    # Use the profile from the Best K
    profile_col = best_k.replace("max_overlap", "profile")
    layers = np.arange(start_layer, start_layer + len(df.iloc[0][profile_col]))
    
    # Plot Top 3 Successes (Green)
    for _, row in df_sorted.head(3).iterrows():
        plt.plot(layers, row[profile_col], color='green', alpha=0.6, linewidth=2,
                 label=f"Success ({row['success_rate']:.1f}): {row['concept_a']}+{row['concept_b']}")
        
    # Plot Bottom 3 Failures (Red)
    for _, row in df_sorted.tail(3).iterrows():
        plt.plot(layers, row[profile_col], color='red', alpha=0.6, linewidth=2,
                 label=f"Fail ({row['success_rate']:.1f}): {row['concept_a']}+{row['concept_b']}")
        
    plt.title(f"Semantic Trajectory Evolution (K={best_k.split('k')[-1]})")
    plt.xlabel("Model Layer")
    plt.ylabel("Active Subspace Overlap (Signed Cosine)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / "trajectory_evolution.png")
    print(f"Saved plots to {save_dir}")

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
    parser.add_argument("--output_dir", default="outputs/dynamic_active_subspace")
    parser.add_argument("--layer", type=int, default=cfg.model.default_layer)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        n_layers = len(model.model.layers)
    
    with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
        concepts = json.load(f)["concepts"]
        
    vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    steering_vectors = {c: v[args.layer].to(model.device) for c, v in vectors_by_layer.items()}
    
    # Run Experiment
    results = run_experiment(
        model, tokenizer, steering_vectors, concepts, 
        start_layer=args.layer, n_layers=n_layers
    )
    
    # Analyze
    analyze_results(results, output_dir, args.layer)
    
    with open(output_dir / "dynamic_results.json", "w") as f:
        json.dump(convert_to_native(results), f, indent=2)

if __name__ == "__main__":
    main()