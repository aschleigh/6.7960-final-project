"""
Trajectory Collapse Analysis: MRI-Scanning the Interference.

Hypothesis:
    Composition failure is caused by "Convergent Evolution" in the model's depth.
    Vectors that start orthogonal (Layer 15) are rotated by MLPs until they 
    collide (collapse) into the same manifold in deeper layers (e.g. Layer 25).
    
    We track the Subspace Overlap at EVERY layer to find the "Crash Point".
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
# 1. DYNAMIC SUBSPACE EXTRACTION
# ==========================================

def get_layerwise_subspaces(
    model, tokenizer, concept, start_layer, n_layers, n_samples=25, k_components=10
):
    """
    Extracts the PCA basis for a concept at ALL layers [start_layer ... n_layers].
    
    Returns:
        Dict {layer_idx: Basis_Matrix[d_model, k]}
    """
    # Use positive prompts to define the "Concept Manifold"
    pairs = get_contrastive_pairs(concept, n_samples)
    prompts = [p[0] for p in pairs]
    
    hooks = ActivationHooks(model)
    trace_layers = list(range(start_layer, n_layers))
    
    # Temporary storage: {layer: [act_tensor, act_tensor...]}
    layer_acts_cpu = {l: [] for l in trace_layers}
    
    batch_size = 2 # Low batch size to prevent OOM
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Register hooks for ALL layers at once
        with hooks.extraction_context(trace_layers):
            with torch.no_grad():
                model(**inputs)
            
            # Harvest activations and immediately move to CPU
            seq_lens = inputs.attention_mask.sum(dim=1) - 1
            for l in trace_layers:
                # Shape: [batch, seq, hidden]
                acts = hooks.cache[f"layer_{l}_residual"]
                for b, length in enumerate(seq_lens):
                    # Store only the last token
                    layer_acts_cpu[l].append(acts[b, length, :].cpu().float())
        
        # Clear GPU cache after each batch
        hooks.clear_hooks()
        torch.cuda.empty_cache()

    # Compute PCA Basis for each layer
    layer_bases = {}
    print(f"  Computing PCA for {concept} across {len(trace_layers)} layers...")
    
    for l in trace_layers:
        # Stack -> [N, d]
        matrix = torch.stack(layer_acts_cpu[l])
        
        # Center the data
        centered = matrix - matrix.mean(dim=0)
        
        # SVD (on CPU is fine for this size, usually faster than moving back and forth)
        try:
            # U: [N, N], S: [N], V: [N, d] -> Transposed: [d, N]
            # We want Principal Components (Eigenvectors of Covariance), which are V
            # In torch.linalg.svd(X), X = U S Vh. Vh is [N, d]. Rows of Vh are components.
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            components = Vh.T # [d, N]
        except:
            # Fallback
            U, S, V = torch.svd(centered)
            components = V # [d, N]
            
        # Keep Top-K components
        # Matrix shape: [d_model, k]
        layer_bases[l] = components[:, :k_components].clone()
        
    return layer_bases


def compute_trajectory_overlap(bases_a, bases_b, layers):
    """
    Computes overlap profile across layers.
    Returns: 
        - max_overlap: The highest overlap found (The "Crash")
        - crash_layer: Which layer the crash happened
        - profile: List of overlap values
    """
    profile = []
    
    for l in layers:
        # Move tiny basis matrices to GPU for fast matmul
        Ba = bases_a[l].cuda() 
        Bb = bases_b[l].cuda()
        
        # Metric: Frobenius Norm of product (Energy of interaction)
        # Range [0, sqrt(k)]
        interaction = Ba.T @ Bb
        overlap = torch.norm(interaction, p="fro").item()
        profile.append(overlap)
        
    max_overlap = max(profile)
    crash_layer = layers[profile.index(max_overlap)]
    
    return max_overlap, crash_layer, profile


# ==========================================
# 2. EXPERIMENT LOOP
# ==========================================

def run_experiment(
    model, tokenizer, steering_vectors, concepts, start_layer, n_layers, n_gen=2
):
    results = []
    
    # 1. Extract Trajectories (Expensive Step)
    print(f"\n{'='*40}\nPHASE 1: Extracting Trajectories\n{'='*40}")
    concept_trajectories = {}
    
    for concept in tqdm(concepts, desc="Mapping Subspaces"):
        concept_trajectories[concept] = get_layerwise_subspaces(
            model, tokenizer, concept, 
            start_layer=start_layer, 
            n_layers=n_layers
        )

    # 2. Run Steering & Comparison
    print(f"\n{'='*40}\nPHASE 2: Steering & Collision Detection\n{'='*40}")
    
    pairs = []
    for i, c1 in enumerate(concepts):
        for j in range(i+1, len(concepts)):
            pairs.append((concepts[i], concepts[j]))
            
    evaluator_cache = {}
    prompts = get_test_prompts()[:2] # 2 prompts
    
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
                # Apply both vectors
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
        
        # --- B. Measure Trajectory Collapse ---
        max_overlap, crash_layer, profile = compute_trajectory_overlap(
            concept_trajectories[c1], 
            concept_trajectories[c2], 
            layers=list(range(start_layer, n_layers))
        )
        
        results.append({
            "concept_a": c1,
            "concept_b": c2,
            "success_rate": success_rate,
            "max_trajectory_overlap": max_overlap,
            "crash_layer": crash_layer,
            "overlap_profile": profile # Store full curve for plotting
        })
        
    return results


# ==========================================
# 3. ANALYSIS & PLOTTING
# ==========================================

def analyze_results(results: List[Dict], save_dir: Path, start_layer: int):
    df = pd.DataFrame(results)
    if len(df) < 5: return

    # Correlation
    r, p = pearsonr(df["max_trajectory_overlap"], df["success_rate"])
    
    print("\n" + "="*60)
    print("TRAJECTORY COLLAPSE SUMMARY")
    print("="*60)
    print(f"Metric: Max Subspace Overlap (Any Layer) vs Success")
    print(f"Correlation: r = {r:.3f} (p={p:.3f})")
    print("-" * 60)
    if r < -0.4:
        print(">> STRONG RESULT: Convergence predicts failure!")
    elif r < -0.2:
        print(">> MODERATE RESULT: Convergence matters, but is noisy.")
    else:
        print(">> NULL RESULT: No clear link found.")
    print("="*60)
    
    # Plot 1: Regression
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.regplot(x="max_trajectory_overlap", y="success_rate", data=df, line_kws={"color": "red"})
    plt.title(f"Max Trajectory Overlap vs. Success (r={r:.2f})")
    plt.xlabel("Max Subspace Overlap (Peak Convergence)")
    plt.ylabel("Joint Success Rate")
    plt.savefig(save_dir / "trajectory_correlation.png")
    
    # Plot 2: The Trajectories (The Cool Visual)
    plt.figure(figsize=(12, 7))
    
    # Get Top 3 Successes and Top 3 Failures
    df = df.sort_values("success_rate", ascending=False)
    top_3 = df.head(3)
    bot_3 = df.tail(3)
    
    layers = np.arange(start_layer, start_layer + len(df.iloc[0]["overlap_profile"]))
    
    for _, row in top_3.iterrows():
        plt.plot(layers, row["overlap_profile"], color='green', alpha=0.7, 
                 label=f"Success: {row['concept_a']}+{row['concept_b']}")
        
    for _, row in bot_3.iterrows():
        plt.plot(layers, row["overlap_profile"], color='red', alpha=0.7, 
                 label=f"Fail: {row['concept_a']}+{row['concept_b']}")
        
    plt.title("Trajectory Evolution: Do Concepts Converge?")
    plt.xlabel("Layer Index")
    plt.ylabel("Subspace Overlap (Frobenius)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_dir / "trajectory_profiles.png")
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
    parser.add_argument("--output_dir", default="outputs/trajectory_analysis")
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
    
    # Load Concepts
    with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
        concepts = json.load(f)["concepts"]
        
    # Load Vectors (for steering)
    vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    steering_vectors = {c: v[args.layer].to(model.device) for c, v in vectors_by_layer.items()}
    
    # Run
    results = run_experiment(
        model, tokenizer, steering_vectors, concepts, 
        start_layer=args.layer, n_layers=n_layers
    )
    
    # Analyze
    analyze_results(results, output_dir, args.layer)
    
    with open(output_dir / "trajectory_results.json", "w") as f:
        json.dump(convert_to_native(results), f, indent=2)

if __name__ == "__main__":
    main()