"""
Interference Experiment: Does Semantic Overlap Predict Composition Success?

Hypothesis:
    Geometric orthogonality (cosine similarity) is a poor predictor of success.
    True failure is caused by "Resource Interference":
    1. Logit Interference: Vectors trying to boost the same tokens.
    2. Neuron Interference: Vectors using the same active neurons.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
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
from evaluation.geometry import compute_cosine_similarity
from extraction.extract_vectors import load_cached_vectors


# ==========================================
# 1. INTERFERENCE METRICS (The "Why")
# ==========================================

def get_top_tokens_and_neurons(
    vector: torch.Tensor, 
    unembedding_matrix: torch.Tensor, 
    k_tokens: int = 20,
    k_neurons_percent: float = 0.05
):
    """
    Project vector to vocab space (Logit Lens) and identify active neurons.
    Handles device/dtype mismatches automatically.
    """
    # FIX: Ensure tensors are on the same device AND dtype before multiplication
    target_device = unembedding_matrix.device
    target_dtype = unembedding_matrix.dtype
    
    # Move vector to the matrix's device/type (vector is smaller, cheaper to move)
    if vector.device != target_device or vector.dtype != target_dtype:
        vector = vector.to(device=target_device, dtype=target_dtype)

    # 1. Logit Lens: Project to vocabulary
    # vector: [d_model] @ W_U: [d_model, vocab_size] -> logits: [vocab_size]
    logits = vector @ unembedding_matrix
    top_token_indices = torch.topk(logits, k_tokens).indices.cpu().numpy().tolist()
    
    # 2. Active Neurons: Top k% magnitude in the vector itself
    k_neurons = int(vector.shape[0] * k_neurons_percent)
    top_neuron_indices = torch.topk(torch.abs(vector), k_neurons).indices.cpu().numpy().tolist()
    
    return set(top_token_indices), set(top_neuron_indices)

def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard Index (Intersection / Union)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def compute_pairwise_metrics(
    steering_vectors: Dict[str, torch.Tensor],
    model
) -> List[Dict]:
    """Compute Cosine, Logit Overlap, and Neuron Overlap for all pairs."""
    concepts = list(steering_vectors.keys())
    pairs_data = []
    
    # Get Unembedding Matrix (Handle different architectures)
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach()
        # Transpose if necessary to match shapes
        if W_U.shape[0] != steering_vectors[concepts[0]].shape[0]:
            W_U = W_U.T
    elif hasattr(model, "get_output_embeddings"):
        W_U = model.get_output_embeddings().weight.detach().T
    else:
        print("Warning: Could not find Unembedding Matrix. Skipping Logit Lens.")
        W_U = None

    print("Pre-computing interference metrics (Logit Lens)...")
    
    # Pre-compute features for all concepts
    concept_features = {}
    for c in concepts:
        if W_U is not None:
            tokens, neurons = get_top_tokens_and_neurons(steering_vectors[c], W_U)
        else:
            tokens, neurons = set(), set()
        concept_features[c] = {"tokens": tokens, "neurons": neurons}

    # Compare all pairs
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i < j: # Upper triangle only
                vec_a = steering_vectors[c1]
                vec_b = steering_vectors[c2]
                
                # Geometry (Cosine)
                cos_sim = float(compute_cosine_similarity(vec_a, vec_b))
                
                # Semantics (Logit Lens)
                logit_overlap = compute_jaccard_similarity(
                    concept_features[c1]["tokens"], 
                    concept_features[c2]["tokens"]
                )
                
                # Mechanism (Neuron Overlap)
                neuron_overlap = compute_jaccard_similarity(
                    concept_features[c1]["neurons"], 
                    concept_features[c2]["neurons"]
                )
                
                pairs_data.append({
                    "concept_a": c1,
                    "concept_b": c2,
                    "cosine_similarity": cos_sim,
                    "abs_cosine_similarity": abs(cos_sim),
                    "logit_overlap": logit_overlap,
                    "neuron_overlap": neuron_overlap
                })
    
    return pairs_data

# ==========================================
# 2. STEERING LOOP (The "Test")
# ==========================================

def test_composition_with_interference(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    pairs_data: List[Dict],
    layer: int,
    prompts: List[str],
    n_generations: int = 5
) -> List[Dict]:
    
    evaluator_cache = {} 
    results = []
    
    # Filter: Remove trivial synonyms (>0.8 similarity)
    # These create false "success" because the concepts are identical
    filtered_pairs = [p for p in pairs_data if p["abs_cosine_similarity"] < 0.8]
    print(f"\nTesting {len(filtered_pairs)} pairs (Filtered out {len(pairs_data) - len(filtered_pairs)} synonyms)")
    
    for pair in tqdm(filtered_pairs, desc="Steering"):
        c1, c2 = pair["concept_a"], pair["concept_b"]
        
        # Lazy load evaluator & FORCE CPU to avoid OOM
        pair_key = tuple(sorted([c1, c2]))
        if pair_key not in evaluator_cache:
            try:
                # Force CPU here saves ~2GB VRAM per evaluator
                evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2], device="cpu")
            except TypeError:
                print(f"Warning: Evaluator for {c1}/{c2} does not accept 'device'. Using default.")
                evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2])

        evaluator = evaluator_cache[pair_key]
        
        vec_a = steering_vectors[c1]
        vec_b = steering_vectors[c2]
        
        success_count = 0
        total = 0
        
        for prompt in prompts:
            for _ in range(n_generations):
                # Apply both vectors
                config = [
                    SteeringConfig(vector=vec_a, layer=layer, coefficient=1.0),
                    SteeringConfig(vector=vec_b, layer=layer, coefficient=1.0)
                ]
                text = generate_with_steering(model, tokenizer, prompt, config)
                
                # Check success (both concepts present)
                scores = evaluator.evaluate(text, [c1, c2])
                if scores[c1] > 0.5 and scores[c2] > 0.5:
                    success_count += 1
                total += 1
        
        pair["joint_success_rate"] = success_count / total
        results.append(pair)
        
    return results

# ==========================================
# 3. VISUALIZATION & SUMMARY
# ==========================================

def print_summary_stats(results: List[Dict]):
    """Print clear correlation statistics to the console."""
    success = [r["joint_success_rate"] for r in results]
    cosine = [r["abs_cosine_similarity"] for r in results]
    logit = [r["logit_overlap"] for r in results]
    
    # Calculate Correlations
    r_cosine, _ = pearsonr(cosine, success)
    r_logit, _ = pearsonr(logit, success)
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total Pairs Tested: {len(results)}")
    print(f"Avg Success Rate:   {np.mean(success):.1%}")
    print("-" * 50)
    print(f"Correlation (Cosine Sim vs Success): r = {r_cosine:.3f}")
    print(f"Correlation (Logit Overlap vs Success): r = {r_logit:.3f}")
    print("-" * 50)
    
    if abs(r_logit) > abs(r_cosine):
        print(">> HYPOTHESIS CONFIRMED: Logit Overlap predicts failure better than Geometry.")
    else:
        print(">> RESULT: Geometry remains a stronger predictor.")
    print("="*50 + "\n")


def plot_results(results: List[Dict], save_dir: Path):
    """Generate clear regression plots and bar charts."""
    sns.set_theme(style="whitegrid")
    
    # Data Prep
    data = {
        "Success Rate": [r["joint_success_rate"] for r in results],
        "Abs Cosine Sim": [r["abs_cosine_similarity"] for r in results],
        "Logit Overlap": [r["logit_overlap"] for r in results],
        "Neuron Overlap": [r["neuron_overlap"] for r in results]
    }
    
    # 1. Regression Plot (The Trend)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Geometry
    sns.regplot(x="Abs Cosine Sim", y="Success Rate", data=data, ax=axes[0], 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_title("Geometry: Cosine Similarity")
    axes[0].set_ylim(-0.1, 1.1)

    # Plot B: Logit Overlap
    sns.regplot(x="Logit Overlap", y="Success Rate", data=data, ax=axes[1], 
                scatter_kws={'alpha':0.5, 'color':'orange'}, line_kws={'color':'red'})
    axes[1].set_title("Semantics: Logit Overlap")
    axes[1].set_ylim(-0.1, 1.1)

    # Plot C: Neuron Overlap
    sns.regplot(x="Neuron Overlap", y="Success Rate", data=data, ax=axes[2], 
                scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'red'})
    axes[2].set_title("Mechanism: Neuron Overlap")
    axes[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(save_dir / "interference_trends.png")
    
    # 2. Binned Bar Chart (Easier to read)
    plt.figure(figsize=(8, 6))
    
    # Create simple bins for Logit Overlap
    overlaps = np.array(data["Logit Overlap"])
    successes = np.array(data["Success Rate"])
    
    low_mask = overlaps < np.percentile(overlaps, 33)
    high_mask = overlaps > np.percentile(overlaps, 66)
    med_mask = (~low_mask) & (~high_mask)
    
    categories = ["Low Overlap", "Med Overlap", "High Overlap"]
    means = [successes[low_mask].mean(), successes[med_mask].mean(), successes[high_mask].mean()]
    
    # Clean up NaNs if a bin is empty
    means = [m if not np.isnan(m) else 0.0 for m in means]
    
    sns.barplot(x=categories, y=means, palette="viridis")
    plt.title("Composition Success by Logit Overlap Level")
    plt.ylabel("Mean Success Rate")
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_dir / "interference_summary_bar.png")
    
    print(f"Saved plots to {save_dir}")

def convert_to_native(obj):
    """Helper to convert numpy types for JSON serialization."""
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
    else:
        return obj

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--output_dir", default="outputs/interference_analysis")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model
    print("Loading model resources...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
    # 2. Load Cached Vectors
    print("Loading vectors...")
    with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
        concepts = json.load(f)["concepts"]
        
    vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    
    # Move vectors to model device for processing
    vectors = {c: v[cfg.model.default_layer].to(model.device) for c, v in vectors_by_layer.items()}
    
    # 3. Compute Metrics (Logit Lens)
    pairs_data = compute_pairwise_metrics(vectors, model)
    
    # 4. Run Steering Tests
    prompts = get_test_prompts()[:2] if args.quick else get_test_prompts()[:5]
    n_gen = 2 if args.quick else 3
    
    results = test_composition_with_interference(
        model, tokenizer, vectors, pairs_data, 
        cfg.model.default_layer, prompts, n_generations=n_gen
    )
    
    # 5. Analyze & Save
    print_summary_stats(results)
    plot_results(results, output_dir)
    
    with open(output_dir / "interference_data.json", "w") as f:
        json.dump(convert_to_native(results), f, indent=2)

if __name__ == "__main__":
    main()