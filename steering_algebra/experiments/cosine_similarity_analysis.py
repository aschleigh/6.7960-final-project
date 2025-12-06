"""
Cosine Similarity Experiment: Does Vector Geometry Predict Composition Success?

Central Hypothesis:
    Orthogonal vectors (low cosine similarity) should compose better than
    aligned or opposing vectors (high absolute cosine similarity).

Tests:
1. Generate with A+B for pairs across similarity spectrum
2. Measure joint success rate (both A and B present)
3. Analyze correlation between |cos(A,B)| and success
4. Identify optimal similarity range for composition
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import (
    SteeringConfig,
    generate_with_steering
)
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics
from evaluation.geometry import compute_cosine_similarity
from extraction.extract_vectors import load_cached_vectors


def compute_all_pairwise_similarities(
    steering_vectors: Dict[str, torch.Tensor]
) -> List[Tuple[str, str, float]]:
    """
    Compute cosine similarity for all concept pairs.
    
    Returns:
        List of (concept_a, concept_b, similarity) sorted by absolute similarity
    """
    concepts = list(steering_vectors.keys())
    pairs = []
    
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i < j:  # Only upper triangle
                sim = compute_cosine_similarity(
                    steering_vectors[c1],
                    steering_vectors[c2]
                )
                pairs.append((c1, c2, float(sim)))
    
    return pairs


def categorize_by_similarity(
    pairs: List[Tuple[str, str, float]]
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Categorize pairs by their cosine similarity.
    
    Categories:
    - Highly orthogonal: |cos| < 0.1
    - Orthogonal: 0.1 <= |cos| < 0.3
    - Weakly related: 0.3 <= |cos| < 0.5
    - Related: 0.5 <= |cos| < 0.7
    - Strongly related: 0.7 <= |cos| < 0.9
    - Nearly identical/opposite: |cos| >= 0.9
    """
    categories = {
        "highly_orthogonal": [],     # |cos| < 0.1
        "orthogonal": [],             # 0.1 <= |cos| < 0.3
        "weakly_related": [],         # 0.3 <= |cos| < 0.5
        "related": [],                # 0.5 <= |cos| < 0.7
        "strongly_related": [],       # 0.7 <= |cos| < 0.9
        "nearly_identical_opposite": [] # |cos| >= 0.9
    }
    
    for c1, c2, sim in pairs:
        abs_sim = abs(sim)
        
        if abs_sim < 0.1:
            categories["highly_orthogonal"].append((c1, c2, sim))
        elif abs_sim < 0.3:
            categories["orthogonal"].append((c1, c2, sim))
        elif abs_sim < 0.5:
            categories["weakly_related"].append((c1, c2, sim))
        elif abs_sim < 0.7:
            categories["related"].append((c1, c2, sim))
        elif abs_sim < 0.9:
            categories["strongly_related"].append((c1, c2, sim))
        else:
            categories["nearly_identical_opposite"].append((c1, c2, sim))
    
    return categories


def test_composition_across_similarity_spectrum(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    pairs: List[Tuple[str, str, float]],
    layer: int,
    prompts: List[str],
    coefficient: float = 1.0,
    n_generations: int = 5,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Test composition for pairs across the full similarity spectrum.
    
    Returns:
        List of results with similarity and success metrics
    """
    quality_metrics = QualityMetrics()
    results = []
    
    print(f"\nTesting {len(pairs)} concept pairs across similarity spectrum...")
    
    for concept_a, concept_b, similarity in tqdm(pairs, desc="Testing pairs"):
        evaluator = MultiAttributeEvaluator([concept_a, concept_b])
        
        vec_a = steering_vectors[concept_a]
        vec_b = steering_vectors[concept_b]
        
        # Track results
        scores_a = []
        scores_b = []
        perplexities = []
        both_present_count = 0
        
        for prompt in prompts:
            for _ in range(n_generations):
                # Generate with A + B
                config = [
                    SteeringConfig(vector=vec_a, layer=layer, coefficient=coefficient),
                    SteeringConfig(vector=vec_b, layer=layer, coefficient=coefficient)
                ]
                text = generate_with_steering(model, tokenizer, prompt, config)
                
                # Evaluate
                scores = evaluator.evaluate(text, [concept_a, concept_b])
                score_a = scores[concept_a]
                score_b = scores[concept_b]
                
                scores_a.append(score_a)
                scores_b.append(score_b)
                
                # Check if both present
                if score_a > threshold and score_b > threshold:
                    both_present_count += 1
                
                # Quality
                ppl = quality_metrics.perplexity_calc.compute(text)
                perplexities.append(ppl)
        
        # Compute statistics
        total_samples = len(scores_a)
        joint_success_rate = both_present_count / total_samples
        
        result = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "cosine_similarity": similarity,
            "abs_cosine_similarity": abs(similarity),
            "mean_score_a": float(np.mean(scores_a)),
            "std_score_a": float(np.std(scores_a)),
            "mean_score_b": float(np.mean(scores_b)),
            "std_score_b": float(np.std(scores_b)),
            "joint_success_rate": float(joint_success_rate),
            "both_present_count": both_present_count,
            "total_samples": total_samples,
            "mean_perplexity": float(np.mean(perplexities)),
            "std_perplexity": float(np.std(perplexities))
        }
        
        results.append(result)
        
        print(f"  {concept_a} + {concept_b}: cos={similarity:.3f}, success={joint_success_rate:.1%}")
    
    return results


def analyze_correlation(results: List[Dict]) -> Dict:
    """
    Analyze correlation between cosine similarity and composition success.
    
    Tests:
    1. Pearson correlation (linear relationship)
    2. Spearman correlation (monotonic relationship)
    3. Optimal similarity range
    """
    similarities = [r["cosine_similarity"] for r in results]
    abs_similarities = [r["abs_cosine_similarity"] for r in results]
    success_rates = [r["joint_success_rate"] for r in results]
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(abs_similarities, success_rates)
    spearman_r, spearman_p = spearmanr(abs_similarities, success_rates)
    
    # Find optimal similarity range
    # Bin by similarity and compute mean success in each bin
    bins = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    bin_indices = np.digitize(abs_similarities, bins)
    
    bin_success = {}
    for bin_idx in range(1, len(bins)):
        in_bin = [success_rates[i] for i in range(len(success_rates)) 
                  if bin_indices[i] == bin_idx]
        if in_bin:
            bin_success[f"{bins[bin_idx-1]:.1f}-{bins[bin_idx]:.1f}"] = {
                "mean_success": float(np.mean(in_bin)),
                "count": len(in_bin)
            }
    
    # Find best and worst bins
    if bin_success:
        best_bin = max(bin_success.items(), key=lambda x: x[1]["mean_success"])
        worst_bin = min(bin_success.items(), key=lambda x: x[1]["mean_success"])
    else:
        best_bin = None
        worst_bin = None
    
    analysis = {
        "pearson_correlation": float(pearson_r),
        "pearson_pvalue": float(pearson_p),
        "spearman_correlation": float(spearman_r),
        "spearman_pvalue": float(spearman_p),
        "bin_success_rates": bin_success,
        "best_similarity_range": best_bin[0] if best_bin else None,
        "best_success_rate": best_bin[1]["mean_success"] if best_bin else None,
        "worst_similarity_range": worst_bin[0] if worst_bin else None,
        "worst_success_rate": worst_bin[1]["mean_success"] if worst_bin else None
    }
    
    return analysis


def test_hypothesis_orthogonal_better(results: List[Dict]) -> Dict:
    """
    Test specific hypothesis: Orthogonal vectors compose better than non-orthogonal.
    
    H0: Success rate is independent of orthogonality
    H1: Orthogonal pairs (|cos| < 0.3) have higher success than others
    """
    orthogonal = [r for r in results if r["abs_cosine_similarity"] < 0.3]
    non_orthogonal = [r for r in results if r["abs_cosine_similarity"] >= 0.3]
    
    if not orthogonal or not non_orthogonal:
        return {"note": "Insufficient data for hypothesis test"}
    
    orthogonal_success = [r["joint_success_rate"] for r in orthogonal]
    non_orthogonal_success = [r["joint_success_rate"] for r in non_orthogonal]
    
    # Compute statistics
    orth_mean = np.mean(orthogonal_success)
    orth_std = np.std(orthogonal_success)
    non_orth_mean = np.mean(non_orthogonal_success)
    non_orth_std = np.std(non_orthogonal_success)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((orth_std**2 + non_orth_std**2) / 2)
    cohens_d = (orth_mean - non_orth_mean) / pooled_std if pooled_std > 0 else 0
    
    # Simple t-test (if you have scipy)
    try:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(orthogonal_success, non_orthogonal_success)
    except:
        t_stat, p_value = None, None
    
    hypothesis_test = {
        "orthogonal_pairs": len(orthogonal),
        "non_orthogonal_pairs": len(non_orthogonal),
        "orthogonal_mean_success": float(orth_mean),
        "orthogonal_std_success": float(orth_std),
        "non_orthogonal_mean_success": float(non_orth_mean),
        "non_orthogonal_std_success": float(non_orth_std),
        "difference": float(orth_mean - non_orth_mean),
        "cohens_d": float(cohens_d),
        "t_statistic": float(t_stat) if t_stat is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "significant": p_value < 0.05 if p_value is not None else None
    }
    
    return hypothesis_test

def test_projection_hypothesis(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    pairs: List[Tuple[str, str, float]],
    layer: int,
    prompts: List[str],
    n_generations: int = 5
) -> List[Dict]:
    """
    Projection Test: Does only the parallel component of B affect concept A?
    
    For each pair (A, B):
    - Decompose B into B_parallel (component parallel to A) and B_orthogonal
    - Test if steering with B_parallel affects concept_a more than B_orthogonal
    
    Hypothesis: If steering respects geometry, B_parallel should affect A more.
    """
    print(f"\n{'='*60}")
    print("PROJECTION TEST: Does Geometry Predict Steering Effect?")
    print(f"{'='*60}\n")
    
    results = []
    
    # Test a subset of pairs with high and low similarity
    test_pairs = []
    
    # Get pairs with different similarity levels
    sorted_pairs = sorted(pairs, key=lambda x: abs(x[2]))
    test_pairs.extend(sorted_pairs[:3])  # Most orthogonal
    test_pairs.extend(sorted_pairs[-3:])  # Most similar
    
    for concept_a, concept_b, similarity in tqdm(test_pairs, desc="Testing projections"):
        evaluator = MultiAttributeEvaluator([concept_a, concept_b])
        
        vec_a = steering_vectors[concept_a]
        vec_b = steering_vectors[concept_b]
        
        # Compute projection of B onto A
        vec_a_normalized = vec_a / vec_a.norm()
        projection_coef = (vec_b @ vec_a_normalized).item()
        vec_b_parallel = projection_coef * vec_a_normalized
        vec_b_orthogonal = vec_b - vec_b_parallel
        
        # Compute magnitudes
        parallel_magnitude = vec_b_parallel.norm().item()
        orthogonal_magnitude = vec_b_orthogonal.norm().item()
        
        print(f"\n{concept_a} <- {concept_b}:")
        print(f"  Cosine similarity: {similarity:.3f}")
        print(f"  Parallel magnitude: {parallel_magnitude:.3f}")
        print(f"  Orthogonal magnitude: {orthogonal_magnitude:.3f}")
        
        # Test three conditions
        scores_full_b = []
        scores_parallel = []
        scores_orthogonal = []
        
        for prompt in prompts[:5]:  # Use subset for speed
            for _ in range(n_generations):
                # Full B vector
                config_b = SteeringConfig(vector=vec_b, layer=layer, coefficient=1.0)
                text_b = generate_with_steering(model, tokenizer, prompt, config_b)
                score_b = evaluator.classifiers[concept_a].score(text_b)
                scores_full_b.append(score_b)
                
                # Parallel component only
                config_parallel = SteeringConfig(vector=vec_b_parallel, layer=layer, coefficient=1.0)
                text_parallel = generate_with_steering(model, tokenizer, prompt, config_parallel)
                score_parallel = evaluator.classifiers[concept_a].score(text_parallel)
                scores_parallel.append(score_parallel)
                
                # Orthogonal component only
                config_orthogonal = SteeringConfig(vector=vec_b_orthogonal, layer=layer, coefficient=1.0)
                text_orthogonal = generate_with_steering(model, tokenizer, prompt, config_orthogonal)
                score_orthogonal = evaluator.classifiers[concept_a].score(text_orthogonal)
                scores_orthogonal.append(score_orthogonal)
        
        # Compute statistics
        mean_full = np.mean(scores_full_b)
        mean_parallel = np.mean(scores_parallel)
        mean_orthogonal = np.mean(scores_orthogonal)
        
        # Test hypothesis: parallel > orthogonal
        hypothesis_holds = mean_parallel > mean_orthogonal
        
        result = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "cosine_similarity": similarity,
            "parallel_magnitude": float(parallel_magnitude),
            "orthogonal_magnitude": float(orthogonal_magnitude),
            "mean_score_full_b": float(mean_full),
            "mean_score_parallel": float(mean_parallel),
            "mean_score_orthogonal": float(mean_orthogonal),
            "parallel_dominates": hypothesis_holds,
            "parallel_to_orthogonal_ratio": float(mean_parallel / mean_orthogonal) if mean_orthogonal > 0 else None
        }
        
        results.append(result)
        
        print(f"  Full B score: {mean_full:.3f}")
        print(f"  Parallel score: {mean_parallel:.3f}")
        print(f"  Orthogonal score: {mean_orthogonal:.3f}")
        print(f"  Parallel dominates: {hypothesis_holds}")
    
    # Summary
    n_parallel_dominates = sum(1 for r in results if r["parallel_dominates"])
    print(f"\n{'='*60}")
    print(f"Projection Test Summary:")
    print(f"  Parallel dominates: {n_parallel_dominates}/{len(results)} pairs")
    print(f"{'='*60}")
    
    return results


def test_magnitude_independence(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concepts: List[str],
    layer: int,
    prompts: List[str],
    n_generations: int = 5
) -> List[Dict]:
    """
    Magnitude Independence Test: Does direction matter more than magnitude?
    
    For each concept:
    - Test with original vector (has some norm)
    - Test with normalized vector at same effective coefficient
    
    Hypothesis: If only direction matters, results should be identical.
    """
    print(f"\n{'='*60}")
    print("MAGNITUDE INDEPENDENCE TEST: Direction vs Magnitude")
    print(f"{'='*60}\n")
    
    results = []
    
    for concept in tqdm(concepts[:5], desc="Testing magnitude independence"):  # Test subset
        evaluator = MultiAttributeEvaluator([concept])
        
        vec_original = steering_vectors[concept]
        original_norm = vec_original.norm().item()
        vec_normalized = vec_original / original_norm
        
        print(f"\n{concept.capitalize()}:")
        print(f"  Original norm: {original_norm:.3f}")
        
        # Test at coefficient = 1.0
        scores_original = []
        scores_normalized_scaled = []
        scores_normalized_unscaled = []
        
        for prompt in prompts[:5]:
            for _ in range(n_generations):
                # Original vector, coefficient 1.0
                config_orig = SteeringConfig(vector=vec_original, layer=layer, coefficient=1.0)
                text_orig = generate_with_steering(model, tokenizer, prompt, config_orig)
                score_orig = evaluator.classifiers[concept].score(text_orig)
                scores_original.append(score_orig)
                
                # Normalized vector, coefficient = original norm (should be equivalent)
                config_norm_scaled = SteeringConfig(
                    vector=vec_normalized, 
                    layer=layer, 
                    coefficient=original_norm
                )
                text_norm_scaled = generate_with_steering(model, tokenizer, prompt, config_norm_scaled)
                score_norm_scaled = evaluator.classifiers[concept].score(text_norm_scaled)
                scores_normalized_scaled.append(score_norm_scaled)
                
                # Normalized vector, coefficient 1.0 (tests pure direction)
                config_norm_unscaled = SteeringConfig(
                    vector=vec_normalized,
                    layer=layer,
                    coefficient=1.0
                )
                text_norm_unscaled = generate_with_steering(model, tokenizer, prompt, config_norm_unscaled)
                score_norm_unscaled = evaluator.classifiers[concept].score(text_norm_unscaled)
                scores_normalized_unscaled.append(score_norm_unscaled)
        
        # Compute statistics
        mean_orig = np.mean(scores_original)
        mean_norm_scaled = np.mean(scores_normalized_scaled)
        mean_norm_unscaled = np.mean(scores_normalized_unscaled)
        
        # Test equivalence
        # Original vs normalized+scaled should be nearly identical if implementation is correct
        implementation_correct = abs(mean_orig - mean_norm_scaled) < 0.05
        
        # Compare effect sizes
        direction_matters = abs(mean_norm_scaled - mean_norm_unscaled) > 0.1
        
        result = {
            "concept": concept,
            "original_norm": float(original_norm),
            "mean_score_original": float(mean_orig),
            "mean_score_normalized_scaled": float(mean_norm_scaled),
            "mean_score_normalized_unscaled": float(mean_norm_unscaled),
            "implementation_correct": implementation_correct,
            "magnitude_independent": abs(mean_orig - mean_norm_scaled) < 0.05,
            "direction_effect": float(mean_norm_scaled),
            "magnitude_effect": float(mean_norm_scaled - mean_norm_unscaled)
        }
        
        results.append(result)
        
        print(f"  Original (coef=1.0): {mean_orig:.3f}")
        print(f"  Normalized (coef={original_norm:.2f}): {mean_norm_scaled:.3f}")
        print(f"  Normalized (coef=1.0): {mean_norm_unscaled:.3f}")
        print(f"  Implementation correct: {implementation_correct}")
    
    # Summary
    n_correct = sum(1 for r in results if r["implementation_correct"])
    print(f"\n{'='*60}")
    print(f"Magnitude Independence Summary:")
    print(f"  Implementation correct: {n_correct}/{len(results)} concepts")
    print(f"{'='*60}")
    
    return results


def plot_similarity_vs_success(
    results: List[Dict],
    save_path: Path = None
):
    """
    Create scatter plot: cosine similarity vs composition success.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Raw cosine similarity (with sign)
    ax1 = axes[0]
    
    similarities = [r["cosine_similarity"] for r in results]
    success_rates = [r["joint_success_rate"] for r in results]
    
    # Color by success rate
    colors = plt.cm.RdYlGn([s for s in success_rates])
    
    ax1.scatter(similarities, success_rates, c=colors, s=100, alpha=0.6, edgecolors='black')
    
    # Add labels for notable pairs
    for r in results:
        if r["joint_success_rate"] > 0.6 or r["joint_success_rate"] < 0.2:
            ax1.annotate(
                f"{r['concept_a'][:3]}+{r['concept_b'][:3]}",
                (r["cosine_similarity"], r["joint_success_rate"]),
                fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points'
            )
    
    # Reference lines
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Success threshold')
    ax1.axvline(x=-0.3, color='blue', linestyle=':', alpha=0.3, label='Orthogonal range')
    ax1.axvline(x=0.3, color='blue', linestyle=':', alpha=0.3)
    
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Joint Success Rate')
    ax1.set_title('Composition Success vs Vector Similarity (Signed)')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Absolute cosine similarity
    ax2 = axes[1]
    
    abs_similarities = [r["abs_cosine_similarity"] for r in results]
    
    ax2.scatter(abs_similarities, success_rates, c=colors, s=100, alpha=0.6, edgecolors='black')
    
    # Fit trend line
    if len(abs_similarities) > 2:
        z = np.polyfit(abs_similarities, success_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax2.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.5, label='Linear fit')
    
    ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Success threshold')
    ax2.axvline(x=0.3, color='blue', linestyle=':', alpha=0.3, label='Orthogonal boundary')
    
    ax2.set_xlabel('Absolute Cosine Similarity')
    ax2.set_ylabel('Joint Success Rate')
    ax2.set_title('Composition Success vs Vector Similarity (Absolute)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_binned_analysis(
    analysis: Dict,
    save_path: Path = None
):
    """
    Plot success rate by similarity bins.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bin_success = analysis["bin_success_rates"]
    
    bins = list(bin_success.keys())
    success_rates = [bin_success[b]["mean_success"] for b in bins]
    counts = [bin_success[b]["count"] for b in bins]
    
    # Create bar plot
    bars = ax.bar(range(len(bins)), success_rates, alpha=0.7, color='steelblue')
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # Mark best and worst
    if analysis.get("best_similarity_range"):
        best_idx = bins.index(analysis["best_similarity_range"])
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(0.9)
    
    if analysis.get("worst_similarity_range"):
        worst_idx = bins.index(analysis["worst_similarity_range"])
        bars[worst_idx].set_color('red')
        bars[worst_idx].set_alpha(0.9)
    
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Success threshold')
    ax.set_xlabel('Absolute Cosine Similarity Range')
    ax.set_ylabel('Mean Joint Success Rate')
    ax.set_title('Composition Success by Similarity Bin')
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_category_comparison(
    categories: Dict,
    results: List[Dict],
    save_path: Path = None
):
    """
    Compare success rates across similarity categories.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Map pairs to their success rates
    pair_to_success = {
        (r["concept_a"], r["concept_b"]): r["joint_success_rate"]
        for r in results
    }
    
    category_names = []
    category_success = []
    category_counts = []
    
    for cat_name, pairs in categories.items():
        if pairs:
            successes = [pair_to_success[(c1, c2)] for c1, c2, _ in pairs 
                        if (c1, c2) in pair_to_success]
            if successes:
                category_names.append(cat_name.replace("_", " ").title())
                category_success.append(np.mean(successes))
                category_counts.append(len(successes))
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(category_names)))
    bars = ax.bar(range(len(category_names)), category_success, color=colors, alpha=0.7)
    
    # Add count labels
    for bar, count, success in zip(bars, category_counts, category_success):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={count}\n{success:.1%}', 
                ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Success threshold')
    ax.set_ylabel('Mean Joint Success Rate')
    ax.set_title('Composition Success by Similarity Category')
    ax.set_xticks(range(len(category_names)))
    ax.set_xticklabels(category_names, rotation=45, ha='right')
    ax.set_ylim(0, max(category_success) * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax

def plot_projection_test(
    results: List[Dict],
    save_path: Path = None
):
    """Plot projection test results."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Parallel vs Orthogonal scores
    ax1 = axes[0]
    
    concepts = [f"{r['concept_a'][:3]}←{r['concept_b'][:3]}" for r in results]
    parallel_scores = [r['mean_score_parallel'] for r in results]
    orthogonal_scores = [r['mean_score_orthogonal'] for r in results]
    
    x = np.arange(len(concepts))
    width = 0.35
    
    ax1.bar(x - width/2, parallel_scores, width, label='Parallel', alpha=0.7, color='blue')
    ax1.bar(x + width/2, orthogonal_scores, width, label='Orthogonal', alpha=0.7, color='orange')
    
    ax1.set_ylabel('Mean Score on Concept A')
    ax1.set_title('Projection Test: Parallel vs Orthogonal Components')
    ax1.set_xticks(x)
    ax1.set_xticklabels(concepts, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Similarity vs Dominance
    ax2 = axes[1]
    
    similarities = [abs(r['cosine_similarity']) for r in results]
    ratios = [r['parallel_to_orthogonal_ratio'] for r in results if r['parallel_to_orthogonal_ratio']]
    sims_with_ratio = [abs(r['cosine_similarity']) for r in results if r['parallel_to_orthogonal_ratio']]
    
    colors = ['green' if r['parallel_dominates'] else 'red' for r in results if r['parallel_to_orthogonal_ratio']]
    
    ax2.scatter(sims_with_ratio, ratios, c=colors, s=100, alpha=0.6, edgecolors='black')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal effect')
    
    ax2.set_xlabel('Absolute Cosine Similarity')
    ax2.set_ylabel('Parallel / Orthogonal Score Ratio')
    ax2.set_title('Does High Similarity → Parallel Dominates?')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


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
    """Run cosine similarity analysis."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/cosine_similarity")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=None, 
                       help="Maximum number of pairs to test")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}")
    
    # Load geometry
    with open(week1_dir / "geometry_analysis.json") as f:
        geometry = json.load(f)
    
    concepts = args.concepts or geometry["concepts"]
    default_layer = cfg.model.default_layer
    
    print("="*60)
    print("COSINE SIMILARITY vs COMPOSITION SUCCESS")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Concepts: {concepts}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load steering vectors
    print("Loading steering vectors...")
    steering_vectors_by_layer = load_cached_vectors(
        week1_dir / "vectors",
        concepts,
        cfg.model.steering_layers
    )
    steering_vectors = {c: vecs[default_layer] for c, vecs in steering_vectors_by_layer.items()}
    
    # Get prompts
    prompts = get_test_prompts()
    if args.quick:
        prompts = prompts[:3]
        n_gen = 2
    else:
        prompts = prompts[:8]
        n_gen = 3
    
    # Step 1: Compute all pairwise similarities
    print("\n" + "="*60)
    print("STEP 1: Computing Pairwise Similarities")
    print("="*60)
    
    all_pairs = compute_all_pairwise_similarities(steering_vectors)
    print(f"\nTotal pairs: {len(all_pairs)}")
    
    # Print similarity distribution
    similarities = [abs(sim) for _, _, sim in all_pairs]
    print(f"Similarity range: [{min(similarities):.3f}, {max(similarities):.3f}]")
    print(f"Mean |similarity|: {np.mean(similarities):.3f}")
    print(f"Median |similarity|: {np.median(similarities):.3f}")
    
    # Categorize pairs
    categories = categorize_by_similarity(all_pairs)
    print("\nSimilarity categories:")
    for cat, pairs in categories.items():
        if pairs:
            print(f"  {cat}: {len(pairs)} pairs")
    
    # Step 2: Select pairs to test
    print("\n" + "="*60)
    print("STEP 2: Selecting Pairs to Test")
    print("="*60)
    
    # Strategy: Sample across the similarity spectrum
    pairs_to_test = []
    
    # Ensure we have representation from each category
    for cat, pairs in categories.items():
        if pairs:
            # Take up to 2 pairs from each category
            n_from_category = len(pairs)
            pairs_to_test.extend(pairs[:n_from_category])
    
    # If max_pairs specified, sample additional pairs
    if args.max_pairs and len(pairs_to_test) < args.max_pairs:
        remaining = args.max_pairs - len(pairs_to_test)
        # Sample randomly from remaining pairs
        tested_set = set(pairs_to_test)
        remaining_pairs = [p for p in all_pairs if p not in tested_set]
        if remaining_pairs:
            import random
            additional = random.sample(remaining_pairs, min(remaining, len(remaining_pairs)))
            pairs_to_test.extend(additional)
    elif args.max_pairs:
        pairs_to_test = pairs_to_test[:args.max_pairs]
    
    print(f"\nTesting {len(pairs_to_test)} pairs across similarity spectrum")
    
    # Step 3: Test composition for all selected pairs
    print("\n" + "="*60)
    print("STEP 3: Testing Composition Across Similarity Spectrum")
    print("="*60)
    
    composition_results = test_composition_across_similarity_spectrum(
        model, tokenizer, steering_vectors,
        pairs_to_test, default_layer, prompts,
        coefficient=1.0, n_generations=n_gen
    )
    
    # Step 4: Analyze correlation
    print("\n" + "="*60)
    print("STEP 4: Analyzing Correlation")
    print("="*60)
    
    correlation_analysis = analyze_correlation(composition_results)
    
    print(f"\nPearson correlation: r={correlation_analysis['pearson_correlation']:.3f}, "
          f"p={correlation_analysis['pearson_pvalue']:.4f}")
    print(f"Spearman correlation: r={correlation_analysis['spearman_correlation']:.3f}, "
          f"p={correlation_analysis['spearman_pvalue']:.4f}")
    
    if correlation_analysis.get("best_similarity_range"):
        print(f"\nBest similarity range: {correlation_analysis['best_similarity_range']}")
        print(f"  Mean success: {correlation_analysis['best_success_rate']:.1%}")
    
    if correlation_analysis.get("worst_similarity_range"):
        print(f"\nWorst similarity range: {correlation_analysis['worst_similarity_range']}")
        print(f"  Mean success: {correlation_analysis['worst_success_rate']:.1%}")
    
    # Step 5: Test hypothesis
    print("\n" + "="*60)
    print("STEP 5: Testing Hypothesis (Orthogonal Better?)")
    print("="*60)
    
    hypothesis_test = test_hypothesis_orthogonal_better(composition_results)
    
    if "note" not in hypothesis_test:
        print(f"\nOrthogonal pairs (|cos| < 0.3): n={hypothesis_test['orthogonal_pairs']}")
        print(f"  Mean success: {hypothesis_test['orthogonal_mean_success']:.1%}")
        
        print(f"\nNon-orthogonal pairs (|cos| >= 0.3): n={hypothesis_test['non_orthogonal_pairs']}")
        print(f"  Mean success: {hypothesis_test['non_orthogonal_mean_success']:.1%}")
        
        print(f"\nDifference: {hypothesis_test['difference']:.1%}")
        print(f"Cohen's d: {hypothesis_test['cohens_d']:.3f}")
        
        if hypothesis_test.get("p_value") is not None:
            print(f"p-value: {hypothesis_test['p_value']:.4f}")
            print(f"Significant (α=0.05): {hypothesis_test['significant']}")
    else:
        print(hypothesis_test["note"])
    
    # Step 5.5: Projection Test
    print("\n" + "="*60)
    print("STEP 5.5: Projection Test")
    print("="*60)
    
    try:
        projection_results = test_projection_hypothesis(
            model, tokenizer, steering_vectors,
            pairs_to_test, default_layer, prompts,
            n_generations=n_gen
        )
    except Exception as e:
        print(f"Error in projection test: {e}")
        projection_results = []
    
    # Step 5.6: Magnitude Independence Test
    print("\n" + "="*60)
    print("STEP 5.6: Magnitude Independence Test")
    print("="*60)
    
    try:
        magnitude_results = test_magnitude_independence(
            model, tokenizer, steering_vectors,
            concepts, default_layer, prompts,
            n_generations=n_gen
        )
    except Exception as e:
        print(f"Error in magnitude test: {e}")
        magnitude_results = []
    
    # Step 6: Generate visualizations
    print("\n" + "="*60)
    print("STEP 6: Generating Visualizations")
    print("="*60)
    
    plot_similarity_vs_success(
        composition_results,
        save_path=figures_dir / "similarity_vs_success.png"
    )
    
    plot_binned_analysis(
        correlation_analysis,
        save_path=figures_dir / "binned_analysis.png"
    )
    
    plot_category_comparison(
        categories,
        composition_results,
        save_path=figures_dir / "category_comparison.png"
    )

    if projection_results:
        plot_projection_test(
            projection_results,
            save_path=figures_dir / "projection_test.png"
        )
    
    # Step 7: Save results
    print("\n" + "="*60)
    print("STEP 7: Saving Results")
    print("="*60)
    
    all_results = {
        "all_pairs": all_pairs,
        "categories": {
            cat: [(c1, c2, float(sim)) for c1, c2, sim in pairs]
            for cat, pairs in categories.items()
        },
        "composition_results": composition_results,
        "correlation_analysis": correlation_analysis,
        "hypothesis_test": hypothesis_test,
        "projection_test": projection_results,  # NEW
        "magnitude_test": magnitude_results     # NEW
    }

    all_results = convert_to_native(all_results)

    with open(output_dir / "cosine_similarity_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_dir / 'cosine_similarity_results.json'}")
    print(f"✓ Figures saved to {figures_dir}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nPairs tested: {len(composition_results)}")
    print(f"Correlation (|sim| vs success): r={correlation_analysis['pearson_correlation']:.3f}")

    if hypothesis_test.get("difference"):
        orth_better = hypothesis_test["difference"] > 0
        print(f"\nOrthogonal pairs {'perform better' if orth_better else 'do not perform better'}")
        print(f"  Difference: {abs(hypothesis_test['difference']):.1%}")

    # Key insight
    if abs(correlation_analysis['pearson_correlation']) < 0.3:
        print("\n💡 WEAK CORRELATION: Geometry alone does not predict composition success")
    elif correlation_analysis['pearson_correlation'] < -0.5:
        print("\n💡 STRONG NEGATIVE CORRELATION: More similar vectors compose worse")
    elif correlation_analysis['pearson_correlation'] > 0.5:
        print("\n💡 STRONG POSITIVE CORRELATION: More similar vectors compose better")
    else:
        print("\n💡 MODERATE CORRELATION: Geometry partially predicts composition success")

if __name__ == "__main__":
    main()