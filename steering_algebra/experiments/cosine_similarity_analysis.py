# """
# Cosine Similarity Experiment: Does Vector Geometry Predict Composition Success?

# Central Hypothesis:
#     Orthogonal vectors (low cosine similarity) should compose better than
#     aligned or opposing vectors (high absolute cosine similarity).

# Tests:
# 1. Generate with A+B for pairs across similarity spectrum
# 2. Measure joint success rate (both A and B present)
# 3. Analyze correlation between |cos(A,B)| and success
# 4. Identify optimal similarity range for composition
# """

# import torch
# import numpy as np
# from typing import Dict, List, Tuple
# from pathlib import Path
# import json
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import spearmanr, pearsonr

# from config import cfg
# from data.prompts import get_test_prompts
# from steering.apply_steering import (
#     SteeringConfig,
#     generate_with_steering
# )
# from evaluation.classifiers import MultiAttributeEvaluator
# from evaluation.metrics import QualityMetrics
# from evaluation.geometry import compute_cosine_similarity
# from extraction.extract_vectors import load_cached_vectors


# def compute_all_pairwise_similarities(
#     steering_vectors: Dict[str, torch.Tensor]
# ) -> List[Tuple[str, str, float]]:
#     """
#     Compute cosine similarity for all concept pairs.
    
#     Returns:
#         List of (concept_a, concept_b, similarity) sorted by absolute similarity
#     """
#     concepts = list(steering_vectors.keys())
#     pairs = []
    
#     for i, c1 in enumerate(concepts):
#         for j, c2 in enumerate(concepts):
#             if i < j:  # Only upper triangle
#                 sim = compute_cosine_similarity(
#                     steering_vectors[c1],
#                     steering_vectors[c2]
#                 )
#                 pairs.append((c1, c2, float(sim)))
    
#     return pairs


# def categorize_by_similarity(
#     pairs: List[Tuple[str, str, float]]
# ) -> Dict[str, List[Tuple[str, str, float]]]:
#     """
#     Categorize pairs by their cosine similarity.
    
#     Categories:
#     - Highly orthogonal: |cos| < 0.1
#     - Orthogonal: 0.1 <= |cos| < 0.3
#     - Weakly related: 0.3 <= |cos| < 0.5
#     - Related: 0.5 <= |cos| < 0.7
#     - Strongly related: 0.7 <= |cos| < 0.9
#     - Nearly identical/opposite: |cos| >= 0.9
#     """
#     categories = {
#         "highly_orthogonal": [],     # |cos| < 0.1
#         "orthogonal": [],             # 0.1 <= |cos| < 0.3
#         "weakly_related": [],         # 0.3 <= |cos| < 0.5
#         "related": [],                # 0.5 <= |cos| < 0.7
#         "strongly_related": [],       # 0.7 <= |cos| < 0.9
#         "nearly_identical_opposite": [] # |cos| >= 0.9
#     }
    
#     for c1, c2, sim in pairs:
#         abs_sim = abs(sim)
        
#         if abs_sim < 0.1:
#             categories["highly_orthogonal"].append((c1, c2, sim))
#         elif abs_sim < 0.3:
#             categories["orthogonal"].append((c1, c2, sim))
#         elif abs_sim < 0.5:
#             categories["weakly_related"].append((c1, c2, sim))
#         elif abs_sim < 0.7:
#             categories["related"].append((c1, c2, sim))
#         elif abs_sim < 0.9:
#             categories["strongly_related"].append((c1, c2, sim))
#         else:
#             categories["nearly_identical_opposite"].append((c1, c2, sim))
    
#     return categories


# def test_composition_across_similarity_spectrum(
#     model,
#     tokenizer,
#     steering_vectors: Dict[str, torch.Tensor],
#     pairs: List[Tuple[str, str, float]],
#     layer: int,
#     prompts: List[str],
#     coefficient: float = 1.0,
#     n_generations: int = 5,
#     threshold: float = 0.5
# ) -> List[Dict]:
#     """
#     Test composition for pairs across the full similarity spectrum.
    
#     Returns:
#         List of results with similarity and success metrics
#     """
#     quality_metrics = QualityMetrics()
#     results = []
    
#     print(f"\nTesting {len(pairs)} concept pairs across similarity spectrum...")
    
#     for concept_a, concept_b, similarity in tqdm(pairs, desc="Testing pairs"):
#         evaluator = MultiAttributeEvaluator([concept_a, concept_b])
        
#         vec_a = steering_vectors[concept_a]
#         vec_b = steering_vectors[concept_b]
        
#         # Track results
#         scores_a = []
#         scores_b = []
#         perplexities = []
#         both_present_count = 0
        
#         for prompt in prompts:
#             for _ in range(n_generations):
#                 # Generate with A + B
#                 config = [
#                     SteeringConfig(vector=vec_a, layer=layer, coefficient=coefficient),
#                     SteeringConfig(vector=vec_b, layer=layer, coefficient=coefficient)
#                 ]
#                 text = generate_with_steering(model, tokenizer, prompt, config)
                
#                 # Evaluate
#                 scores = evaluator.evaluate(text, [concept_a, concept_b])
#                 score_a = scores[concept_a]
#                 score_b = scores[concept_b]
                
#                 scores_a.append(score_a)
#                 scores_b.append(score_b)
                
#                 # Check if both present
#                 if score_a > threshold and score_b > threshold:
#                     both_present_count += 1
                
#                 # Quality
#                 ppl = quality_metrics.perplexity_calc.compute(text)
#                 perplexities.append(ppl)
        
#         # Compute statistics
#         total_samples = len(scores_a)
#         joint_success_rate = both_present_count / total_samples
        
#         result = {
#             "concept_a": concept_a,
#             "concept_b": concept_b,
#             "cosine_similarity": similarity,
#             "abs_cosine_similarity": abs(similarity),
#             "mean_score_a": float(np.mean(scores_a)),
#             "std_score_a": float(np.std(scores_a)),
#             "mean_score_b": float(np.mean(scores_b)),
#             "std_score_b": float(np.std(scores_b)),
#             "joint_success_rate": float(joint_success_rate),
#             "both_present_count": both_present_count,
#             "total_samples": total_samples,
#             "mean_perplexity": float(np.mean(perplexities)),
#             "std_perplexity": float(np.std(perplexities))
#         }
        
#         results.append(result)
        
#         print(f"  {concept_a} + {concept_b}: cos={similarity:.3f}, success={joint_success_rate:.1%}")
    
#     return results


# def analyze_correlation(results: List[Dict]) -> Dict:
#     """
#     Analyze correlation between cosine similarity and composition success.
    
#     Tests:
#     1. Pearson correlation (linear relationship)
#     2. Spearman correlation (monotonic relationship)
#     3. Optimal similarity range
#     """
#     similarities = [r["cosine_similarity"] for r in results]
#     abs_similarities = [r["abs_cosine_similarity"] for r in results]
#     success_rates = [r["joint_success_rate"] for r in results]
    
#     # Compute correlations
#     pearson_r, pearson_p = pearsonr(abs_similarities, success_rates)
#     spearman_r, spearman_p = spearmanr(abs_similarities, success_rates)
    
#     # Find optimal similarity range
#     # Bin by similarity and compute mean success in each bin
#     bins = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
#     bin_indices = np.digitize(abs_similarities, bins)
    
#     bin_success = {}
#     for bin_idx in range(1, len(bins)):
#         in_bin = [success_rates[i] for i in range(len(success_rates)) 
#                   if bin_indices[i] == bin_idx]
#         if in_bin:
#             bin_success[f"{bins[bin_idx-1]:.1f}-{bins[bin_idx]:.1f}"] = {
#                 "mean_success": float(np.mean(in_bin)),
#                 "count": len(in_bin)
#             }
    
#     # Find best and worst bins
#     if bin_success:
#         best_bin = max(bin_success.items(), key=lambda x: x[1]["mean_success"])
#         worst_bin = min(bin_success.items(), key=lambda x: x[1]["mean_success"])
#     else:
#         best_bin = None
#         worst_bin = None
    
#     analysis = {
#         "pearson_correlation": float(pearson_r),
#         "pearson_pvalue": float(pearson_p),
#         "spearman_correlation": float(spearman_r),
#         "spearman_pvalue": float(spearman_p),
#         "bin_success_rates": bin_success,
#         "best_similarity_range": best_bin[0] if best_bin else None,
#         "best_success_rate": best_bin[1]["mean_success"] if best_bin else None,
#         "worst_similarity_range": worst_bin[0] if worst_bin else None,
#         "worst_success_rate": worst_bin[1]["mean_success"] if worst_bin else None
#     }
    
#     return analysis


# def test_hypothesis_orthogonal_better(results: List[Dict]) -> Dict:
#     """
#     Test specific hypothesis: Orthogonal vectors compose better than non-orthogonal.
    
#     H0: Success rate is independent of orthogonality
#     H1: Orthogonal pairs (|cos| < 0.3) have higher success than others
#     """
#     orthogonal = [r for r in results if r["abs_cosine_similarity"] < 0.3]
#     non_orthogonal = [r for r in results if r["abs_cosine_similarity"] >= 0.3]
    
#     if not orthogonal or not non_orthogonal:
#         return {"note": "Insufficient data for hypothesis test"}
    
#     orthogonal_success = [r["joint_success_rate"] for r in orthogonal]
#     non_orthogonal_success = [r["joint_success_rate"] for r in non_orthogonal]
    
#     # Compute statistics
#     orth_mean = np.mean(orthogonal_success)
#     orth_std = np.std(orthogonal_success)
#     non_orth_mean = np.mean(non_orthogonal_success)
#     non_orth_std = np.std(non_orthogonal_success)
    
#     # Effect size (Cohen's d)
#     pooled_std = np.sqrt((orth_std**2 + non_orth_std**2) / 2)
#     cohens_d = (orth_mean - non_orth_mean) / pooled_std if pooled_std > 0 else 0
    
#     # Simple t-test (if you have scipy)
#     try:
#         from scipy.stats import ttest_ind
#         t_stat, p_value = ttest_ind(orthogonal_success, non_orthogonal_success)
#     except:
#         t_stat, p_value = None, None
    
#     hypothesis_test = {
#         "orthogonal_pairs": len(orthogonal),
#         "non_orthogonal_pairs": len(non_orthogonal),
#         "orthogonal_mean_success": float(orth_mean),
#         "orthogonal_std_success": float(orth_std),
#         "non_orthogonal_mean_success": float(non_orth_mean),
#         "non_orthogonal_std_success": float(non_orth_std),
#         "difference": float(orth_mean - non_orth_mean),
#         "cohens_d": float(cohens_d),
#         "t_statistic": float(t_stat) if t_stat is not None else None,
#         "p_value": float(p_value) if p_value is not None else None,
#         "significant": p_value < 0.05 if p_value is not None else None
#     }
    
#     return hypothesis_test

# def test_projection_hypothesis(
#     model,
#     tokenizer,
#     steering_vectors: Dict[str, torch.Tensor],
#     pairs: List[Tuple[str, str, float]],
#     layer: int,
#     prompts: List[str],
#     n_generations: int = 5
# ) -> List[Dict]:
#     """
#     Projection Test: Does only the parallel component of B affect concept A?
    
#     For each pair (A, B):
#     - Decompose B into B_parallel (component parallel to A) and B_orthogonal
#     - Test if steering with B_parallel affects concept_a more than B_orthogonal
    
#     Hypothesis: If steering respects geometry, B_parallel should affect A more.
#     """
#     print(f"\n{'='*60}")
#     print("PROJECTION TEST: Does Geometry Predict Steering Effect?")
#     print(f"{'='*60}\n")
    
#     results = []
    
#     # Test a subset of pairs with high and low similarity
#     test_pairs = []
    
#     # Get pairs with different similarity levels
#     sorted_pairs = sorted(pairs, key=lambda x: abs(x[2]))
#     test_pairs.extend(sorted_pairs[:3])  # Most orthogonal
#     test_pairs.extend(sorted_pairs[-3:])  # Most similar
    
#     for concept_a, concept_b, similarity in tqdm(test_pairs, desc="Testing projections"):
#         evaluator = MultiAttributeEvaluator([concept_a, concept_b])
        
#         vec_a = steering_vectors[concept_a]
#         vec_b = steering_vectors[concept_b]
        
#         # Compute projection of B onto A
#         vec_a_normalized = vec_a / vec_a.norm()
#         projection_coef = (vec_b @ vec_a_normalized).item()
#         vec_b_parallel = projection_coef * vec_a_normalized
#         vec_b_orthogonal = vec_b - vec_b_parallel
        
#         # Compute magnitudes
#         parallel_magnitude = vec_b_parallel.norm().item()
#         orthogonal_magnitude = vec_b_orthogonal.norm().item()
        
#         print(f"\n{concept_a} <- {concept_b}:")
#         print(f"  Cosine similarity: {similarity:.3f}")
#         print(f"  Parallel magnitude: {parallel_magnitude:.3f}")
#         print(f"  Orthogonal magnitude: {orthogonal_magnitude:.3f}")
        
#         # Test three conditions
#         scores_full_b = []
#         scores_parallel = []
#         scores_orthogonal = []
        
#         for prompt in prompts[:5]:  # Use subset for speed
#             for _ in range(n_generations):
#                 # Full B vector
#                 config_b = SteeringConfig(vector=vec_b, layer=layer, coefficient=1.0)
#                 text_b = generate_with_steering(model, tokenizer, prompt, config_b)
#                 score_b = evaluator.classifiers[concept_a].score(text_b)
#                 scores_full_b.append(score_b)
                
#                 # Parallel component only
#                 config_parallel = SteeringConfig(vector=vec_b_parallel, layer=layer, coefficient=1.0)
#                 text_parallel = generate_with_steering(model, tokenizer, prompt, config_parallel)
#                 score_parallel = evaluator.classifiers[concept_a].score(text_parallel)
#                 scores_parallel.append(score_parallel)
                
#                 # Orthogonal component only
#                 config_orthogonal = SteeringConfig(vector=vec_b_orthogonal, layer=layer, coefficient=1.0)
#                 text_orthogonal = generate_with_steering(model, tokenizer, prompt, config_orthogonal)
#                 score_orthogonal = evaluator.classifiers[concept_a].score(text_orthogonal)
#                 scores_orthogonal.append(score_orthogonal)
        
#         # Compute statistics
#         mean_full = np.mean(scores_full_b)
#         mean_parallel = np.mean(scores_parallel)
#         mean_orthogonal = np.mean(scores_orthogonal)
        
#         # Test hypothesis: parallel > orthogonal
#         hypothesis_holds = mean_parallel > mean_orthogonal
        
#         result = {
#             "concept_a": concept_a,
#             "concept_b": concept_b,
#             "cosine_similarity": similarity,
#             "parallel_magnitude": float(parallel_magnitude),
#             "orthogonal_magnitude": float(orthogonal_magnitude),
#             "mean_score_full_b": float(mean_full),
#             "mean_score_parallel": float(mean_parallel),
#             "mean_score_orthogonal": float(mean_orthogonal),
#             "parallel_dominates": hypothesis_holds,
#             "parallel_to_orthogonal_ratio": float(mean_parallel / mean_orthogonal) if mean_orthogonal > 0 else None
#         }
        
#         results.append(result)
        
#         print(f"  Full B score: {mean_full:.3f}")
#         print(f"  Parallel score: {mean_parallel:.3f}")
#         print(f"  Orthogonal score: {mean_orthogonal:.3f}")
#         print(f"  Parallel dominates: {hypothesis_holds}")
    
#     # Summary
#     n_parallel_dominates = sum(1 for r in results if r["parallel_dominates"])
#     print(f"\n{'='*60}")
#     print(f"Projection Test Summary:")
#     print(f"  Parallel dominates: {n_parallel_dominates}/{len(results)} pairs")
#     print(f"{'='*60}")
    
#     return results


# def test_magnitude_independence(
#     model,
#     tokenizer,
#     steering_vectors: Dict[str, torch.Tensor],
#     concepts: List[str],
#     layer: int,
#     prompts: List[str],
#     n_generations: int = 5
# ) -> List[Dict]:
#     """
#     Magnitude Independence Test: Does direction matter more than magnitude?
    
#     For each concept:
#     - Test with original vector (has some norm)
#     - Test with normalized vector at same effective coefficient
    
#     Hypothesis: If only direction matters, results should be identical.
#     """
#     print(f"\n{'='*60}")
#     print("MAGNITUDE INDEPENDENCE TEST: Direction vs Magnitude")
#     print(f"{'='*60}\n")
    
#     results = []
    
#     for concept in tqdm(concepts[:5], desc="Testing magnitude independence"):  # Test subset
#         evaluator = MultiAttributeEvaluator([concept])
        
#         vec_original = steering_vectors[concept]
#         original_norm = vec_original.norm().item()
#         vec_normalized = vec_original / original_norm
        
#         print(f"\n{concept.capitalize()}:")
#         print(f"  Original norm: {original_norm:.3f}")
        
#         # Test at coefficient = 1.0
#         scores_original = []
#         scores_normalized_scaled = []
#         scores_normalized_unscaled = []
        
#         for prompt in prompts[:5]:
#             for _ in range(n_generations):
#                 # Original vector, coefficient 1.0
#                 config_orig = SteeringConfig(vector=vec_original, layer=layer, coefficient=1.0)
#                 text_orig = generate_with_steering(model, tokenizer, prompt, config_orig)
#                 score_orig = evaluator.classifiers[concept].score(text_orig)
#                 scores_original.append(score_orig)
                
#                 # Normalized vector, coefficient = original norm (should be equivalent)
#                 config_norm_scaled = SteeringConfig(
#                     vector=vec_normalized, 
#                     layer=layer, 
#                     coefficient=original_norm
#                 )
#                 text_norm_scaled = generate_with_steering(model, tokenizer, prompt, config_norm_scaled)
#                 score_norm_scaled = evaluator.classifiers[concept].score(text_norm_scaled)
#                 scores_normalized_scaled.append(score_norm_scaled)
                
#                 # Normalized vector, coefficient 1.0 (tests pure direction)
#                 config_norm_unscaled = SteeringConfig(
#                     vector=vec_normalized,
#                     layer=layer,
#                     coefficient=1.0
#                 )
#                 text_norm_unscaled = generate_with_steering(model, tokenizer, prompt, config_norm_unscaled)
#                 score_norm_unscaled = evaluator.classifiers[concept].score(text_norm_unscaled)
#                 scores_normalized_unscaled.append(score_norm_unscaled)
        
#         # Compute statistics
#         mean_orig = np.mean(scores_original)
#         mean_norm_scaled = np.mean(scores_normalized_scaled)
#         mean_norm_unscaled = np.mean(scores_normalized_unscaled)
        
#         # Test equivalence
#         # Original vs normalized+scaled should be nearly identical if implementation is correct
#         implementation_correct = abs(mean_orig - mean_norm_scaled) < 0.05
        
#         # Compare effect sizes
#         direction_matters = abs(mean_norm_scaled - mean_norm_unscaled) > 0.1
        
#         result = {
#             "concept": concept,
#             "original_norm": float(original_norm),
#             "mean_score_original": float(mean_orig),
#             "mean_score_normalized_scaled": float(mean_norm_scaled),
#             "mean_score_normalized_unscaled": float(mean_norm_unscaled),
#             "implementation_correct": implementation_correct,
#             "magnitude_independent": abs(mean_orig - mean_norm_scaled) < 0.05,
#             "direction_effect": float(mean_norm_scaled),
#             "magnitude_effect": float(mean_norm_scaled - mean_norm_unscaled)
#         }
        
#         results.append(result)
        
#         print(f"  Original (coef=1.0): {mean_orig:.3f}")
#         print(f"  Normalized (coef={original_norm:.2f}): {mean_norm_scaled:.3f}")
#         print(f"  Normalized (coef=1.0): {mean_norm_unscaled:.3f}")
#         print(f"  Implementation correct: {implementation_correct}")
    
#     # Summary
#     n_correct = sum(1 for r in results if r["implementation_correct"])
#     print(f"\n{'='*60}")
#     print(f"Magnitude Independence Summary:")
#     print(f"  Implementation correct: {n_correct}/{len(results)} concepts")
#     print(f"{'='*60}")
    
#     return results


# def plot_similarity_vs_success(
#     results: List[Dict],
#     save_path: Path = None
# ):
#     """
#     Create scatter plot: cosine similarity vs composition success.
#     """
#     plt.style.use('seaborn-v0_8-whitegrid')
    
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Panel 1: Raw cosine similarity (with sign)
#     ax1 = axes[0]
    
#     similarities = [r["cosine_similarity"] for r in results]
#     success_rates = [r["joint_success_rate"] for r in results]
    
#     # Color by success rate
#     colors = plt.cm.RdYlGn([s for s in success_rates])
    
#     ax1.scatter(similarities, success_rates, c=colors, s=100, alpha=0.6, edgecolors='black')
    
#     # Add labels for notable pairs
#     for r in results:
#         if r["joint_success_rate"] > 0.6 or r["joint_success_rate"] < 0.2:
#             ax1.annotate(
#                 f"{r['concept_a'][:3]}+{r['concept_b'][:3]}",
#                 (r["cosine_similarity"], r["joint_success_rate"]),
#                 fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points'
#             )
    
#     # Reference lines
#     ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
#     ax1.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Success threshold')
#     ax1.axvline(x=-0.3, color='blue', linestyle=':', alpha=0.3, label='Orthogonal range')
#     ax1.axvline(x=0.3, color='blue', linestyle=':', alpha=0.3)
    
#     ax1.set_xlabel('Cosine Similarity')
#     ax1.set_ylabel('Joint Success Rate')
#     ax1.set_title('Composition Success vs Vector Similarity (Signed)')
#     ax1.set_xlim(-1, 1)
#     ax1.set_ylim(0, 1)
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Panel 2: Absolute cosine similarity
#     ax2 = axes[1]
    
#     abs_similarities = [r["abs_cosine_similarity"] for r in results]
    
#     ax2.scatter(abs_similarities, success_rates, c=colors, s=100, alpha=0.6, edgecolors='black')
    
#     # Fit trend line
#     if len(abs_similarities) > 2:
#         z = np.polyfit(abs_similarities, success_rates, 1)
#         p = np.poly1d(z)
#         x_line = np.linspace(0, 1, 100)
#         ax2.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.5, label='Linear fit')
    
#     ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Success threshold')
#     ax2.axvline(x=0.3, color='blue', linestyle=':', alpha=0.3, label='Orthogonal boundary')
    
#     ax2.set_xlabel('Absolute Cosine Similarity')
#     ax2.set_ylabel('Joint Success Rate')
#     ax2.set_title('Composition Success vs Vector Similarity (Absolute)')
#     ax2.set_xlim(0, 1)
#     ax2.set_ylim(0, 1)
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved: {save_path}")
    
#     return fig, axes


# def plot_binned_analysis(
#     analysis: Dict,
#     save_path: Path = None
# ):
#     """
#     Plot success rate by similarity bins.
#     """
#     plt.style.use('seaborn-v0_8-whitegrid')
    
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     bin_success = analysis["bin_success_rates"]
    
#     bins = list(bin_success.keys())
#     success_rates = [bin_success[b]["mean_success"] for b in bins]
#     counts = [bin_success[b]["count"] for b in bins]
    
#     # Create bar plot
#     bars = ax.bar(range(len(bins)), success_rates, alpha=0.7, color='steelblue')
    
#     # Add count labels
#     for i, (bar, count) in enumerate(zip(bars, counts)):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
#                 f'n={count}', ha='center', va='bottom', fontsize=9)
    
#     # Mark best and worst
#     if analysis.get("best_similarity_range"):
#         best_idx = bins.index(analysis["best_similarity_range"])
#         bars[best_idx].set_color('green')
#         bars[best_idx].set_alpha(0.9)
    
#     if analysis.get("worst_similarity_range"):
#         worst_idx = bins.index(analysis["worst_similarity_range"])
#         bars[worst_idx].set_color('red')
#         bars[worst_idx].set_alpha(0.9)
    
#     ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Success threshold')
#     ax.set_xlabel('Absolute Cosine Similarity Range')
#     ax.set_ylabel('Mean Joint Success Rate')
#     ax.set_title('Composition Success by Similarity Bin')
#     ax.set_xticks(range(len(bins)))
#     ax.set_xticklabels(bins, rotation=45, ha='right')
#     ax.set_ylim(0, 1)
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved: {save_path}")
    
#     return fig, ax


# def plot_category_comparison(
#     categories: Dict,
#     results: List[Dict],
#     save_path: Path = None
# ):
#     """
#     Compare success rates across similarity categories.
#     """
#     plt.style.use('seaborn-v0_8-whitegrid')
    
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Map pairs to their success rates
#     pair_to_success = {
#         (r["concept_a"], r["concept_b"]): r["joint_success_rate"]
#         for r in results
#     }
    
#     category_names = []
#     category_success = []
#     category_counts = []
    
#     for cat_name, pairs in categories.items():
#         if pairs:
#             successes = [pair_to_success[(c1, c2)] for c1, c2, _ in pairs 
#                         if (c1, c2) in pair_to_success]
#             if successes:
#                 category_names.append(cat_name.replace("_", " ").title())
#                 category_success.append(np.mean(successes))
#                 category_counts.append(len(successes))
    
#     # Create bar plot
#     colors = plt.cm.viridis(np.linspace(0, 1, len(category_names)))
#     bars = ax.bar(range(len(category_names)), category_success, color=colors, alpha=0.7)
    
#     # Add count labels
#     for bar, count, success in zip(bars, category_counts, category_success):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
#                 f'n={count}\n{success:.1%}', 
#                 ha='center', va='bottom', fontsize=10)
    
#     ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Success threshold')
#     ax.set_ylabel('Mean Joint Success Rate')
#     ax.set_title('Composition Success by Similarity Category')
#     ax.set_xticks(range(len(category_names)))
#     ax.set_xticklabels(category_names, rotation=45, ha='right')
#     ax.set_ylim(0, max(category_success) * 1.2)
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis='y')
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved: {save_path}")
    
#     return fig, ax

# def plot_projection_test(
#     results: List[Dict],
#     save_path: Path = None
# ):
#     """Plot projection test results."""
#     plt.style.use('seaborn-v0_8-whitegrid')
    
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Panel 1: Parallel vs Orthogonal scores
#     ax1 = axes[0]
    
#     concepts = [f"{r['concept_a'][:3]}←{r['concept_b'][:3]}" for r in results]
#     parallel_scores = [r['mean_score_parallel'] for r in results]
#     orthogonal_scores = [r['mean_score_orthogonal'] for r in results]
    
#     x = np.arange(len(concepts))
#     width = 0.35
    
#     ax1.bar(x - width/2, parallel_scores, width, label='Parallel', alpha=0.7, color='blue')
#     ax1.bar(x + width/2, orthogonal_scores, width, label='Orthogonal', alpha=0.7, color='orange')
    
#     ax1.set_ylabel('Mean Score on Concept A')
#     ax1.set_title('Projection Test: Parallel vs Orthogonal Components')
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(concepts, rotation=45, ha='right')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3, axis='y')
    
#     # Panel 2: Similarity vs Dominance
#     ax2 = axes[1]
    
#     similarities = [abs(r['cosine_similarity']) for r in results]
#     ratios = [r['parallel_to_orthogonal_ratio'] for r in results if r['parallel_to_orthogonal_ratio']]
#     sims_with_ratio = [abs(r['cosine_similarity']) for r in results if r['parallel_to_orthogonal_ratio']]
    
#     colors = ['green' if r['parallel_dominates'] else 'red' for r in results if r['parallel_to_orthogonal_ratio']]
    
#     ax2.scatter(sims_with_ratio, ratios, c=colors, s=100, alpha=0.6, edgecolors='black')
#     ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal effect')
    
#     ax2.set_xlabel('Absolute Cosine Similarity')
#     ax2.set_ylabel('Parallel / Orthogonal Score Ratio')
#     ax2.set_title('Does High Similarity → Parallel Dominates?')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved: {save_path}")
    
#     return fig, axes


# def convert_to_native(obj):
#     """Convert numpy types to Python native types."""
#     if isinstance(obj, dict):
#         return {k: convert_to_native(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_native(item) for item in obj]
#     elif isinstance(obj, (np.integer, np.int64, np.int32)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float64, np.float32)):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, np.bool_):
#         return bool(obj)
#     else:
#         return obj


# def main():
#     """Run cosine similarity analysis."""
#     import argparse
#     from transformers import AutoModelForCausalLM, AutoTokenizer
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default=cfg.model.name)
#     parser.add_argument("--output_dir", default="outputs/cosine_similarity")
#     parser.add_argument("--week1_dir", default="outputs/week1")
#     parser.add_argument("--concepts", nargs="+", default=None)
#     parser.add_argument("--quick", action="store_true")
#     parser.add_argument("--max_pairs", type=int, default=None, 
#                        help="Maximum number of pairs to test")
#     args = parser.parse_args()
    
#     # Setup
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     figures_dir = output_dir / "figures"
#     figures_dir.mkdir(exist_ok=True)
    
#     week1_dir = Path(args.week1_dir)
#     if not week1_dir.exists():
#         raise FileNotFoundError(f"Week 1 results not found at {week1_dir}")
    
#     # Load geometry
#     with open(week1_dir / "geometry_analysis.json") as f:
#         geometry = json.load(f)
    
#     concepts = args.concepts or geometry["concepts"]
#     default_layer = cfg.model.default_layer
    
#     print("="*60)
#     print("COSINE SIMILARITY vs COMPOSITION SUCCESS")
#     print("="*60)
#     print(f"Model: {args.model}")
#     print(f"Concepts: {concepts}")
#     print(f"Output: {output_dir}")
    
#     # Load model
#     print("\nLoading model...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )
#     model.eval()
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # Load steering vectors
#     print("Loading steering vectors...")
#     steering_vectors_by_layer = load_cached_vectors(
#         week1_dir / "vectors",
#         concepts,
#         cfg.model.steering_layers
#     )
#     steering_vectors = {c: vecs[default_layer] for c, vecs in steering_vectors_by_layer.items()}
    
#     # Get prompts
#     prompts = get_test_prompts()
#     if args.quick:
#         prompts = prompts[:3]
#         n_gen = 2
#     else:
#         prompts = prompts[:8]
#         n_gen = 3
    
#     # Step 1: Compute all pairwise similarities
#     print("\n" + "="*60)
#     print("STEP 1: Computing Pairwise Similarities")
#     print("="*60)
    
#     all_pairs = compute_all_pairwise_similarities(steering_vectors)
#     print(f"\nTotal pairs: {len(all_pairs)}")
    
#     # Print similarity distribution
#     similarities = [abs(sim) for _, _, sim in all_pairs]
#     print(f"Similarity range: [{min(similarities):.3f}, {max(similarities):.3f}]")
#     print(f"Mean |similarity|: {np.mean(similarities):.3f}")
#     print(f"Median |similarity|: {np.median(similarities):.3f}")
    
#     # Categorize pairs
#     categories = categorize_by_similarity(all_pairs)
#     print("\nSimilarity categories:")
#     for cat, pairs in categories.items():
#         if pairs:
#             print(f"  {cat}: {len(pairs)} pairs")
    
#     # Step 2: Select pairs to test
#     print("\n" + "="*60)
#     print("STEP 2: Selecting Pairs to Test")
#     print("="*60)
    
#     # Strategy: Sample across the similarity spectrum
#     pairs_to_test = []
    
#     # Ensure we have representation from each category
#     for cat, pairs in categories.items():
#         if pairs:
#             # Take up to 2 pairs from each category
#             n_from_category = len(pairs)
#             pairs_to_test.extend(pairs[:n_from_category])
    
#     # If max_pairs specified, sample additional pairs
#     if args.max_pairs and len(pairs_to_test) < args.max_pairs:
#         remaining = args.max_pairs - len(pairs_to_test)
#         # Sample randomly from remaining pairs
#         tested_set = set(pairs_to_test)
#         remaining_pairs = [p for p in all_pairs if p not in tested_set]
#         if remaining_pairs:
#             import random
#             additional = random.sample(remaining_pairs, min(remaining, len(remaining_pairs)))
#             pairs_to_test.extend(additional)
#     elif args.max_pairs:
#         pairs_to_test = pairs_to_test[:args.max_pairs]
    
#     print(f"\nTesting {len(pairs_to_test)} pairs across similarity spectrum")
    
#     # Step 3: Test composition for all selected pairs
#     print("\n" + "="*60)
#     print("STEP 3: Testing Composition Across Similarity Spectrum")
#     print("="*60)
    
#     composition_results = test_composition_across_similarity_spectrum(
#         model, tokenizer, steering_vectors,
#         pairs_to_test, default_layer, prompts,
#         coefficient=1.0, n_generations=n_gen
#     )
    
#     # Step 4: Analyze correlation
#     print("\n" + "="*60)
#     print("STEP 4: Analyzing Correlation")
#     print("="*60)
    
#     correlation_analysis = analyze_correlation(composition_results)
    
#     print(f"\nPearson correlation: r={correlation_analysis['pearson_correlation']:.3f}, "
#           f"p={correlation_analysis['pearson_pvalue']:.4f}")
#     print(f"Spearman correlation: r={correlation_analysis['spearman_correlation']:.3f}, "
#           f"p={correlation_analysis['spearman_pvalue']:.4f}")
    
#     if correlation_analysis.get("best_similarity_range"):
#         print(f"\nBest similarity range: {correlation_analysis['best_similarity_range']}")
#         print(f"  Mean success: {correlation_analysis['best_success_rate']:.1%}")
    
#     if correlation_analysis.get("worst_similarity_range"):
#         print(f"\nWorst similarity range: {correlation_analysis['worst_similarity_range']}")
#         print(f"  Mean success: {correlation_analysis['worst_success_rate']:.1%}")
    
#     # Step 5: Test hypothesis
#     print("\n" + "="*60)
#     print("STEP 5: Testing Hypothesis (Orthogonal Better?)")
#     print("="*60)
    
#     hypothesis_test = test_hypothesis_orthogonal_better(composition_results)
    
#     if "note" not in hypothesis_test:
#         print(f"\nOrthogonal pairs (|cos| < 0.3): n={hypothesis_test['orthogonal_pairs']}")
#         print(f"  Mean success: {hypothesis_test['orthogonal_mean_success']:.1%}")
        
#         print(f"\nNon-orthogonal pairs (|cos| >= 0.3): n={hypothesis_test['non_orthogonal_pairs']}")
#         print(f"  Mean success: {hypothesis_test['non_orthogonal_mean_success']:.1%}")
        
#         print(f"\nDifference: {hypothesis_test['difference']:.1%}")
#         print(f"Cohen's d: {hypothesis_test['cohens_d']:.3f}")
        
#         if hypothesis_test.get("p_value") is not None:
#             print(f"p-value: {hypothesis_test['p_value']:.4f}")
#             print(f"Significant (α=0.05): {hypothesis_test['significant']}")
#     else:
#         print(hypothesis_test["note"])
    
#     # Step 5.5: Projection Test
#     print("\n" + "="*60)
#     print("STEP 5.5: Projection Test")
#     print("="*60)
    
#     try:
#         projection_results = test_projection_hypothesis(
#             model, tokenizer, steering_vectors,
#             pairs_to_test, default_layer, prompts,
#             n_generations=n_gen
#         )
#     except Exception as e:
#         print(f"Error in projection test: {e}")
#         projection_results = []
    
#     # Step 5.6: Magnitude Independence Test
#     print("\n" + "="*60)
#     print("STEP 5.6: Magnitude Independence Test")
#     print("="*60)
    
#     try:
#         magnitude_results = test_magnitude_independence(
#             model, tokenizer, steering_vectors,
#             concepts, default_layer, prompts,
#             n_generations=n_gen
#         )
#     except Exception as e:
#         print(f"Error in magnitude test: {e}")
#         magnitude_results = []
    
#     # Step 6: Generate visualizations
#     print("\n" + "="*60)
#     print("STEP 6: Generating Visualizations")
#     print("="*60)
    
#     plot_similarity_vs_success(
#         composition_results,
#         save_path=figures_dir / "similarity_vs_success.png"
#     )
    
#     plot_binned_analysis(
#         correlation_analysis,
#         save_path=figures_dir / "binned_analysis.png"
#     )
    
#     plot_category_comparison(
#         categories,
#         composition_results,
#         save_path=figures_dir / "category_comparison.png"
#     )

#     if projection_results:
#         plot_projection_test(
#             projection_results,
#             save_path=figures_dir / "projection_test.png"
#         )
    
#     # Step 7: Save results
#     print("\n" + "="*60)
#     print("STEP 7: Saving Results")
#     print("="*60)
    
#     all_results = {
#         "all_pairs": all_pairs,
#         "categories": {
#             cat: [(c1, c2, float(sim)) for c1, c2, sim in pairs]
#             for cat, pairs in categories.items()
#         },
#         "composition_results": composition_results,
#         "correlation_analysis": correlation_analysis,
#         "hypothesis_test": hypothesis_test,
#         "projection_test": projection_results,  # NEW
#         "magnitude_test": magnitude_results     # NEW
#     }

#     all_results = convert_to_native(all_results)

#     with open(output_dir / "cosine_similarity_results.json", "w") as f:
#         json.dump(all_results, f, indent=2)

#     print(f"\n✓ Results saved to {output_dir / 'cosine_similarity_results.json'}")
#     print(f"✓ Figures saved to {figures_dir}")

#     # Summary
#     print("\n" + "="*60)
#     print("SUMMARY")
#     print("="*60)

#     print(f"\nPairs tested: {len(composition_results)}")
#     print(f"Correlation (|sim| vs success): r={correlation_analysis['pearson_correlation']:.3f}")

#     if hypothesis_test.get("difference"):
#         orth_better = hypothesis_test["difference"] > 0
#         print(f"\nOrthogonal pairs {'perform better' if orth_better else 'do not perform better'}")
#         print(f"  Difference: {abs(hypothesis_test['difference']):.1%}")

#     # Key insight
#     if abs(correlation_analysis['pearson_correlation']) < 0.3:
#         print("\n💡 WEAK CORRELATION: Geometry alone does not predict composition success")
#     elif correlation_analysis['pearson_correlation'] < -0.5:
#         print("\n💡 STRONG NEGATIVE CORRELATION: More similar vectors compose worse")
#     elif correlation_analysis['pearson_correlation'] > 0.5:
#         print("\n💡 STRONG POSITIVE CORRELATION: More similar vectors compose better")
#     else:
#         print("\n💡 MODERATE CORRELATION: Geometry partially predicts composition success")

# if __name__ == "__main__":
#     main()


# # """
# # Interference Experiment: Does Semantic Overlap Predict Composition Success?

# # Hypothesis:
# #     Geometric orthogonality (cosine similarity) is a poor predictor of success.
# #     True failure is caused by "Resource Interference":
# #     1. Logit Interference: Vectors trying to boost the same tokens.
# #     2. Neuron Interference: Vectors using the same active neurons.
# # """

# # import torch
# # import numpy as np
# # import pandas as pd
# # from typing import Dict, List, Tuple
# # from pathlib import Path
# # import json
# # from tqdm import tqdm
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from scipy.stats import pearsonr

# # from config import cfg
# # from data.prompts import get_test_prompts
# # from steering.apply_steering import SteeringConfig, generate_with_steering
# # from evaluation.classifiers import MultiAttributeEvaluator
# # from evaluation.geometry import compute_cosine_similarity
# # from extraction.extract_vectors import load_cached_vectors


# # # ==========================================
# # # 1. INTERFERENCE METRICS
# # # ==========================================

# # def get_top_logits_and_neurons(
# #     vector: torch.Tensor, 
# #     unembedding_matrix: torch.Tensor, 
# #     k_tokens: int = 200,  # Increased to 200 to catch tail effects
# #     k_neurons_percent: float = 0.05
# # ):
# #     """
# #     Project vector to vocab space (Logit Lens) and identify active neurons.
# #     Returns INDICES and PROBABILITIES to measure soft overlap.
# #     """
# #     target_device = unembedding_matrix.device
# #     target_dtype = unembedding_matrix.dtype
    
# #     if vector.device != target_device or vector.dtype != target_dtype:
# #         vector = vector.to(device=target_device, dtype=target_dtype)

# #     # 1. Logit Lens: Project to vocabulary
# #     logits = vector @ unembedding_matrix
    
# #     # Softmax to get "probabilities" (activations)
# #     probs = torch.softmax(logits, dim=-1)
    
# #     # Get top K tokens
# #     top_k_vals, top_k_indices = torch.topk(probs, k_tokens)
# #     top_tokens = {
# #         idx.item(): val.item() 
# #         for idx, val in zip(top_k_indices, top_k_vals)
# #     }
    
# #     # 2. Active Neurons: Top k% magnitude in the vector itself
# #     k_neurons = int(vector.shape[0] * k_neurons_percent)
# #     top_neuron_indices = torch.topk(torch.abs(vector), k_neurons).indices.cpu().numpy().tolist()
    
# #     return top_tokens, set(top_neuron_indices)

# # def compute_soft_overlap(dict_a: Dict[int, float], dict_b: Dict[int, float]) -> float:
# #     """
# #     Compute 'Soft Overlap': Sum of shared probability mass.
# #     Much smoother than Jaccard similarity for sparse data.
# #     """
# #     keys_a = set(dict_a.keys())
# #     keys_b = set(dict_b.keys())
# #     intersection = keys_a.intersection(keys_b)
    
# #     if not intersection:
# #         return 0.0
    
# #     # For every shared token, how much 'energy' do they both put into it?
# #     # We take the minimum energy contributed by either vector.
# #     overlap_mass = sum(min(dict_a[k], dict_b[k]) for k in intersection)
    
# #     # Normalize by the total mass (which is roughly sum of top-k probs)
# #     total_mass_a = sum(dict_a.values())
# #     total_mass_b = sum(dict_b.values())
    
# #     return overlap_mass / (min(total_mass_a, total_mass_b) + 1e-9)

# # def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
# #     """Compute Jaccard Index (Intersection / Union)."""
# #     if not set_a or not set_b:
# #         return 0.0
# #     intersection = len(set_a.intersection(set_b))
# #     union = len(set_a.union(set_b))
# #     return intersection / union

# # def compute_pairwise_metrics(
# #     steering_vectors: Dict[str, torch.Tensor],
# #     model
# # ) -> List[Dict]:
# #     """Compute Cosine, Logit Overlap (Soft), and Neuron Overlap."""
# #     concepts = list(steering_vectors.keys())
# #     pairs_data = []
    
# #     # Get Unembedding Matrix
# #     if hasattr(model, "lm_head"):
# #         W_U = model.lm_head.weight.detach()
# #         if W_U.shape[0] != steering_vectors[concepts[0]].shape[0]:
# #             W_U = W_U.T
# #     elif hasattr(model, "get_output_embeddings"):
# #         W_U = model.get_output_embeddings().weight.detach().T
# #     else:
# #         print("Warning: Could not find Unembedding Matrix. Skipping Logit Lens.")
# #         W_U = None

# #     print("Pre-computing interference metrics (Logit Lens)...")
    
# #     concept_features = {}
# #     for c in concepts:
# #         if W_U is not None:
# #             # Returns dict{token_id: prob} and set{neuron_id}
# #             tokens, neurons = get_top_logits_and_neurons(steering_vectors[c], W_U)
# #         else:
# #             tokens, neurons = {}, set()
# #         concept_features[c] = {"tokens": tokens, "neurons": neurons}

# #     # Compare all pairs
# #     for i, c1 in enumerate(concepts):
# #         for j, c2 in enumerate(concepts):
# #             if i < j: # Upper triangle only
# #                 vec_a = steering_vectors[c1]
# #                 vec_b = steering_vectors[c2]
                
# #                 # Geometry
# #                 cos_sim = float(compute_cosine_similarity(vec_a, vec_b))
                
# #                 # Semantics (Soft Overlap)
# #                 logit_overlap = compute_soft_overlap(
# #                     concept_features[c1]["tokens"], 
# #                     concept_features[c2]["tokens"]
# #                 )
                
# #                 # Mechanism
# #                 neuron_overlap = compute_jaccard_similarity(
# #                     concept_features[c1]["neurons"], 
# #                     concept_features[c2]["neurons"]
# #                 )
                
# #                 pairs_data.append({
# #                     "concept_a": c1,
# #                     "concept_b": c2,
# #                     "cosine_similarity": cos_sim,
# #                     "abs_cosine_similarity": abs(cos_sim),
# #                     "logit_overlap": logit_overlap,
# #                     "neuron_overlap": neuron_overlap
# #                 })
    
# #     return pairs_data

# # # ==========================================
# # # 2. STEERING LOOP
# # # ==========================================

# # def test_composition_with_interference(
# #     model,
# #     tokenizer,
# #     steering_vectors: Dict[str, torch.Tensor],
# #     pairs_data: List[Dict],
# #     layer: int,
# #     prompts: List[str],
# #     n_generations: int = 5
# # ) -> List[Dict]:
    
# #     evaluator_cache = {} 
# #     results = []
    
# #     # Filter: Remove trivial synonyms (>0.8 similarity)
# #     filtered_pairs = [p for p in pairs_data if p["abs_cosine_similarity"] < 0.8]
# #     print(f"\nTesting {len(filtered_pairs)} pairs (Filtered out {len(pairs_data) - len(filtered_pairs)} synonyms)")
    
# #     for pair in tqdm(filtered_pairs, desc="Steering"):
# #         c1, c2 = pair["concept_a"], pair["concept_b"]
        
# #         # Lazy load evaluator & FORCE CPU
# #         pair_key = tuple(sorted([c1, c2]))
# #         if pair_key not in evaluator_cache:
# #             try:
# #                 evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2], device="cpu")
# #             except TypeError:
# #                 print(f"Warning: Evaluator for {c1}/{c2} does not accept 'device'. Using default.")
# #                 evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2])

# #         evaluator = evaluator_cache[pair_key]
        
# #         vec_a = steering_vectors[c1]
# #         vec_b = steering_vectors[c2]
        
# #         success_count = 0
# #         total = 0
        
# #         for prompt in prompts:
# #             for _ in range(n_generations):
# #                 # Apply both vectors
# #                 config = [
# #                     SteeringConfig(vector=vec_a, layer=layer, coefficient=1.0),
# #                     SteeringConfig(vector=vec_b, layer=layer, coefficient=1.0)
# #                 ]
# #                 text = generate_with_steering(model, tokenizer, prompt, config)
                
# #                 # Check success
# #                 scores = evaluator.evaluate(text, [c1, c2])
# #                 if scores[c1] > 0.5 and scores[c2] > 0.5:
# #                     success_count += 1
# #                 total += 1
        
# #         pair["joint_success_rate"] = success_count / total
# #         results.append(pair)
        
# #     return results

# # # ==========================================
# # # 3. VISUALIZATION & SUMMARY
# # # ==========================================

# # def print_summary_stats(results: List[Dict]):
# #     """Print clear correlation statistics."""
# #     success = [r["joint_success_rate"] for r in results]
# #     cosine = [r["abs_cosine_similarity"] for r in results]
# #     logit = [r["logit_overlap"] for r in results]
    
# #     def safe_corr(x, y):
# #         if np.std(x) == 0 or np.std(y) == 0:
# #             return 0.0
# #         return pearsonr(x, y)[0]

# #     r_cosine = safe_corr(cosine, success)
# #     r_logit = safe_corr(logit, success)
    
# #     print("\n" + "="*50)
# #     print("EXPERIMENT SUMMARY")
# #     print("="*50)
# #     print(f"Total Pairs Tested: {len(results)}")
# #     print(f"Avg Success Rate:   {np.mean(success):.1%}")
# #     print(f"Avg Logit Overlap:  {np.mean(logit):.4f}")
# #     print("-" * 50)
# #     print(f"Correlation (Cosine Sim vs Success): r = {r_cosine:.3f}")
# #     print(f"Correlation (Logit Overlap vs Success): r = {r_logit:.3f}")
# #     print("-" * 50)
    
# #     if abs(r_logit) > abs(r_cosine):
# #         print(">> HYPOTHESIS CONFIRMED: Logit Overlap predicts failure better than Geometry.")
# #     else:
# #         print(">> RESULT: Geometry remains a stronger predictor (or Overlap is too sparse).")
# #     print("="*50 + "\n")


# # def plot_results(results: List[Dict], save_dir: Path):
# #     """Generate clear regression plots, robust against sparse data."""
# #     sns.set_theme(style="whitegrid")
    
# #     # Create DataFrame
# #     df = pd.DataFrame({
# #         "Success Rate": [r["joint_success_rate"] for r in results],
# #         "Abs Cosine Sim": [r["abs_cosine_similarity"] for r in results],
# #         "Logit Overlap": [r["logit_overlap"] for r in results],
# #         "Neuron Overlap": [r["neuron_overlap"] for r in results]
# #     })
    
# #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
# #     # Plot A: Geometry
# #     sns.regplot(x="Abs Cosine Sim", y="Success Rate", data=df, ax=axes[0], 
# #                 scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
# #     axes[0].set_title("Geometry: Cosine Similarity")
# #     axes[0].set_ylim(-0.1, 1.1)

# #     # Plot B: Logit Overlap
# #     sns.regplot(x="Logit Overlap", y="Success Rate", data=df, ax=axes[1], 
# #                 scatter_kws={'alpha':0.5, 'color':'orange'}, line_kws={'color':'red'})
# #     axes[1].set_title("Semantics: Logit Overlap")
# #     axes[1].set_ylim(-0.1, 1.1)

# #     # Plot C: Neuron Overlap
# #     sns.regplot(x="Neuron Overlap", y="Success Rate", data=df, ax=axes[2], 
# #                 scatter_kws={'alpha':0.5, 'color':'purple'}, line_kws={'color':'red'})
# #     axes[2].set_title("Mechanism: Neuron Overlap")
# #     axes[2].set_ylim(-0.1, 1.1)

# #     plt.tight_layout()
# #     plt.savefig(save_dir / "interference_trends.png")
    
# #     # Bar Chart: Robust Binning
# #     plt.figure(figsize=(8, 6))
    
# #     # Handle sparse data: only bin if we have enough variance
# #     if df["Logit Overlap"].std() > 0.001:
# #         # Try qcut (quantiles) first
# #         try:
# #             df['Overlap Bin'] = pd.qcut(df['Logit Overlap'], q=3, labels=["Low", "Med", "High"])
# #         except ValueError:
# #             # Fallback to cut (fixed ranges) if data is clumpy
# #             df['Overlap Bin'] = pd.cut(df['Logit Overlap'], bins=3, labels=["Low", "Med", "High"])
            
# #         sns.barplot(x="Overlap Bin", y="Success Rate", data=df, palette="viridis")
# #         plt.title("Composition Success by Logit Overlap Level")
# #         plt.ylim(0, 1.0)
# #         plt.savefig(save_dir / "interference_summary_bar.png")
# #     else:
# #         print("Skipping bar chart: Not enough variance in Logit Overlap.")
    
# #     print(f"Saved plots to {save_dir}")

# # def convert_to_native(obj):
# #     """Helper for JSON serialization."""
# #     if isinstance(obj, dict):
# #         return {k: convert_to_native(v) for k, v in obj.items()}
# #     elif isinstance(obj, list):
# #         return [convert_to_native(item) for item in obj]
# #     elif isinstance(obj, (np.integer, np.int64, np.int32)):
# #         return int(obj)
# #     elif isinstance(obj, (np.floating, np.float64, np.float32)):
# #         return float(obj)
# #     elif isinstance(obj, np.ndarray):
# #         return obj.tolist()
# #     else:
# #         return obj

# # # ==========================================
# # # 4. MAIN EXECUTION
# # # ==========================================

# # def main():
# #     import argparse
# #     from transformers import AutoModelForCausalLM, AutoTokenizer
    
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--model", default=cfg.model.name)
# #     parser.add_argument("--week1_dir", default="outputs/week1")
# #     parser.add_argument("--output_dir", default="outputs/interference_analysis")
# #     parser.add_argument("--quick", action="store_true")
# #     args = parser.parse_args()
    
# #     output_dir = Path(args.output_dir)
# #     output_dir.mkdir(parents=True, exist_ok=True)
    
# #     # 1. Load Model
# #     print("Loading model resources...")
# #     tokenizer = AutoTokenizer.from_pretrained(args.model)
# #     model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
# #     # 2. Load Cached Vectors
# #     print("Loading vectors...")
# #     with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
# #         concepts = json.load(f)["concepts"]
        
# #     vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    
# #     # Move vectors to model device for processing
# #     vectors = {c: v[cfg.model.default_layer].to(model.device) for c, v in vectors_by_layer.items()}
    
# #     # 3. Compute Metrics (Logit Lens)
# #     pairs_data = compute_pairwise_metrics(vectors, model)
    
# #     # 4. Run Steering Tests
# #     prompts = get_test_prompts()[:2] if args.quick else get_test_prompts()[:5]
# #     n_gen = 2 if args.quick else 3
    
# #     results = test_composition_with_interference(
# #         model, tokenizer, vectors, pairs_data, 
# #         cfg.model.default_layer, prompts, n_generations=n_gen
# #     )
    
# #     # 5. Analyze & Save
# #     print_summary_stats(results)
# #     plot_results(results, output_dir)
    
# #     with open(output_dir / "interference_data.json", "w") as f:
# #         json.dump(convert_to_native(results), f, indent=2)

# # if __name__ == "__main__":
# #     main()


# """
# Vocabulary-Projected Cosine Analysis: The "True" Semantic Angle.

# Hypothesis:
#     Interaction happens in the vocabulary output space. 
#     We project vectors into Logit Space (v @ W_U) and measure the angle there.
#     This dense metric captures semantic alignment better than sparse token overlap.
# """

# import torch
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple, Any
# from pathlib import Path
# import json
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# from config import cfg
# from data.prompts import get_test_prompts
# from steering.apply_steering import SteeringConfig, generate_with_steering
# from evaluation.classifiers import MultiAttributeEvaluator
# from evaluation.geometry import compute_cosine_similarity
# from extraction.extract_vectors import load_cached_vectors


# # ==========================================
# # 1. METRICS: RESIDUAL vs. VOCABULARY
# # ==========================================

# def compute_vocab_projected_similarity(
#     vec_a: torch.Tensor,
#     vec_b: torch.Tensor, 
#     unembedding_matrix: torch.Tensor
# ) -> float:
#     """
#     1. Project vectors to Vocab Space (Logits).
#     2. Compute Cosine Similarity between the Logit Vectors.
#     """
#     # Ensure correct device/dtype
#     target_device = unembedding_matrix.device
#     target_dtype = unembedding_matrix.dtype
    
#     # Clone to avoid modifying originals, move to target device
#     va = vec_a.to(device=target_device, dtype=target_dtype)
#     vb = vec_b.to(device=target_device, dtype=target_dtype)

#     # Project: (d_model) -> (vocab_size)
#     logit_vec_a = va @ unembedding_matrix
#     logit_vec_b = vb @ unembedding_matrix
    
#     # Compute Cosine Similarity in Vocab Space
#     sim = torch.nn.functional.cosine_similarity(
#         logit_vec_a.unsqueeze(0), 
#         logit_vec_b.unsqueeze(0)
#     )
#     return sim.item()


# def compute_pairwise_metrics(
#     steering_vectors: Dict[str, torch.Tensor],
#     model
# ) -> List[Dict]:
#     """Compute both standard cosine (residual) and projected cosine (vocab)."""
#     concepts = list(steering_vectors.keys())
#     pairs_data = []
    
#     # Get Unembedding Matrix
#     if hasattr(model, "lm_head"):
#         W_U = model.lm_head.weight.detach()
#         # Transpose if necessary (standard shape [vocab, hidden], we need [hidden, vocab] for v @ W)
#         if W_U.shape[0] != steering_vectors[concepts[0]].shape[0]:
#             W_U = W_U.T
#     elif hasattr(model, "get_output_embeddings"):
#         W_U = model.get_output_embeddings().weight.detach().T
#     else:
#         raise ValueError("Could not find Unembedding Matrix.")

#     print("Computing Vocabulary-Projected Similarities...")

#     for i, c1 in enumerate(concepts):
#         for j, c2 in enumerate(concepts):
#             if i < j: # Upper triangle only
#                 vec_a = steering_vectors[c1]
#                 vec_b = steering_vectors[c2]
                
#                 # 1. Standard Geometric Cosine (Residual Stream)
#                 geo_sim = float(compute_cosine_similarity(vec_a, vec_b))
                
#                 # 2. Semantic Cosine (Vocab Stream)
#                 vocab_sim = compute_vocab_projected_similarity(vec_a, vec_b, W_U)
                
#                 pairs_data.append({
#                     "concept_a": c1,
#                     "concept_b": c2,
#                     "geo_similarity": geo_sim,
#                     "abs_geo_similarity": abs(geo_sim),
#                     "vocab_similarity": vocab_sim,
#                     "abs_vocab_similarity": abs(vocab_sim)
#                 })
    
#     return pairs_data


# # ==========================================
# # 2. EXPERIMENT LOOP
# # ==========================================

# def test_composition(
#     model,
#     tokenizer,
#     steering_vectors: Dict[str, torch.Tensor],
#     pairs_data: List[Dict],
#     layer: int,
#     prompts: List[str],
#     n_generations: int = 3,
#     coefficient: float = 1.5
# ) -> List[Dict]:
    
#     evaluator_cache = {} 
#     results = []
    
#     # Filter synonyms using the VOCAB metric (Vocab Similarity > 0.85 = Synonym)
#     # This is smarter than residual cosine because synonyms naturally align in vocab space.
#     filtered_pairs = [p for p in pairs_data if p["abs_vocab_similarity"] < 0.85]
#     print(f"\nTesting {len(filtered_pairs)} pairs (Filtered out {len(pairs_data) - len(filtered_pairs)} synonyms)...")
    
#     for pair in tqdm(filtered_pairs, desc="Steering"):
#         c1, c2 = pair["concept_a"], pair["concept_b"]
        
#         # Lazy load evaluator on CPU
#         pair_key = tuple(sorted([c1, c2]))
#         if pair_key not in evaluator_cache:
#             try:
#                 evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2], device="cpu")
#             except TypeError:
#                 evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2])

#         evaluator = evaluator_cache[pair_key]
        
#         vec_a = steering_vectors[c1]
#         vec_b = steering_vectors[c2]
        
#         success_count = 0
#         total = 0
        
#         for prompt in prompts:
#             for _ in range(n_generations):
#                 # Apply both vectors
#                 config = [
#                     SteeringConfig(vector=vec_a, layer=layer, coefficient=coefficient),
#                     SteeringConfig(vector=vec_b, layer=layer, coefficient=coefficient)
#                 ]
#                 text = generate_with_steering(model, tokenizer, prompt, config)
                
#                 # Check success
#                 scores = evaluator.evaluate(text, [c1, c2])
#                 if scores[c1] > 0.5 and scores[c2] > 0.5:
#                     success_count += 1
#                 total += 1
        
#         pair["joint_success_rate"] = success_count / total
#         results.append(pair)
        
#     return results


# # ==========================================
# # 3. ANALYSIS & PLOTTING
# # ==========================================

# def safe_pearson(x, y):
#     if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
#         return 0.0, 1.0
#     return pearsonr(x, y)

# def analyze_and_plot(results: List[Dict], save_dir: Path):
#     df = pd.DataFrame(results)
    
#     if len(df) < 3:
#         print("Not enough data to analyze.")
#         return

#     # Correlations
#     r_geo, p_geo = safe_pearson(df["abs_geo_similarity"], df["joint_success_rate"])
#     r_vocab, p_vocab = safe_pearson(df["abs_vocab_similarity"], df["joint_success_rate"])
    
#     # --- SUMMARY CARD ---
#     print("\n" + "="*60)
#     print("EXPERIMENTAL SUMMARY CARD")
#     print("="*60)
#     print(f"Total Pairs: {len(df)}")
#     print(f"Mean Success Rate: {df['joint_success_rate'].mean():.1%}")
#     print("-" * 60)
#     print(f"{'METRIC':<25} | {'CORRELATION (r)':<15} | {'P-VALUE':<10}")
#     print("-" * 60)
#     print(f"{'Residual Geometry':<25} | {r_geo:<15.3f} | {p_geo:<10.3f}")
#     print(f"{'Vocab Projection':<25} | {r_vocab:<15.3f} | {p_vocab:<10.3f}")
#     print("-" * 60)
    
#     winner = "Residual Geometry" if abs(r_geo) > abs(r_vocab) else "Vocab Projection"
#     diff = abs(abs(r_geo) - abs(r_vocab))
    
#     print(f">> WINNER: {winner} (by {diff:.3f})")
#     if diff < 0.05:
#         print(">> NOTE: Difference is marginal. Both metrics perform similarly.")
#     elif winner == "Vocab Projection":
#         print(">> CONCLUSION: Output semantics predict failure better than internal geometry.")
#     else:
#         print(">> CONCLUSION: Internal geometry remains the dominant predictor.")
#     print("="*60)
        
#     # --- PLOTTING ---
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
#     # Helper to annotate points
#     def annotate_points(ax, x_col, y_col):
#         # Annotate outliers (high success/low sim OR low success/high sim)
#         for idx, row in df.iterrows():
#             label = f"{row['concept_a'][:3]}+{row['concept_b'][:3]}"
#             # Only label interesting points to avoid clutter
#             if row['joint_success_rate'] > 0.8 or row['joint_success_rate'] < 0.2:
#                 ax.text(row[x_col], row[y_col], label, fontsize=8, alpha=0.7)

#     # Plot 1: Residual Geometry
#     sns.regplot(x="abs_geo_similarity", y="joint_success_rate", data=df, ax=axes[0], 
#                 scatter_kws={'alpha':0.6, 's': 60}, line_kws={'color':'red', 'linewidth': 2})
#     annotate_points(axes[0], "abs_geo_similarity", "joint_success_rate")
#     axes[0].set_title(f"A. Residual Stream Geometry\n(Correlation: r={r_geo:.2f})", fontsize=14)
#     axes[0].set_xlabel("|Cosine Similarity| (Hidden Layer)", fontsize=12)
#     axes[0].set_ylabel("Joint Success Rate", fontsize=12)
#     axes[0].set_ylim(-0.05, 1.05)
    
#     # Plot 2: Vocab Geometry
#     sns.regplot(x="abs_vocab_similarity", y="joint_success_rate", data=df, ax=axes[1], 
#                 scatter_kws={'alpha':0.6, 's': 60, 'color': 'green'}, line_kws={'color':'blue', 'linewidth': 2})
#     annotate_points(axes[1], "abs_vocab_similarity", "joint_success_rate")
#     axes[1].set_title(f"B. Output Vocabulary Geometry\n(Correlation: r={r_vocab:.2f})", fontsize=14)
#     axes[1].set_xlabel("|Cosine Similarity| (Output Logits)", fontsize=12)
#     axes[1].set_ylabel("") 
#     axes[1].set_ylim(-0.05, 1.05)
    
#     plt.tight_layout()
#     plt.savefig(save_dir / "vocab_vs_geometry_annotated.png")
#     print(f"\n[Visuals] Saved annotated comparison plot to {save_dir / 'vocab_vs_geometry_annotated.png'}")


# def convert_to_native(obj):
#     """Helper for JSON serialization."""
#     if isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
#     if isinstance(obj, list): return [convert_to_native(i) for i in obj]
#     if isinstance(obj, (np.int64, np.int32)): return int(obj)
#     if isinstance(obj, (np.float64, np.float32)): return float(obj)
#     return obj


# # ==========================================
# # 4. MAIN
# # ==========================================

# def main():
#     import argparse
#     from transformers import AutoModelForCausalLM, AutoTokenizer
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default=cfg.model.name)
#     parser.add_argument("--week1_dir", default="outputs/week1")
#     parser.add_argument("--output_dir", default="outputs/vocab_analysis")
#     parser.add_argument("--quick", action="store_true")
#     parser.add_argument("--coef", type=float, default=1.5)
#     args = parser.parse_args()
    
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("Loading resources...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
#     with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
#         concepts = json.load(f)["concepts"]
        
#     vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
#     # Move vectors to correct device
#     vectors = {c: v[cfg.model.default_layer].to(model.device) for c, v in vectors_by_layer.items()}
    
#     # 1. Compute Metrics
#     pairs_data = compute_pairwise_metrics(vectors, model)
    
#     # 2. Run Test
#     prompts = get_test_prompts()[:2] if args.quick else get_test_prompts()[:5]
#     n_gen = 2 if args.quick else 3
    
#     results = test_composition(
#         model, tokenizer, vectors, pairs_data, 
#         cfg.model.default_layer, prompts, n_generations=n_gen, coefficient=args.coef
#     )
    
#     # 3. Analyze
#     analyze_and_plot(results, output_dir)
    
#     with open(output_dir / "vocab_results.json", "w") as f:
#         json.dump(convert_to_native(results), f, indent=2)

# if __name__ == "__main__":
#     main()

# """
# Centered Active Subspace Analysis.

# Improvements:
# 1. Centered Logits: Removes "common token" noise from the similarity metric.
# 2. Continuous Scoring: Uses classifier probability (not binary success) for a smoother trend.
# """

# import torch
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple
# from pathlib import Path
# import json
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import pearsonr

# from config import cfg
# from data.prompts import get_test_prompts
# from steering.apply_steering import SteeringConfig, generate_with_steering
# from evaluation.classifiers import MultiAttributeEvaluator
# from extraction.extract_vectors import load_cached_vectors


# def compute_centered_subspace_similarity(
#     vec_a: torch.Tensor,
#     vec_b: torch.Tensor, 
#     unembedding_matrix: torch.Tensor,
#     top_k: int = 1000
# ) -> float:
#     """
#     Compute Cosine Similarity on Top-K logits AFTER removing the mean logit.
#     This filters out common words ("the", "and") that dominate the direction.
#     """
#     target_device = unembedding_matrix.device
#     target_dtype = unembedding_matrix.dtype
    
#     va = vec_a.to(device=target_device, dtype=target_dtype)
#     vb = vec_b.to(device=target_device, dtype=target_dtype)
    
#     # 1. Project
#     logits_a = va @ unembedding_matrix
#     logits_b = vb @ unembedding_matrix
    
#     # 2. CENTER the logits (The Fix)
#     # Subtracting the mean removes the "background direction" of the vocabulary
#     logits_a = logits_a - logits_a.mean()
#     logits_b = logits_b - logits_b.mean()
    
#     # 3. Identify Active Subspace on Centered Logits
#     top_indices_a = torch.topk(logits_a.abs(), top_k).indices
#     top_indices_b = torch.topk(logits_b.abs(), top_k).indices
#     active_mask = torch.unique(torch.cat([top_indices_a, top_indices_b]))
    
#     # 4. Compute Signed Cosine on this subspace
#     filtered_a = logits_a[active_mask]
#     filtered_b = logits_b[active_mask]
    
#     sim = torch.nn.functional.cosine_similarity(
#         filtered_a.unsqueeze(0), 
#         filtered_b.unsqueeze(0)
#     )
#     return sim.item()


# def compute_metrics(steering_vectors: Dict[str, torch.Tensor], model) -> List[Dict]:
#     concepts = list(steering_vectors.keys())
#     pairs_data = []
    
#     if hasattr(model, "lm_head"):
#         W_U = model.lm_head.weight.detach()
#         if W_U.shape[0] != steering_vectors[concepts[0]].shape[0]: W_U = W_U.T
#     elif hasattr(model, "get_output_embeddings"):
#         W_U = model.get_output_embeddings().weight.detach().T
#     else:
#         raise ValueError("No Unembedding Matrix found.")

#     print("Computing Centered Subspace Metrics...")

#     for i, c1 in enumerate(concepts):
#         for j, c2 in enumerate(concepts):
#             if i < j:
#                 vec_a = steering_vectors[c1]
#                 vec_b = steering_vectors[c2]
                
#                 # Compare Centered vs Raw to see which works better
#                 sim_centered = compute_centered_subspace_similarity(vec_a, vec_b, W_U, top_k=1000)
                
#                 pairs_data.append({
#                     "concept_a": c1,
#                     "concept_b": c2,
#                     "sim_centered": sim_centered,
#                 })
#     return pairs_data


# def test_composition(
#     model, tokenizer, steering_vectors, pairs_data, layer, prompts, n_gen=3, coef=2.0
# ) -> List[Dict]:
    
#     # Filter Synonyms
#     filtered_pairs = [p for p in pairs_data if p["sim_centered"] < 0.85]
#     print(f"\nTesting {len(filtered_pairs)} pairs (Coefficient: {coef})...")
    
#     evaluator_cache = {}
#     results = []
    
#     for pair in tqdm(filtered_pairs):
#         c1, c2 = pair["concept_a"], pair["concept_b"]
#         pair_key = tuple(sorted([c1, c2]))
        
#         if pair_key not in evaluator_cache:
#             try:
#                 evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2], device="cpu")
#             except TypeError:
#                 evaluator_cache[pair_key] = MultiAttributeEvaluator([c1, c2])
        
#         vec_a = steering_vectors[c1]
#         vec_b = steering_vectors[c2]
        
#         # CONTINUOUS SCORING ACCUMULATORS
#         total_joint_prob = 0.0
#         count = 0
        
#         for prompt in prompts:
#             for _ in range(n_gen):
#                 config = [
#                     SteeringConfig(vector=vec_a, layer=layer, coefficient=coef),
#                     SteeringConfig(vector=vec_b, layer=layer, coefficient=coef)
#                 ]
#                 text = generate_with_steering(model, tokenizer, prompt, config)
                
#                 # Get raw probabilities (continuous) instead of thresholded boolean
#                 scores = evaluator_cache[pair_key].evaluate(text, [c1, c2])
                
#                 # Joint Probability = P(A) * P(B)
#                 # This captures "marginal success" much better than binary checks
#                 joint_prob = scores[c1] * scores[c2]
#                 total_joint_prob += joint_prob
#                 count += 1
        
#         pair["joint_score"] = total_joint_prob / count
#         results.append(pair)
        
#     return results


# def analyze_and_plot(results: List[Dict], save_dir: Path):
#     df = pd.DataFrame(results)
#     if len(df) < 5: return

#     # Check correlation
#     r, p = pearsonr(df["sim_centered"], df["joint_score"])

#     print("\n" + "="*60)
#     print("CENTERED SUBSPACE ANALYSIS")
#     print("="*60)
#     print(f"Metric: Centered Logit Cosine (Top-1000) vs. Joint Probability")
#     print("-" * 60)
#     print(f"Correlation (r): {r:.3f}")
#     print(f"P-Value (p):     {p:.3f}")
#     print("-" * 60)
    
#     if abs(r) > 0.3:
#         print(">> RESULT: Clearer trend detected!")
#     else:
#         print(">> RESULT: Still noisy. Interaction is likely highly non-linear.")
#     print("="*60)

#     # Plot
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(10, 6))
    
#     # Scatter with regression
#     sns.regplot(x="sim_centered", y="joint_score", data=df, 
#                 scatter_kws={'alpha':0.6, 's':60}, line_kws={'color':'red'})
    
#     # Annotate significant points
#     for idx, row in df.iterrows():
#         if row['joint_score'] > 0.6 or row['joint_score'] < 0.1:
#             plt.text(row["sim_centered"], row['joint_score'], 
#                      f"{row['concept_a'][:3]}+{row['concept_b'][:3]}", fontsize=9, alpha=0.7)

#     plt.title(f"Centered Logit Similarity vs. Joint Score\n(Correlation r={r:.2f})")
#     plt.xlabel("Signed Centered Cosine Similarity (Union of Top-1000 Tokens)")
#     plt.ylabel("Joint Probability Score (P(A) * P(B))")
#     plt.ylim(-0.05, 1.05)
#     plt.xlim(-1.0, 1.0)
#     plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(save_dir / "centered_trend.png")
#     print(f"Saved plot to {save_dir}")

# def convert_to_native(obj):
#     if isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
#     if isinstance(obj, list): return [convert_to_native(i) for i in obj]
#     if isinstance(obj, (np.int64, np.int32)): return int(obj)
#     if isinstance(obj, (np.float64, np.float32)): return float(obj)
#     return obj

# def main():
#     import argparse
#     from transformers import AutoModelForCausalLM, AutoTokenizer
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default=cfg.model.name)
#     parser.add_argument("--week1_dir", default="outputs/week1")
#     parser.add_argument("--output_dir", default="outputs/centered_analysis")
#     parser.add_argument("--quick", action="store_true")
#     parser.add_argument("--coef", type=float, default=2.0)
#     args = parser.parse_args()
    
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("Loading resources...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
#     with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
#         concepts = json.load(f)["concepts"]
        
#     vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
#     vectors = {c: v[cfg.model.default_layer].to(model.device) for c, v in vectors_by_layer.items()}
    
#     # 1. Compute Centered Metrics
#     pairs_data = compute_metrics(vectors, model)
    
#     # 2. Run Test with Continuous Scoring
#     prompts = get_test_prompts()[:2] if args.quick else get_test_prompts()[:5]
#     n_gen = 2 if args.quick else 3
    
#     results = test_composition(
#         model, tokenizer, vectors, pairs_data, 
#         cfg.model.default_layer, prompts, n_gen, args.coef
#     )
    
#     # 3. Analyze
#     analyze_and_plot(results, output_dir)
    
#     with open(output_dir / "centered_results.json", "w") as f:
#         json.dump(convert_to_native(results), f, indent=2)

# if __name__ == "__main__":
#     main()


"""
Comprehensive Orthogonality Experiment.
Updated with Centering AND Whitening options.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from itertools import combinations
import scipy.stats as stats

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import SteeringConfig, generate_with_steering
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.geometry import compute_cosine_similarity
from extraction.extract_vectors import load_cached_vectors

def test_pair_composition(
    model, 
    tokenizer, 
    vec_a_steer: torch.Tensor, # Raw vector for steering
    vec_b_steer: torch.Tensor, # Raw vector for steering
    vec_a_geom: torch.Tensor,  # Processed vector for measuring angle
    vec_b_geom: torch.Tensor,  # Processed vector for measuring angle
    concept_a: str, 
    concept_b: str,
    layer: int,
    prompts: List[str],
    evaluator: MultiAttributeEvaluator
) -> Dict:
    """
    Test composition. Uses Raw vectors for steering, Processed for geometry.
    """
    # 1. Geometry (Measure on Whitened/Centered Vectors)
    similarity = compute_cosine_similarity(vec_a_geom, vec_b_geom)
    
    # 2. Arithmetic (Steer with Original Vectors)
    vec_comb = vec_a_steer + vec_b_steer
    
    scores_comb_a = []
    scores_comb_b = []
    scores_a_only = []
    scores_b_only = []
    
    for p in prompts:
        # A Only
        text_a = generate_with_steering(
            model, tokenizer, p, 
            SteeringConfig(vec_a_steer, layer, cfg.model.default_coefficient)
        )
        scores_a_only.append(evaluator.classifiers[concept_a].score(text_a))

        # B Only
        text_b = generate_with_steering(
            model, tokenizer, p, 
            SteeringConfig(vec_b_steer, layer, cfg.model.default_coefficient)
        )
        scores_b_only.append(evaluator.classifiers[concept_b].score(text_b))

        # Combined
        text_comb = generate_with_steering(
            model, tokenizer, p, 
            SteeringConfig(vec_comb, layer, cfg.model.default_coefficient)
        )
        res = evaluator.evaluate(text_comb, [concept_a, concept_b])
        scores_comb_a.append(res[concept_a])
        scores_comb_b.append(res[concept_b])

    # Metrics
    mean_a_only = np.mean(scores_a_only)
    mean_b_only = np.mean(scores_b_only)
    mean_comb_a = np.mean(scores_comb_a)
    mean_comb_b = np.mean(scores_comb_b)
    
    # Interference
    int_a = max(0, (mean_a_only - mean_comb_a) / mean_a_only) if mean_a_only > 0.1 else 0
    int_b = max(0, (mean_b_only - mean_comb_b) / mean_b_only) if mean_b_only > 0.1 else 0
    avg_interference = (int_a + int_b) / 2.0
    
    joint_success = np.sqrt(mean_comb_a * mean_comb_b)

    return {
        "pair": f"{concept_a}+{concept_b}",
        "similarity": float(similarity),
        "abs_similarity": abs(float(similarity)),
        "interference": float(avg_interference),
        "joint_success": float(joint_success)
    }

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import PCA # New import
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--output_dir", default="outputs/orthogonality")
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--center", action="store_true", help="Center vectors (Mean Subtraction)")
    parser.add_argument("--whiten", action="store_true", help="Whiten vectors (PCA + Variance Scaling)")
    args = parser.parse_args()
    
    # Force whiten if user asks (Strongest scaling)
    if args.whiten:
        print("(!) Scaling Mode: Whitening (PCA)")
    elif args.center:
        print("(!) Scaling Mode: Centering")
    else:
        print("(!) Scaling Mode: None (Raw)")
    
    w1_dir = Path(args.week1_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(w1_dir / "validation_summary.json") as f:
        summary = json.load(f)
        layer = summary.get("common_layer", cfg.model.default_layer)
    
    print("="*60)
    print(f"COMPREHENSIVE ORTHOGONALITY TEST")
    print("="*60)

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16 if cfg.model.torch_dtype=="float16" else torch.bfloat16, 
        device_map="auto"
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load Raw Vectors
    vectors_map = load_cached_vectors(w1_dir / "vectors", cfg.concepts, [layer])
    raw_vectors = {c: vectors_map[c][layer].float() for c in cfg.concepts if c in vectors_map and layer in vectors_map[c]}
    concepts = list(raw_vectors.keys())
    
    # --- GEOMETRY PROCESSING ---
    geom_vectors = raw_vectors # Default
    
    if args.whiten:
        print("Applying PCA Whitening to vectors...")
        # Stack vectors: (N_concepts, Hidden_dim)
        vec_stack = torch.stack([raw_vectors[c] for c in concepts]).cpu().numpy()
        
        # PCA Whitening
        # We can only keep components <= number of samples (concepts)
        n_comps = min(len(concepts), vec_stack.shape[1])
        pca = PCA(n_components=n_comps, whiten=True)
        whitened_stack = pca.fit_transform(vec_stack)
        
        # Convert back to dict
        geom_vectors = {
            c: torch.tensor(whitened_stack[i]).to(model.device).float()
            for i, c in enumerate(concepts)
        }
        
    elif args.center:
        print("Applying Centering (Mean Subtraction) to vectors...")
        vec_stack = torch.stack([raw_vectors[c] for c in concepts])
        mean_vec = vec_stack.mean(dim=0)
        geom_vectors = {c: v - mean_vec for c, v in raw_vectors.items()}

    prompts = get_test_prompts()[:args.num_prompts]
    evaluator = MultiAttributeEvaluator(concepts)
    
    results = []
    pairs = list(combinations(concepts, 2))
    
    for c_a, c_b in tqdm(pairs, desc="Testing Pairs"):
        try:
            res = test_pair_composition(
                model, tokenizer, 
                vec_a_steer=raw_vectors[c_a], # Always steer with RAW
                vec_b_steer=raw_vectors[c_b], 
                vec_a_geom=geom_vectors[c_a], # Measure angle with PROCESSED
                vec_b_geom=geom_vectors[c_b],
                concept_a=c_a, concept_b=c_b,
                layer=layer, prompts=prompts, evaluator=evaluator
            )
            results.append(res)
        except Exception as e:
            print(f"Error {c_a}+{c_b}: {e}")

    # Analysis
    sims = [r["abs_similarity"] for r in results]
    intfs = [r["interference"] for r in results]
    succs = [r["joint_success"] for r in results]
    
    corr_int, p_int = stats.pearsonr(sims, intfs)
    corr_succ, p_succ = stats.pearsonr(sims, succs)
    
    # Save
    summary = {
        "correlations": {
            "similarity_vs_interference": {"r": corr_int, "p": p_int},
            "similarity_vs_success": {"r": corr_succ, "p": p_succ}
        },
        "pair_data": results
    }
    
    with open(out_dir / "orthogonality_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- SUMMARY CARD ---
    most_aligned = sorted(results, key=lambda x: x["abs_similarity"], reverse=True)[:5]
    most_interfering = sorted(results, key=lambda x: x["interference"], reverse=True)[:5]
    
    print("\n" + "="*60)
    print(f"{'SUMMARY CARD':^60}")
    print("="*60)
    print(f"  > Sim vs Interference: r = {corr_int:+.3f} (p={p_int:.4f})")
    print("-" * 60)
    print("  [Highest Similarity Pairs] (Right Side of Graph)")
    for r in most_aligned:
        print(f"    • {r['pair']:<25} | Sim: {r['abs_similarity']:.2f} | Intf: {r['interference']:.2f}")
    print("-" * 60)
    print("  [Highest Interference Pairs] (Top of Graph)")
    for r in most_interfering:
        print(f"    • {r['pair']:<25} | Sim: {r['abs_similarity']:.2f} | Intf: {r['interference']:.2f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()