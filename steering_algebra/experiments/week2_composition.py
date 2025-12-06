"""
Week 2 Experiment: Test steering vector composition.

Core questions:
1. Does A + B produce text with both attributes A and B?
2. Do opposing vectors cancel (A + (-A) ≈ baseline)?
3. Does vector arithmetic work (A + B - A ≈ B)?
4. Does geometric similarity predict composition success?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass
import ast

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import (
    SteeringConfig,
    generate_with_steering,
    generate_baseline
)
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics
from evaluation.geometry import (
    compute_cosine_similarity,
    predict_interference
)
from extraction.extract_vectors import load_cached_vectors
from steering.layer_selection import find_optimal_layers_batch





@dataclass
class CompositionResult:
    """Results from a single composition experiment."""
    concept_a: str
    concept_b: str
    layer_a: int
    layer_b: int
    coefficient_a: float
    coefficient_b: float
    
    # Scores
    baseline_scores: Dict[str, float]
    a_only_scores: Dict[str, float]
    b_only_scores: Dict[str, float]
    composition_scores: Dict[str, float]
    
    # Quality
    baseline_perplexity: float
    composition_perplexity: float
    
    # Geometry
    cosine_similarity: float
    
    # Success metrics
    both_present: bool  # Are both A and B present in composition?
    composition_success_rate: float


def parse_concept_pairs(geometry: Dict, category: str) -> List[Tuple[str, str]]:
    """
    Robustly parse concept pairs from geometry analysis.
    Handles multiple possible formats.
    
    Args:
        geometry: Geometry analysis dict from Week 1
        category: "orthogonal", "opposing", or "aligned"
    
    Returns:
        List of (concept_a, concept_b) tuples
    """
    pairs = []
    
    if category not in geometry["categories"]:
        return pairs
    
    items = geometry["categories"][category]
    print("ITEMS", items)
    
    pairs = [(t[0], t[1]) for t in (ast.literal_eval(x) for x in items)]
    
    return pairs
    

def test_additive_composition(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    optimal_layers: Dict[str, int],
    optimal_coefficients: Dict[str, float],
    prompts: List[str],
    n_generations: int = 5,
    threshold: float = 0.5
) -> CompositionResult:
    """
    Test if A + B produces text with both attributes.
    
    H1: composition_score(A) > threshold AND composition_score(B) > threshold
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    quality_metrics = QualityMetrics()
    
    vec_a = steering_vectors[concept_a]
    vec_b = steering_vectors[concept_b]
    
    # Use optimal layers and coefficients
    layer_a = optimal_layers[concept_a]
    layer_b = optimal_layers[concept_b]
    coef_a = optimal_coefficients.get(concept_a, cfg.model.default_coefficient)
    coef_b = optimal_coefficients.get(concept_b, cfg.model.default_coefficient)
    
    
    # Storage for results
    baseline_texts = []
    a_only_texts = []
    b_only_texts = []
    composition_texts = []
    
    baseline_scores_a = []
    baseline_scores_b = []
    a_only_scores_a = []
    a_only_scores_b = []
    b_only_scores_a = []
    b_only_scores_b = []
    composition_scores_a = []
    composition_scores_b = []
    
    print(f"\nTesting: {concept_a} (layer {layer_a}, coef {coef_a:.2f}) + {concept_b} (layer {layer_b}, coef {coef_b:.2f})")
    print(f"Cosine similarity: {compute_cosine_similarity(vec_a, vec_b):.3f}")
    
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # Baseline
            baseline = generate_baseline(model, tokenizer, prompt)
            baseline_texts.append(baseline)
            scores = evaluator.evaluate(baseline, [concept_a, concept_b])
            baseline_scores_a.append(scores[concept_a])
            baseline_scores_b.append(scores[concept_b])
            
            # A only
            config_a = SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coef_a)
            text_a = generate_with_steering(model, tokenizer, prompt, config_a)
            a_only_texts.append(text_a)
            scores = evaluator.evaluate(text_a, [concept_a, concept_b])
            a_only_scores_a.append(scores[concept_a])
            a_only_scores_b.append(scores[concept_b])
            
            # B only
            config_b = SteeringConfig(vector=vec_b, layer=layer_b, coefficient=coef_b)
            text_b = generate_with_steering(model, tokenizer, prompt, config_b)
            b_only_texts.append(text_b)
            scores = evaluator.evaluate(text_b, [concept_a, concept_b])
            b_only_scores_a.append(scores[concept_a])
            b_only_scores_b.append(scores[concept_b])
            
            # A + B composition (each at their optimal layer)
            config_ab = [
                SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coef_a),
                SteeringConfig(vector=vec_b, layer=layer_b, coefficient=coef_b)
            ]
            text_ab = generate_with_steering(model, tokenizer, prompt, config_ab)
            composition_texts.append(text_ab)
            scores = evaluator.evaluate(text_ab, [concept_a, concept_b])
            composition_scores_a.append(scores[concept_a])
            composition_scores_b.append(scores[concept_b])
    
    # Compute success metrics
    both_present_count = sum(
        1 for sa, sb in zip(composition_scores_a, composition_scores_b)
        if sa > threshold and sb > threshold
    )
    both_present = both_present_count > len(composition_texts) * 0.5  # 60% threshold
    composition_success_rate = both_present_count / len(composition_texts)
    
    # Quality metrics
    baseline_ppls = quality_metrics.perplexity_calc.compute_batch(baseline_texts)
    composition_ppls = quality_metrics.perplexity_calc.compute_batch(composition_texts)
    
    result = CompositionResult(
        concept_a=concept_a,
        concept_b=concept_b,
        layer_a=layer_a,
        layer_b=layer_b,
        coefficient_a=coef_a,
        coefficient_b=coef_b,
        baseline_scores={
            concept_a: float(np.mean(baseline_scores_a)),
            concept_b: float(np.mean(baseline_scores_b))
        },
        a_only_scores={
            concept_a: float(np.mean(a_only_scores_a)),
            concept_b: float(np.mean(a_only_scores_b))
        },
        b_only_scores={
            concept_a: float(np.mean(b_only_scores_a)),
            concept_b: float(np.mean(b_only_scores_b))
        },
        composition_scores={
            concept_a: float(np.mean(composition_scores_a)),
            concept_b: float(np.mean(composition_scores_b))
        },
        baseline_perplexity=float(np.mean(baseline_ppls)),
        composition_perplexity=float(np.mean(composition_ppls)),
        cosine_similarity=float(compute_cosine_similarity(vec_a, vec_b)),
        both_present=both_present,
        composition_success_rate=float(composition_success_rate)
    )
    
    
    print(f"\nResults:")
    print(f"  Baseline:    {concept_a}={result.baseline_scores[concept_a]:.3f}, {concept_b}={result.baseline_scores[concept_b]:.3f}")
    print(f"  A only:      {concept_a}={result.a_only_scores[concept_a]:.3f}, {concept_b}={result.a_only_scores[concept_b]:.3f}")
    print(f"  B only:      {concept_a}={result.b_only_scores[concept_a]:.3f}, {concept_b}={result.b_only_scores[concept_b]:.3f}")
    print(f"  A + B:       {concept_a}={result.composition_scores[concept_a]:.3f}, {concept_b}={result.composition_scores[concept_b]:.3f}")
    print(f"  Both present: {result.both_present} ({result.composition_success_rate:.1%})")
    print(f"  Perplexity Δ: {result.composition_perplexity - result.baseline_perplexity:+.1f}")
    
    return result


def test_opposing_composition(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    optimal_layers: Dict[str, int],
    optimal_coefficients: Dict[str, float],
    prompts: List[str],
    n_generations: int = 5
) -> Dict:
    """
    Test if A + (-A) cancels out to produce baseline-like text.
    
    H2: opposing vectors should cancel or create noise
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    quality_metrics = QualityMetrics()
    
    vec_a = steering_vectors[concept_a]
    vec_b = steering_vectors[concept_b]
    
    vec_a = steering_vectors[concept_a]
    vec_b = steering_vectors[concept_b]
    
    # Use optimal layers and coefficients
    layer_a = optimal_layers[concept_a]
    layer_b = optimal_layers[concept_b]
    coef_a = optimal_coefficients.get(concept_a, cfg.model.default_coefficient)
    coef_b = optimal_coefficients.get(concept_b, cfg.model.default_coefficient)
    
    baseline_texts = []
    a_only_texts = []
    b_only_texts = []
    composition_texts = []
    
    baseline_scores = []
    a_only_scores = []
    b_only_scores = []
    composition_scores = []
    
    print(f"\nTesting opposing: {concept_a} (layer {layer_a}, coef {coef_a:.2f}) + {concept_b} (layer {layer_b}, coef {coef_b:.2f})")
    print(f"Cosine similarity: {compute_cosine_similarity(vec_a, vec_b):.3f}")
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # Baseline
            baseline = generate_baseline(model, tokenizer, prompt)
            baseline_texts.append(baseline)
            baseline_scores.append(evaluator.classifiers[concept_a].score(baseline))
            
            # A only
            config_a = SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coef_a)
            text_a = generate_with_steering(model, tokenizer, prompt, config_a)
            a_only_texts.append(text_a)
            a_only_scores.append(evaluator.classifiers[concept_a].score(text_a))
            
            # B only (opposite)
            config_b = SteeringConfig(vector=vec_b, layer=layer_b, coefficient=coef_b)
            text_b = generate_with_steering(model, tokenizer, prompt, config_b)
            b_only_texts.append(text_b)
            b_only_scores.append(evaluator.classifiers[concept_a].score(text_b))
            
            # A + B composition
            config_ab = [
                SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coef_a),
                SteeringConfig(vector=vec_b, layer=layer_b, coefficient=coef_b)
            ]
            text_ab = generate_with_steering(model, tokenizer, prompt, config_ab)
            composition_texts.append(text_ab)
            composition_scores.append(evaluator.classifiers[concept_a].score(text_ab))
    
    # Analysis
    baseline_mean = np.mean(baseline_scores)
    a_only_mean = np.mean(a_only_scores)
    b_only_mean = np.mean(b_only_scores)
    composition_mean = np.mean(composition_scores)
    
    # Check if composition is closer to baseline than to A or B
    distance_to_baseline = abs(composition_mean - baseline_mean)
    distance_to_a = abs(composition_mean - a_only_mean)
    distance_to_b = abs(composition_mean - b_only_mean)
    
    cancels_out = distance_to_baseline < min(distance_to_a, distance_to_b)
    
    # Quality
    baseline_ppls = quality_metrics.perplexity_calc.compute_batch(baseline_texts)
    composition_ppls = quality_metrics.perplexity_calc.compute_batch(composition_texts)
    
    result = {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "coefficient_a": coef_a,
        "coefficient_b": coef_b,
        "cosine_similarity": float(compute_cosine_similarity(vec_a, vec_b)),
        "baseline_mean": float(baseline_mean),
        "a_only_mean": float(a_only_mean),
        "b_only_mean": float(b_only_mean),
        "composition_mean": float(composition_mean),
        "distance_to_baseline": float(distance_to_baseline),
        "distance_to_a": float(distance_to_a),
        "distance_to_b": float(distance_to_b),
        "cancels_out": cancels_out,
        "baseline_perplexity": float(np.mean(baseline_ppls)),
        "composition_perplexity": float(np.mean(composition_ppls))
    }
    
    print(f"\nResults:")
    print(f"  Baseline:     {baseline_mean:.3f}")
    print(f"  A only:       {a_only_mean:.3f}")
    print(f"  B only:       {b_only_mean:.3f}")
    print(f"  A + B:        {composition_mean:.3f}")
    print(f"  Distance to baseline: {distance_to_baseline:.3f}")
    print(f"  Cancels out:  {cancels_out}")
    
    return result


def test_arithmetic_composition(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    optimal_layers: Dict[str, int],
    optimal_coefficients: Dict[str, float],
    prompts: List[str],
    n_generations: int = 5
) -> Dict:
    """
    Test if steering vector arithmetic works semantically like word embeddings.
    
    H3: Vector arithmetic should compose behaviors:
    - A + B should produce BOTH behaviors
    - A + B should score high on both A and B
    - (A + B) effectiveness ≈ individual A and B effectiveness combined
    
    This tests BEHAVIORAL composition, not trivial vector math.
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    
    vec_a = steering_vectors[concept_a].cpu()
    vec_b = steering_vectors[concept_b].cpu()
    
    # Use same layer for fair comparison (average of optimal layers)
    layer_a = optimal_layers[concept_a]
    layer_b = optimal_layers[concept_b]
    layer_combined = (layer_a + layer_b) // 2
    
    coef_a = optimal_coefficients.get(concept_a, cfg.model.default_coefficient)
    coef_b = optimal_coefficients.get(concept_b, cfg.model.default_coefficient)
    coef_combined = (coef_a + coef_b) / 2  # Average coefficient
    
    # Storage for all conditions
    baseline_a_scores = []
    baseline_b_scores = []
    a_only_a_scores = []
    a_only_b_scores = []
    b_only_a_scores = []
    b_only_b_scores = []
    combined_a_scores = []
    combined_b_scores = []
    
    print(f"\n{'='*60}")
    print(f"Testing Arithmetic: {concept_a} + {concept_b}")
    print(f"{'='*60}")
    print(f"Layer A: {layer_a}, Layer B: {layer_b}, Combined: {layer_combined}")
    print(f"Coef A: {coef_a:.2f}, Coef B: {coef_b:.2f}, Combined: {coef_combined:.2f}")
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # Baseline: no steering
            text_baseline = generate_with_steering(
                model, tokenizer, prompt, 
                SteeringConfig(vector=None, layer=0, coefficient=0)
            )
            baseline_a_scores.append(evaluator.classifiers[concept_a].score(text_baseline))
            baseline_b_scores.append(evaluator.classifiers[concept_b].score(text_baseline))
            
            # Condition 1: A only
            config_a = SteeringConfig(
                vector=vec_a.to(model.device), 
                layer=layer_combined,  # Use same layer for fair comparison
                coefficient=coef_a
            )
            text_a = generate_with_steering(model, tokenizer, prompt, config_a)
            a_only_a_scores.append(evaluator.classifiers[concept_a].score(text_a))
            a_only_b_scores.append(evaluator.classifiers[concept_b].score(text_a))
            
            # Condition 2: B only
            config_b = SteeringConfig(
                vector=vec_b.to(model.device), 
                layer=layer_combined,
                coefficient=coef_b
            )
            text_b = generate_with_steering(model, tokenizer, prompt, config_b)
            b_only_a_scores.append(evaluator.classifiers[concept_a].score(text_b))
            b_only_b_scores.append(evaluator.classifiers[concept_b].score(text_b))
            
            # Condition 3: A + B (the actual test!)
            # This is where we test if vectors compose semantically
            combined_vec = (vec_a + vec_b).to(model.device)
            config_combined = SteeringConfig(
                vector=combined_vec,
                layer=layer_combined,
                coefficient=coef_combined
            )
            text_combined = generate_with_steering(model, tokenizer, prompt, config_combined)
            combined_a_scores.append(evaluator.classifiers[concept_a].score(text_combined))
            combined_b_scores.append(evaluator.classifiers[concept_b].score(text_combined))
    
    # Analysis
    baseline_a = np.mean(baseline_a_scores)
    baseline_b = np.mean(baseline_b_scores)
    
    a_only_a = np.mean(a_only_a_scores)
    a_only_b = np.mean(a_only_b_scores)
    
    b_only_a = np.mean(b_only_a_scores)
    b_only_b = np.mean(b_only_b_scores)
    
    combined_a = np.mean(combined_a_scores)
    combined_b = np.mean(combined_b_scores)
    
    # Key metrics for composition
    # 1. Does A+B preserve A's effect?
    a_preservation = combined_a / a_only_a if a_only_a > 0 else 0
    
    # 2. Does A+B preserve B's effect?
    b_preservation = combined_b / b_only_b if b_only_b > 0 else 0
    
    # 3. Does A+B show BOTH behaviors?
    both_behaviors = (combined_a > baseline_a) and (combined_b > baseline_b)
    
    # 4. Compositional score: how well does A+B capture both?
    # Ideal: combined_a ≈ a_only_a AND combined_b ≈ b_only_b
    compositional_score = (a_preservation + b_preservation) / 2
    
    # 5. Check for interference (does adding B hurt A, or vice versa?)
    a_interference = (a_only_a - combined_a) / a_only_a if a_only_a > 0 else 0
    b_interference = (b_only_b - combined_b) / b_only_b if b_only_b > 0 else 0
    
    result = {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "layer": layer_combined,
        "coefficient": coef_combined,
        
        # Baseline
        "baseline_a": float(baseline_a),
        "baseline_b": float(baseline_b),
        
        # Individual effects
        "a_only_a": float(a_only_a),
        "a_only_b": float(a_only_b),
        "b_only_a": float(b_only_a),
        "b_only_b": float(b_only_b),
        
        # Combined effects
        "combined_a": float(combined_a),
        "combined_b": float(combined_b),
        
        # Composition metrics
        "a_preservation": float(a_preservation),
        "b_preservation": float(b_preservation),
        "compositional_score": float(compositional_score),
        "both_behaviors_present": both_behaviors,
        
        # Interference metrics
        "a_interference": float(a_interference),
        "b_interference": float(b_interference),
        
        # Success criteria
        "arithmetic_works": compositional_score > 0.7 and both_behaviors
    }
    
    # Pretty print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nBaseline (no steering):")
    print(f"  {concept_a}: {baseline_a:.3f}")
    print(f"  {concept_b}: {baseline_b:.3f}")
    
    print(f"\n{concept_a} only:")
    print(f"  {concept_a} score: {a_only_a:.3f}")
    print(f"  {concept_b} score: {a_only_b:.3f}")
    
    print(f"\n{concept_b} only:")
    print(f"  {concept_a} score: {b_only_a:.3f}")
    print(f"  {concept_b} score: {b_only_b:.3f}")
    
    print(f"\n{concept_a} + {concept_b} (Combined):")
    print(f"  {concept_a} score: {combined_a:.3f}")
    print(f"  {concept_b} score: {combined_b:.3f}")
    
    print(f"\nComposition Analysis:")
    print(f"  {concept_a} preservation: {a_preservation:.1%}")
    print(f"  {concept_b} preservation: {b_preservation:.1%}")
    print(f"  Compositional score: {compositional_score:.3f}")
    print(f"  Both behaviors present: {both_behaviors}")
    
    print(f"\nInterference:")
    print(f"  {concept_a} interference: {a_interference:.1%}")
    print(f"  {concept_b} interference: {b_interference:.1%}")
    
    print(f"\n{'='*60}")
    print(f"Arithmetic works: {result['arithmetic_works']}")
    print(f"{'='*60}\n")
    
    return result

def test_coefficient_scaling(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    optimal_layers: Dict[str, int],
    prompts: List[str],
    alpha_range: List[float] = [0.25, 0.5, 1.0, 1.5, 2.0],
    beta_range: List[float] = [0.25, 0.5, 1.0, 1.5, 2.0],
    n_generations: int = 3
) -> Dict:
    """
    Test how relative coefficient magnitude affects dominance.
    
    H4: Larger coefficient should dominate the composition
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    quality_metrics = QualityMetrics()
    
    vec_a = steering_vectors[concept_a]
    vec_b = steering_vectors[concept_b]

    layer_a = optimal_layers[concept_a]
    layer_b = optimal_layers[concept_b]
    
    results_grid = {}
    
    print(f"\nTesting coefficient scaling: {concept_a} (layer {layer_a}) × α + {concept_b} (layer {layer_b}) × β")

    
    for alpha in tqdm(alpha_range, desc="Alpha"):
        for beta in beta_range:
            key = f"a{alpha}_b{beta}"
            scores_a = []
            scores_b = []
            ppls = []
            
            for prompt in prompts:  # Use fewer prompts for speed
                for _ in range(n_generations):
                    config = [
                        SteeringConfig(vector=vec_a, layer=layer_a, coefficient=alpha),
                        SteeringConfig(vector=vec_b, layer=layer_b, coefficient=beta)
                    ]
                    text = generate_with_steering(model, tokenizer, prompt, config)
                    
                    scores = evaluator.evaluate(text, [concept_a, concept_b])
                    scores_a.append(scores[concept_a])
                    scores_b.append(scores[concept_b])
                    
                    ppl = quality_metrics.perplexity_calc.compute(text)
                    ppls.append(ppl)
            
            results_grid[key] = {
                "alpha": alpha,
                "beta": beta,
                "score_a": float(np.mean(scores_a)),
                "score_b": float(np.mean(scores_b)),
                "perplexity": float(np.mean(ppls)),
                "ratio": alpha / beta if beta > 0 else float('inf')
            }

    return {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "results_grid": results_grid
    }



def analyze_composition_vs_geometry(
    composition_results: List[CompositionResult]
) -> Dict:
    """
    Analyze correlation between geometric similarity and composition success.
    
    Do orthogonal vectors compose better than aligned/opposing ones?
    """
    similarities = []
    success_rates = []
    
    for result in composition_results:
        similarities.append(abs(result.cosine_similarity))
        success_rates.append(result.composition_success_rate)
    
    if len(similarities) < 2:
        return {"note": "Need at least 2 results for correlation"}
    
    correlation = np.corrcoef(similarities, success_rates)[0, 1]
    
    # Categorize by similarity
    orthogonal = [r for r in composition_results if abs(r.cosine_similarity) < 0.2]
    aligned = [r for r in composition_results if r.cosine_similarity > 0.5]
    opposing = [r for r in composition_results if r.cosine_similarity < -0.5]

    analysis = {
        "correlation": float(correlation),
        "orthogonal_success": float(np.mean([r.composition_success_rate for r in orthogonal])) if orthogonal else None,
        "aligned_success": float(np.mean([r.composition_success_rate for r in aligned])) if aligned else None,
        "opposing_success": float(np.mean([r.composition_success_rate for r in opposing])) if opposing else None,
        "n_orthogonal": len(orthogonal),
        "n_aligned": len(aligned),
        "n_opposing": len(opposing)
    }
    
    print("\n" + "="*60)
    print("GEOMETRY vs COMPOSITION ANALYSIS")
    print("="*60)
    print(f"Correlation (|similarity| vs success): {correlation:.3f}")
    if orthogonal:
        print(f"Orthogonal pairs success: {analysis['orthogonal_success']:.1%} (n={len(orthogonal)})")
    if aligned:
        print(f"Aligned pairs success: {analysis['aligned_success']:.1%} (n={len(aligned)})")
    if opposing:
        print(f"Opposing pairs success: {analysis['opposing_success']:.1%} (n={len(opposing)})")
    
    return analysis



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
    """Run all Week 2 composition experiments."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/week2")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer samples")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}. Run week 1 first!")
    
    # =========================================================================
    # Load optimal parameters from Week 1
    # =========================================================================
    print("="*60)
    print("WEEK 2: Steering Vector Composition")
    print("="*60)
    
    optimal_layers_path = week1_dir / "optimal_layers.json"
    if not optimal_layers_path.exists():
        raise FileNotFoundError(
            f"Optimal layers not found at {optimal_layers_path}. "
            "Run Week 1 without --skip_optimal_layers first."
        )
    
    with open(optimal_layers_path) as f:
        optimal_layers = json.load(f)
    print(f"\n✓ Loaded optimal layers from {optimal_layers_path}")
    print(f"Optimal layers: {optimal_layers}")
    
    optimal_coefficients_path = week1_dir / "optimal_coefficients.json"
    if not optimal_coefficients_path.exists():
        raise FileNotFoundError(
            f"Optimal coefficients not found at {optimal_coefficients_path}. "
            "Run Week 1 without --skip_optimal_coefficient first."
        )
    
    with open(optimal_coefficients_path) as f:
        optimal_coefficients = json.load(f)
    print(f"✓ Loaded optimal coefficients from {optimal_coefficients_path}")
    print(f"Optimal coefficients: {optimal_coefficients}")
    
    # Load Week 1 geometry analysis to find good pairs
    geometry_path = week1_dir / "geometry_analysis.json"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry analysis not found at {geometry_path}")
    
    with open(geometry_path) as f:
        geometry = json.load(f)
    
    concepts = args.concepts or geometry["concepts"]
    
    print(f"\nModel: {args.model}")
    print(f"Concepts: {concepts}")
    print(f"Output: {output_dir}")
    
    # =========================================================================
    # Load model
    # =========================================================================
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
    
    # =========================================================================
    # Load steering vectors from Week 1
    # =========================================================================
    print("Loading steering vectors from Week 1...")
    steering_vectors_by_layer = load_cached_vectors(
        week1_dir / "vectors",
        concepts,
        cfg.model.steering_layers
    )
    
    # Create steering vectors dict using optimal layers
    steering_vectors = {
        c: steering_vectors_by_layer[c][optimal_layers[c]] 
        for c in concepts if c in steering_vectors_by_layer
    }
    
    print(f"✓ Loaded {len(steering_vectors)} steering vectors")
    
    # Get test prompts
    prompts = get_test_prompts()
    if args.quick:
        prompts = prompts[:5]
        n_gen = 2
    else:
        prompts = prompts[:10]
        n_gen = 3
    
    # Initialize result variables
    additive_results = []
    opposing_results = []
    arithmetic_results = []
    scaling_results = {"note": "Not run"}
    
    # =========================================================================
    # Experiment 1: Additive Composition (Orthogonal Pairs)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Additive Composition (Orthogonal Pairs)")
    print("="*60)
    
    orthogonal_pairs = parse_concept_pairs(geometry, "orthogonal")
    
    if not orthogonal_pairs:
        print("No orthogonal pairs found in geometry analysis.")
        print("Creating pairs from available concepts...")
        sim_matrix = np.array(geometry["similarity_matrix"])
        concept_names = geometry["concepts"]
        
        for i, c1 in enumerate(concept_names):
            for j, c2 in enumerate(concept_names):
                if i < j and abs(sim_matrix[i, j]) < 0.2:
                    if c1 in concepts and c2 in concepts:
                        orthogonal_pairs.append((c1, c2))
        
        if not orthogonal_pairs and len(concepts) >= 2:
            print("Using first two concepts as fallback...")
            orthogonal_pairs = [(concepts[0], concepts[1])]
    
    print(f"Testing {len(orthogonal_pairs)} orthogonal pairs: {orthogonal_pairs}")
    
    for concept_a, concept_b in orthogonal_pairs:
        if concept_a in steering_vectors and concept_b in steering_vectors:
            try:
                result = test_additive_composition(
                    model, tokenizer, steering_vectors,
                    concept_a, concept_b,
                    optimal_layers, optimal_coefficients,
                    prompts, n_generations=n_gen
                )
                additive_results.append(result)
            except Exception as e:
                print(f"Error testing {concept_a} + {concept_b}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # =========================================================================
    # Experiment 2: Opposing Composition
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Opposing Vector Cancellation")
    print("="*60)
    
    opposing_pairs = parse_concept_pairs(geometry, "opposing")
    
    if not opposing_pairs:
        print("No opposing pairs found in geometry analysis.")
        print("Using known opposing concepts...")
        known_opposing = [
            ("formal", "casual"),
            ("positive", "negative"),
            ("verbose", "concise"),
            ("confident", "uncertain"),
            ("technical", "simple")
        ]
        opposing_pairs = [(a, b) for a, b in known_opposing if a in concepts and b in concepts]

    print(f"Testing {len(opposing_pairs)} opposing pairs: {opposing_pairs}")

    if not opposing_pairs:
        print("No opposing pairs available! Skipping opposing test...")
    else:
        for concept_a, concept_b in opposing_pairs:
            if concept_a in steering_vectors and concept_b in steering_vectors:
                try:
                    result = test_opposing_composition(
                        model, tokenizer, steering_vectors,
                        concept_a, concept_b,
                        optimal_layers, optimal_coefficients,
                        prompts, n_generations=n_gen
                    )
                    opposing_results.append(result)
                except Exception as e:
                    print(f"Error testing {concept_a} + {concept_b}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # =========================================================================
    # Experiment 3: Arithmetic Composition
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Vector Arithmetic")
    print("="*60)

    test_pairs = orthogonal_pairs

    if not test_pairs:
        print("No pairs available for arithmetic test!")
    else:
        for concept_a, concept_b in test_pairs:
            if concept_a in steering_vectors and concept_b in steering_vectors:
                try:
                    result = test_arithmetic_composition(
                        model, tokenizer, steering_vectors,
                        concept_a, concept_b,
                        optimal_layers, optimal_coefficients,
                        prompts, n_generations=n_gen
                    )
                    arithmetic_results.append(result)
                except Exception as e:
                    print(f"Error testing arithmetic {concept_a}, {concept_b}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # =========================================================================
    # Experiment 4: Coefficient Scaling
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Coefficient Scaling")
    print("="*60)

    if orthogonal_pairs and len(orthogonal_pairs) > 0:
        concept_a, concept_b = orthogonal_pairs[0]
        if concept_a in steering_vectors and concept_b in steering_vectors:
            if args.quick:
                alpha_range = [0.5, 1.0, 1.5]
                beta_range = [0.5, 1.0, 1.5]
            else:
                alpha_range = [0.25, 0.5, 1.0, 1.5, 2.0]
                beta_range = [0.25, 0.5, 1.0, 1.5, 2.0]
            
            try:
                scaling_results = test_coefficient_scaling(
                    model, tokenizer, steering_vectors,
                    concept_a, concept_b,
                    optimal_layers,
                    prompts, alpha_range, beta_range,
                    n_generations=2
                )
            except Exception as e:
                print(f"Error in coefficient scaling: {e}")
                import traceback
                traceback.print_exc()
                scaling_results = {"note": "Failed", "error": str(e)}
        else:
            print(f"Concepts not found in steering vectors: {concept_a}, {concept_b}")
    else:
        print("No orthogonal pairs for scaling test")

    # =========================================================================
    # Experiment 5: Geometry vs Composition Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Geometry vs Composition Success")
    print("="*60)

    if len(additive_results) > 0:
        try:
            geometry_analysis = analyze_composition_vs_geometry(additive_results)
        except Exception as e:
            print(f"Error in geometry analysis: {e}")
            import traceback
            traceback.print_exc()
            geometry_analysis = {"note": "Failed", "error": str(e)}
    else:
        print("No additive composition results available for geometry analysis")
        geometry_analysis = {"note": "No results to analyze"}

    # =========================================================================
    # Save all results
    # =========================================================================
    all_results = {
        "optimal_layers": optimal_layers,
        "optimal_coefficients": optimal_coefficients,
        "additive_composition": [
            {
                "concept_a": r.concept_a,
                "concept_b": r.concept_b,
                "layer_a": r.layer_a,
                "layer_b": r.layer_b,
                "coefficient_a": r.coefficient_a,
                "coefficient_b": r.coefficient_b,
                "baseline_scores": r.baseline_scores,
                "a_only_scores": r.a_only_scores,
                "b_only_scores": r.b_only_scores,
                "composition_scores": r.composition_scores,
                "cosine_similarity": r.cosine_similarity,
                "both_present": r.both_present,
                "composition_success_rate": r.composition_success_rate,
                "baseline_perplexity": r.baseline_perplexity,
                "composition_perplexity": r.composition_perplexity
            }
            for r in additive_results
        ],
        "opposing_composition": opposing_results,
        "arithmetic_composition": arithmetic_results,
        "coefficient_scaling": scaling_results,
        "geometry_analysis": geometry_analysis
    }

    # Convert and save
    all_results = convert_to_native(all_results)

    try:
        with open(output_dir / "composition_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {output_dir / 'composition_results.json'}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("WEEK 2 COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    print("\nKey Findings:")
    if additive_results:
        avg_success = np.mean([r.composition_success_rate for r in additive_results])
        print(f"  Additive composition success rate: {avg_success:.1%}")
        print(f"  Pairs tested: {len(additive_results)}")
    else:
        print("  No additive composition results")

    if opposing_results:
        cancel_count = sum(1 for r in opposing_results if r.get("cancels_out", False))
        print(f"  Opposing vectors cancel: {cancel_count}/{len(opposing_results)}")
    else:
        print("  No opposing composition results")

    if arithmetic_results:
        arith_work_count = sum(1 for r in arithmetic_results if r.get("arithmetic_works", False))
        print(f"  Arithmetic composition works: {arith_work_count}/{len(arithmetic_results)}")
    else:
        print("  No arithmetic composition results")


if __name__ == "__main__":
    main()