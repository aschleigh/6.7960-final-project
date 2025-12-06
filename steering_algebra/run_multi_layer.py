"""
Multi-Layer Steering Experiment: Test effectiveness of applying steering vectors to multiple layers.

Core question:
Does applying a steering vector to multiple optimal layers simultaneously improve effectiveness
compared to single-layer steering?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import (
    SteeringConfig,
    generate_with_steering,
    generate_baseline
)
from evaluation.classifiers import MultiAttributeEvaluator, AttributeClassifier
from evaluation.metrics import QualityMetrics


@dataclass
class MultiLayerResult:
    """Results from multi-layer steering experiment."""
    concept: str
    layers_used: List[int]
    n_layers: int
    coefficient: float
    coefficients: List[float]  # Actual coefficients used per layer
    strategy: str  # "uniform", "scaled", or "decaying"
    
    # Scores
    baseline_mean: float
    baseline_std: float
    steered_mean: float
    steered_std: float
    improvement: float
    success_rate: float
    
    # Quality
    baseline_perplexity: float
    steered_perplexity: float
    perplexity_delta: float


def rank_layers_by_performance(
    model,
    tokenizer,
    steering_vectors_by_layer: Dict[int, torch.Tensor],
    concept: str,
    test_prompts: List[str],
    coefficient: float,
    n_prompts: int = 5,
    n_generations: int = 2
) -> List[Tuple[int, float]]:
    """
    Rank all available layers by their steering effectiveness for a given concept.
    
    Returns:
        List of (layer, mean_score) tuples, sorted by score descending
    """
    print(f"\nRanking layers for '{concept}'...")
    
    classifier = AttributeClassifier(concept)
    prompts = test_prompts[:n_prompts]
    
    layer_scores = {}
    
    for layer, vector in tqdm(steering_vectors_by_layer.items(), desc="Testing layers"):
        scores = []
        
        for prompt in prompts:
            config = SteeringConfig(vector=vector, layer=layer, coefficient=coefficient)
            for _ in range(n_generations):
                text = generate_with_steering(model, tokenizer, prompt, config)
                score = classifier.score(text)
                scores.append(score)
        
        layer_scores[layer] = np.mean(scores)
        print(f"  Layer {layer:2d}: {layer_scores[layer]:.3f}")
    
    # Sort by score descending
    ranked = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 layers for '{concept}':")
    for i, (layer, score) in enumerate(ranked[:5], 1):
        print(f"  {i}. Layer {layer}: {score:.3f}")
    
    return ranked


def test_multi_layer_steering(
    model,
    tokenizer,
    steering_vectors_by_layer: Dict[int, torch.Tensor],
    concept: str,
    ranked_layers: List[Tuple[int, float]],
    coefficient: float,
    test_prompts: List[str],
    max_layers: int = 5,
    n_prompts: int = 10,
    n_generations: int = 3,
    strategy: str = "uniform"
) -> List[MultiLayerResult]:
    """
    Test steering effectiveness when applying vector to top-k layers.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        steering_vectors_by_layer: Dict mapping layer -> steering vector
        concept: Concept being tested
        ranked_layers: Ranked list of (layer, score) from best to worst
        coefficient: Steering coefficient to use
        test_prompts: List of test prompts
        max_layers: Maximum number of layers to test (tests 1, 2, 3, ..., max_layers)
        n_prompts: Number of prompts to use
        n_generations: Number of generations per prompt
        strategy: Coefficient strategy - "uniform", "scaled", or "decaying"
    
    Returns:
        List of MultiLayerResult objects
    """
    print(f"\n{'='*60}")
    print(f"Testing Multi-Layer Steering for '{concept}'")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    classifier = AttributeClassifier(concept)
    quality_metrics = QualityMetrics()
    
    prompts = test_prompts[:n_prompts]
    results = []
    
    # Generate baseline once (same for all conditions)
    print("\nGenerating baseline samples...")
    baseline_texts = []
    baseline_scores = []
    
    for prompt in tqdm(prompts, desc="Baseline"):
        for _ in range(n_generations):
            text = generate_baseline(model, tokenizer, prompt)
            baseline_texts.append(text)
            score = classifier.score(text)
            baseline_scores.append(score)
    
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)
    baseline_ppls = quality_metrics.perplexity_calc.compute_batch(baseline_texts)
    baseline_perplexity = np.mean(baseline_ppls)
    
    print(f"Baseline score: {baseline_mean:.3f} ± {baseline_std:.3f}")
    print(f"Baseline perplexity: {baseline_perplexity:.1f}")
    
    # Test with increasing numbers of layers
    for n_layers in range(1, min(max_layers + 1, len(ranked_layers) + 1)):
        top_layers = [layer for layer, _ in ranked_layers[:n_layers]]
        
        # Calculate coefficients based on strategy
        if strategy == "uniform":
            # Same coefficient for all layers
            coefficients = [coefficient] * n_layers
        elif strategy == "scaled":
            # Divide by number of layers
            coefficients = [coefficient / n_layers] * n_layers
        elif strategy == "decaying":
            # Each subsequent layer gets coefficient * 0.75
            decay_factor = 0.75
            coefficients = [coefficient * (decay_factor ** i) for i in range(n_layers)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"\n--- Testing with top {n_layers} layer(s): {top_layers} ---")
        print(f"Coefficients: {[f'{c:.3f}' for c in coefficients]}")
        
        steered_texts = []
        steered_scores = []
        
        # Create multi-layer steering config
        configs = [
            SteeringConfig(
                vector=steering_vectors_by_layer[layer],
                layer=layer,
                coefficient=coef
            )
            for layer, coef in zip(top_layers, coefficients)
        ]
        
        # Generate with multi-layer steering
        for prompt in tqdm(prompts, desc=f"Steering ({n_layers} layers)"):
            for _ in range(n_generations):
                text = generate_with_steering(model, tokenizer, prompt, configs)
                steered_texts.append(text)
                score = classifier.score(text)
                steered_scores.append(score)
        
        # Calculate metrics
        steered_mean = np.mean(steered_scores)
        steered_std = np.std(steered_scores)
        improvement = steered_mean - baseline_mean
        
        # Success rate: fraction where steered > baseline
        successes = sum(
            1 for s, b in zip(steered_scores, baseline_scores) 
            if s > b
        )
        success_rate = successes / len(steered_scores)
        
        # Quality
        steered_ppls = quality_metrics.perplexity_calc.compute_batch(steered_texts)
        steered_perplexity = np.mean(steered_ppls)
        perplexity_delta = steered_perplexity - baseline_perplexity
        
        result = MultiLayerResult(
            concept=concept,
            layers_used=top_layers,
            n_layers=n_layers,
            coefficient=coefficient,
            coefficients=coefficients,
            strategy=strategy,
            baseline_mean=float(baseline_mean),
            baseline_std=float(baseline_std),
            steered_mean=float(steered_mean),
            steered_std=float(steered_std),
            improvement=float(improvement),
            success_rate=float(success_rate),
            baseline_perplexity=float(baseline_perplexity),
            steered_perplexity=float(steered_perplexity),
            perplexity_delta=float(perplexity_delta)
        )
        
        results.append(result)
        
        print(f"  Steered score: {steered_mean:.3f} ± {steered_std:.3f}")
        print(f"  Improvement: {improvement:+.3f}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Perplexity Δ: {perplexity_delta:+.1f}")
    
    return results


def analyze_multi_layer_results(
    results: List[MultiLayerResult],
    strategy: str
) -> Dict:
    """
    Analyze the relationship between number of layers and effectiveness.
    """
    print(f"\n{'='*60}")
    print(f"MULTI-LAYER ANALYSIS ({strategy.upper()} strategy)")
    print(f"{'='*60}")
    
    if not results:
        return {"note": "No results to analyze"}
    
    # Find best configuration
    best_result = max(results, key=lambda r: r.improvement)
    
    # Analyze trend
    n_layers_list = [r.n_layers for r in results]
    improvements = [r.improvement for r in results]
    success_rates = [r.success_rate for r in results]
    perplexity_deltas = [r.perplexity_delta for r in results]
    
    # Check if improvement increases with more layers
    improvement_trend = np.corrcoef(n_layers_list, improvements)[0, 1] if len(results) > 1 else None
    
    analysis = {
        "concept": results[0].concept,
        "strategy": strategy,
        "best_config": {
            "n_layers": best_result.n_layers,
            "layers": best_result.layers_used,
            "coefficients": best_result.coefficients,
            "improvement": best_result.improvement,
            "success_rate": best_result.success_rate,
            "perplexity_delta": best_result.perplexity_delta
        },
        "trend": {
            "improvement_correlation": float(improvement_trend) if improvement_trend is not None else None,
            "max_improvement": float(max(improvements)),
            "min_improvement": float(min(improvements)),
            "avg_success_rate": float(np.mean(success_rates)),
            "avg_perplexity_delta": float(np.mean(perplexity_deltas))
        },
        "by_n_layers": [
            {
                "n_layers": r.n_layers,
                "layers": r.layers_used,
                "coefficients": r.coefficients,
                "improvement": r.improvement,
                "success_rate": r.success_rate,
                "perplexity_delta": r.perplexity_delta
            }
            for r in results
        ]
    }
    
    print(f"\nBest Configuration: {best_result.n_layers} layers")
    print(f"  Layers: {best_result.layers_used}")
    print(f"  Improvement: {best_result.improvement:+.3f}")
    print(f"  Success rate: {best_result.success_rate:.1%}")
    
    if improvement_trend is not None:
        print(f"\nImprovement vs. # Layers Correlation: {improvement_trend:.3f}")
        if improvement_trend > 0.5:
            print("  → More layers tend to improve effectiveness")
        elif improvement_trend < -0.5:
            print("  → More layers tend to decrease effectiveness")
        else:
            print("  → Number of layers has weak correlation with effectiveness")
    
    # Print table
    print(f"\n{'Layers':<8} {'Improvement':>12} {'Success Rate':>15} {'Perplexity Δ':>15}")
    print("-" * 55)
    for r in results:
        print(f"{r.n_layers:<8} {r.improvement:>+12.3f} {r.success_rate:>14.1%} {r.perplexity_delta:>+15.1f}")
    
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
    """Run multi-layer steering experiment."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from extraction.extract_vectors import load_cached_vectors
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/multi_layer")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--max_layers", type=int, default=5, 
                       help="Maximum number of layers to test")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with fewer samples")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}. Run week 1 first!")
    
    print("="*60)
    print("MULTI-LAYER STEERING EXPERIMENT")
    print("="*60)
    
    # Load optimal coefficients from Week 1
    optimal_coefficients_path = week1_dir / "optimal_coefficients.json"
    if not optimal_coefficients_path.exists():
        raise FileNotFoundError(f"Optimal coefficients not found at {optimal_coefficients_path}")
    
    with open(optimal_coefficients_path) as f:
        optimal_coefficients = json.load(f)
    print(f"\n✓ Loaded optimal coefficients from {optimal_coefficients_path}")
    
    # Get concepts
    concepts = args.concepts or list(optimal_coefficients.keys())
    
    print(f"\nModel: {args.model}")
    print(f"Concepts: {concepts}")
    print(f"Max layers to test: {args.max_layers}")
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
    
    # Load steering vectors from Week 1
    print("\nLoading steering vectors from Week 1...")
    steering_vectors_by_concept = load_cached_vectors(
        week1_dir / "vectors",
        concepts,
        cfg.model.steering_layers
    )
    
    print(f"✓ Loaded steering vectors for {len(steering_vectors_by_concept)} concepts")
    
    # Get test prompts
    test_prompts = get_test_prompts()
    if args.quick:
        n_prompts = 5
        n_gen = 2
    else:
        n_prompts = 10
        n_gen = 3
    
    # Run experiments for each concept
    all_results = {}
    all_analyses = {}
    
    for concept in concepts:
        if concept not in steering_vectors_by_concept:
            print(f"\n⚠ Warning: No vectors found for '{concept}', skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"CONCEPT: {concept.upper()}")
        print(f"{'='*60}")
        
        coefficient = optimal_coefficients.get(concept, cfg.model.default_coefficient)
        print(f"Using coefficient: {coefficient}")
        
        # Step 1: Rank layers
        ranked_layers = rank_layers_by_performance(
            model,
            tokenizer,
            steering_vectors_by_concept[concept],
            concept,
            test_prompts,
            coefficient,
            n_prompts=5,
            n_generations=2
        )
        
        # Step 2: Test multi-layer steering
        results = test_multi_layer_steering(
            model,
            tokenizer,
            steering_vectors_by_concept[concept],
            concept,
            ranked_layers,
            coefficient,
            test_prompts,
            max_layers=args.max_layers,
            n_prompts=n_prompts,
            n_generations=n_gen
        )
        
        # Step 3: Analyze results
        analysis = analyze_multi_layer_results(results)
        
        all_results[concept] = [
            {
                "n_layers": r.n_layers,
                "layers_used": r.layers_used,
                "coefficient": r.coefficient,
                "baseline_mean": r.baseline_mean,
                "baseline_std": r.baseline_std,
                "steered_mean": r.steered_mean,
                "steered_std": r.steered_std,
                "improvement": r.improvement,
                "success_rate": r.success_rate,
                "baseline_perplexity": r.baseline_perplexity,
                "steered_perplexity": r.steered_perplexity,
                "perplexity_delta": r.perplexity_delta
            }
            for r in results
        ]
        all_analyses[concept] = analysis
    
    # Save results
    output_data = {
        "experiment": "multi_layer_steering",
        "max_layers_tested": args.max_layers,
        "concepts": concepts,
        "results": all_results,
        "analyses": all_analyses
    }
    
    output_data = convert_to_native(output_data)
    
    try:
        with open(output_dir / "multi_layer_results.json", "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {output_dir / 'multi_layer_results.json'}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    
    print("\nKey Findings:")
    for concept, analysis in all_analyses.items():
        if "best_config" in analysis:
            best = analysis["best_config"]
            print(f"\n{concept}:")
            print(f"  Best: {best['n_layers']} layers (improvement: {best['improvement']:+.3f})")
            print(f"  Layers used: {best['layers']}")


if __name__ == "__main__":
    main()



# """
# Multi-Layer Steering Experiment: Test effectiveness of applying steering vectors to multiple layers.

# Core question:
# Does applying a steering vector to multiple optimal layers simultaneously improve effectiveness
# compared to single-layer steering?
# """

# import torch
# import numpy as np
# from typing import Dict, List, Tuple
# from pathlib import Path
# import json
# from tqdm import tqdm
# from dataclasses import dataclass

# from config import cfg
# from data.prompts import get_test_prompts
# from steering.apply_steering import (
#     SteeringConfig,
#     generate_with_steering,
#     generate_baseline
# )
# from evaluation.classifiers import MultiAttributeEvaluator, AttributeClassifier
# from evaluation.metrics import QualityMetrics


# @dataclass
# class MultiLayerResult:
#     """Results from multi-layer steering experiment."""
#     concept: str
#     layers_used: List[int]
#     n_layers: int
#     coefficient: float
    
#     # Scores
#     baseline_mean: float
#     baseline_std: float
#     steered_mean: float
#     steered_std: float
#     improvement: float
#     success_rate: float
    
#     # Quality
#     baseline_perplexity: float
#     steered_perplexity: float
#     perplexity_delta: float


# def rank_layers_by_performance(
#     model,
#     tokenizer,
#     steering_vectors_by_layer: Dict[int, torch.Tensor],
#     concept: str,
#     test_prompts: List[str],
#     coefficient: float,
#     n_prompts: int = 5,
#     n_generations: int = 2
# ) -> List[Tuple[int, float]]:
#     """
#     Rank all available layers by their steering effectiveness for a given concept.
    
#     Returns:
#         List of (layer, mean_score) tuples, sorted by score descending
#     """
#     print(f"\nRanking layers for '{concept}'...")
    
#     classifier = AttributeClassifier(concept)
#     prompts = test_prompts[:n_prompts]
    
#     layer_scores = {}
    
#     for layer, vector in tqdm(steering_vectors_by_layer.items(), desc="Testing layers"):
#         scores = []
        
#         for prompt in prompts:
#             config = SteeringConfig(vector=vector, layer=layer, coefficient=coefficient)
#             for _ in range(n_generations):
#                 text = generate_with_steering(model, tokenizer, prompt, config)
#                 score = classifier.score(text)
#                 scores.append(score)
        
#         layer_scores[layer] = np.mean(scores)
#         print(f"  Layer {layer:2d}: {layer_scores[layer]:.3f}")
    
#     # Sort by score descending
#     ranked = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    
#     print(f"\nTop 5 layers for '{concept}':")
#     for i, (layer, score) in enumerate(ranked[:5], 1):
#         print(f"  {i}. Layer {layer}: {score:.3f}")
    
#     return ranked


# def test_multi_layer_steering(
#     model,
#     tokenizer,
#     steering_vectors_by_layer: Dict[int, torch.Tensor],
#     concept: str,
#     ranked_layers: List[Tuple[int, float]],
#     coefficient: float,
#     test_prompts: List[str],
#     max_layers: int = 5,
#     n_prompts: int = 10,
#     n_generations: int = 3
# ) -> List[MultiLayerResult]:
#     """
#     Test steering effectiveness when applying vector to top-k layers.
    
#     Args:
#         model: The language model
#         tokenizer: The tokenizer
#         steering_vectors_by_layer: Dict mapping layer -> steering vector
#         concept: Concept being tested
#         ranked_layers: Ranked list of (layer, score) from best to worst
#         coefficient: Steering coefficient to use
#         test_prompts: List of test prompts
#         max_layers: Maximum number of layers to test (tests 1, 2, 3, ..., max_layers)
#         n_prompts: Number of prompts to use
#         n_generations: Number of generations per prompt
    
#     Returns:
#         List of MultiLayerResult objects
#     """
#     print(f"\n{'='*60}")
#     print(f"Testing Multi-Layer Steering for '{concept}'")
#     print(f"{'='*60}")
    
#     classifier = AttributeClassifier(concept)
#     quality_metrics = QualityMetrics()
    
#     prompts = test_prompts[:n_prompts]
#     results = []
    
#     # Generate baseline once (same for all conditions)
#     print("\nGenerating baseline samples...")
#     baseline_texts = []
#     baseline_scores = []
    
#     for prompt in tqdm(prompts, desc="Baseline"):
#         for _ in range(n_generations):
#             text = generate_baseline(model, tokenizer, prompt)
#             baseline_texts.append(text)
#             score = classifier.score(text)
#             baseline_scores.append(score)
    
#     baseline_mean = np.mean(baseline_scores)
#     baseline_std = np.std(baseline_scores)
#     baseline_ppls = quality_metrics.perplexity_calc.compute_batch(baseline_texts)
#     baseline_perplexity = np.mean(baseline_ppls)
    
#     print(f"Baseline score: {baseline_mean:.3f} ± {baseline_std:.3f}")
#     print(f"Baseline perplexity: {baseline_perplexity:.1f}")
    
#     # Test with increasing numbers of layers
#     for n_layers in range(1, min(max_layers + 1, len(ranked_layers) + 1)):
#         top_layers = [layer for layer, _ in ranked_layers[:n_layers]]
        
#         print(f"\n--- Testing with top {n_layers} layer(s): {top_layers} ---")
        
#         steered_texts = []
#         steered_scores = []
        
#         # Create multi-layer steering config
#         configs = [
#             SteeringConfig(
#                 vector=steering_vectors_by_layer[layer],
#                 layer=layer,
#                 coefficient=coefficient
#             )
#             for layer in top_layers
#         ]
        
#         # Generate with multi-layer steering
#         for prompt in tqdm(prompts, desc=f"Steering ({n_layers} layers)"):
#             for _ in range(n_generations):
#                 text = generate_with_steering(model, tokenizer, prompt, configs)
#                 steered_texts.append(text)
#                 score = classifier.score(text)
#                 steered_scores.append(score)
        
#         # Calculate metrics
#         steered_mean = np.mean(steered_scores)
#         steered_std = np.std(steered_scores)
#         improvement = steered_mean - baseline_mean
        
#         # Success rate: fraction where steered > baseline
#         successes = sum(
#             1 for s, b in zip(steered_scores, baseline_scores) 
#             if s > b
#         )
#         success_rate = successes / len(steered_scores)
        
#         # Quality
#         steered_ppls = quality_metrics.perplexity_calc.compute_batch(steered_texts)
#         steered_perplexity = np.mean(steered_ppls)
#         perplexity_delta = steered_perplexity - baseline_perplexity
        
#         result = MultiLayerResult(
#             concept=concept,
#             layers_used=top_layers,
#             n_layers=n_layers,
#             coefficient=coefficient,
#             baseline_mean=float(baseline_mean),
#             baseline_std=float(baseline_std),
#             steered_mean=float(steered_mean),
#             steered_std=float(steered_std),
#             improvement=float(improvement),
#             success_rate=float(success_rate),
#             baseline_perplexity=float(baseline_perplexity),
#             steered_perplexity=float(steered_perplexity),
#             perplexity_delta=float(perplexity_delta)
#         )
        
#         results.append(result)
        
#         print(f"  Steered score: {steered_mean:.3f} ± {steered_std:.3f}")
#         print(f"  Improvement: {improvement:+.3f}")
#         print(f"  Success rate: {success_rate:.1%}")
#         print(f"  Perplexity Δ: {perplexity_delta:+.1f}")
    
#     return results


# def analyze_multi_layer_results(
#     results: List[MultiLayerResult]
# ) -> Dict:
#     """
#     Analyze the relationship between number of layers and effectiveness.
#     """
#     print(f"\n{'='*60}")
#     print("MULTI-LAYER ANALYSIS")
#     print(f"{'='*60}")
    
#     if not results:
#         return {"note": "No results to analyze"}
    
#     # Find best configuration
#     best_result = max(results, key=lambda r: r.improvement)
    
#     # Analyze trend
#     n_layers_list = [r.n_layers for r in results]
#     improvements = [r.improvement for r in results]
#     success_rates = [r.success_rate for r in results]
#     perplexity_deltas = [r.perplexity_delta for r in results]
    
#     # Check if improvement increases with more layers
#     improvement_trend = np.corrcoef(n_layers_list, improvements)[0, 1] if len(results) > 1 else None
    
#     analysis = {
#         "concept": results[0].concept,
#         "best_config": {
#             "n_layers": best_result.n_layers,
#             "layers": best_result.layers_used,
#             "improvement": best_result.improvement,
#             "success_rate": best_result.success_rate,
#             "perplexity_delta": best_result.perplexity_delta
#         },
#         "trend": {
#             "improvement_correlation": float(improvement_trend) if improvement_trend is not None else None,
#             "max_improvement": float(max(improvements)),
#             "min_improvement": float(min(improvements)),
#             "avg_success_rate": float(np.mean(success_rates)),
#             "avg_perplexity_delta": float(np.mean(perplexity_deltas))
#         },
#         "by_n_layers": [
#             {
#                 "n_layers": r.n_layers,
#                 "layers": r.layers_used,
#                 "improvement": r.improvement,
#                 "success_rate": r.success_rate,
#                 "perplexity_delta": r.perplexity_delta
#             }
#             for r in results
#         ]
#     }
    
#     print(f"\nBest Configuration: {best_result.n_layers} layers")
#     print(f"  Layers: {best_result.layers_used}")
#     print(f"  Improvement: {best_result.improvement:+.3f}")
#     print(f"  Success rate: {best_result.success_rate:.1%}")
    
#     if improvement_trend is not None:
#         print(f"\nImprovement vs. # Layers Correlation: {improvement_trend:.3f}")
#         if improvement_trend > 0.5:
#             print("  → More layers tend to improve effectiveness")
#         elif improvement_trend < -0.5:
#             print("  → More layers tend to decrease effectiveness")
#         else:
#             print("  → Number of layers has weak correlation with effectiveness")
    
#     # Print table
#     print(f"\n{'Layers':<8} {'Improvement':>12} {'Success Rate':>15} {'Perplexity Δ':>15}")
#     print("-" * 55)
#     for r in results:
#         print(f"{r.n_layers:<8} {r.improvement:>+12.3f} {r.success_rate:>14.1%} {r.perplexity_delta:>+15.1f}")
    
#     return analysis


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
#     """Run multi-layer steering experiment."""
#     import argparse
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     from extraction.extract_vectors import load_cached_vectors
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", default=cfg.model.name)
#     parser.add_argument("--output_dir", default="outputs/multi_layer")
#     parser.add_argument("--week1_dir", default="outputs/week1")
#     parser.add_argument("--concepts", nargs="+", default=None)
#     parser.add_argument("--max_layers", type=int, default=5, 
#                        help="Maximum number of layers to test")
#     parser.add_argument("--quick", action="store_true", 
#                        help="Run quick test with fewer samples")
#     args = parser.parse_args()
    
#     # Setup
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     week1_dir = Path(args.week1_dir)
#     if not week1_dir.exists():
#         raise FileNotFoundError(f"Week 1 results not found at {week1_dir}. Run week 1 first!")
    
#     print("="*60)
#     print("MULTI-LAYER STEERING EXPERIMENT")
#     print("="*60)
    
#     # Load optimal coefficients from Week 1
#     optimal_coefficients_path = week1_dir / "optimal_coefficients.json"
#     if not optimal_coefficients_path.exists():
#         raise FileNotFoundError(f"Optimal coefficients not found at {optimal_coefficients_path}")
    
#     with open(optimal_coefficients_path) as f:
#         optimal_coefficients = json.load(f)
#     print(f"\n✓ Loaded optimal coefficients from {optimal_coefficients_path}")
    
#     # Get concepts
#     concepts = args.concepts or list(optimal_coefficients.keys())
    
#     print(f"\nModel: {args.model}")
#     print(f"Concepts: {concepts}")
#     print(f"Max layers to test: {args.max_layers}")
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
    
#     # Load steering vectors from Week 1
#     print("\nLoading steering vectors from Week 1...")
#     steering_vectors_by_concept = load_cached_vectors(
#         week1_dir / "vectors",
#         concepts,
#         cfg.model.steering_layers
#     )
    
#     print(f"✓ Loaded steering vectors for {len(steering_vectors_by_concept)} concepts")
    
#     # Get test prompts
#     test_prompts = get_test_prompts()
#     if args.quick:
#         n_prompts = 5
#         n_gen = 2
#     else:
#         n_prompts = 10
#         n_gen = 3
    
#     # Run experiments for each concept
#     all_results = {}
#     all_analyses = {}
    
#     for concept in concepts:
#         if concept not in steering_vectors_by_concept:
#             print(f"\n⚠ Warning: No vectors found for '{concept}', skipping...")
#             continue
        
#         print(f"\n{'='*60}")
#         print(f"CONCEPT: {concept.upper()}")
#         print(f"{'='*60}")
        
#         coefficient = optimal_coefficients.get(concept, cfg.model.default_coefficient)
#         print(f"Using coefficient: {coefficient}")
        
#         # Step 1: Rank layers
#         ranked_layers = rank_layers_by_performance(
#             model,
#             tokenizer,
#             steering_vectors_by_concept[concept],
#             concept,
#             test_prompts,
#             coefficient,
#             n_prompts=5,
#             n_generations=2
#         )
        
#         # Step 2: Test multi-layer steering
#         results = test_multi_layer_steering(
#             model,
#             tokenizer,
#             steering_vectors_by_concept[concept],
#             concept,
#             ranked_layers,
#             coefficient,
#             test_prompts,
#             max_layers=args.max_layers,
#             n_prompts=n_prompts,
#             n_generations=n_gen
#         )
        
#         # Step 3: Analyze results
#         analysis = analyze_multi_layer_results(results)
        
#         all_results[concept] = [
#             {
#                 "n_layers": r.n_layers,
#                 "layers_used": r.layers_used,
#                 "coefficient": r.coefficient,
#                 "baseline_mean": r.baseline_mean,
#                 "baseline_std": r.baseline_std,
#                 "steered_mean": r.steered_mean,
#                 "steered_std": r.steered_std,
#                 "improvement": r.improvement,
#                 "success_rate": r.success_rate,
#                 "baseline_perplexity": r.baseline_perplexity,
#                 "steered_perplexity": r.steered_perplexity,
#                 "perplexity_delta": r.perplexity_delta
#             }
#             for r in results
#         ]
#         all_analyses[concept] = analysis
    
#     # Save results
#     output_data = {
#         "experiment": "multi_layer_steering",
#         "max_layers_tested": args.max_layers,
#         "concepts": concepts,
#         "results": all_results,
#         "analyses": all_analyses
#     }
    
#     output_data = convert_to_native(output_data)
    
#     try:
#         with open(output_dir / "multi_layer_results.json", "w") as f:
#             json.dump(output_data, f, indent=2)
#         print(f"\n✓ Results saved to {output_dir / 'multi_layer_results.json'}")
#     except Exception as e:
#         print(f"⚠ Warning: Failed to save results: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # Summary
#     print(f"\n{'='*60}")
#     print("EXPERIMENT COMPLETE")
#     print(f"{'='*60}")
#     print(f"\nResults saved to: {output_dir}")
    
#     print("\nKey Findings:")
#     for concept, analysis in all_analyses.items():
#         if "best_config" in analysis:
#             best = analysis["best_config"]
#             print(f"\n{concept}:")
#             print(f"  Best: {best['n_layers']} layers (improvement: {best['improvement']:+.3f})")
#             print(f"  Layers used: {best['layers']}")


# if __name__ == "__main__":
#     main()