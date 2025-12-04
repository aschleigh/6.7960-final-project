"""
Week 1 Experiment: Validate single-vector steering works.

This script:
1. Extracts steering vectors for all concepts
2. Validates each vector separates positive/negative examples
3. Tests steering on neutral prompts
4. Evaluates steering success with classifiers
"""

import torch
from typing import Dict, List
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from config import cfg
from data.contrastive_pairs import get_all_pairs
from data.prompts import get_test_prompts
from extraction.extract_vectors import (
    extract_all_vectors,
    analyze_extraction_quality,
    load_cached_vectors
)
from steering.apply_steering import (
    generate_with_steering,
    generate_baseline,
    coefficient_sweep,
    SteeringConfig
)
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics
from evaluation.geometry import compute_similarity_matrix

# helper conversion function
def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)  

def run_extraction_validation(
    model,
    tokenizer,
    concepts: List[str],
    contrastive_pairs: Dict,
    layers: List[int],
    output_dir: Path
) -> Dict:
    
    print("\n" + "="*60)
    print("STEP 1: Extracting and validating steering vectors")
    print("="*60)
    
    results = {"extraction_quality": {}}
    
    steering_vectors = extract_all_vectors(
        model=model,
        tokenizer=tokenizer,
        concepts=concepts,
        contrastive_pairs=contrastive_pairs,
        layers=layers,
        cache_dir=output_dir / "vectors"
    )
    
    for concept in concepts:
        results["extraction_quality"][concept] = {}
        
        for layer in layers:
            quality = analyze_extraction_quality(
                model=model,
                tokenizer=tokenizer,
                contrastive_pairs=contrastive_pairs[concept][:50],  # Use subset for speed
                steering_vector=steering_vectors[concept][layer],
                layer_idx=layer
            )
            results["extraction_quality"][concept][layer] = quality
            
            print(f"\n{concept} @ layer {layer}:")
            print(f"  Cohen's d: {quality['cohens_d']:.3f}")
            print(f"  Separation accuracy: {quality['separation_accuracy']:.1%}")
    
    # try:
    extraction_quality = convert_to_native(results["extraction_quality"])
    with open(output_dir / "extraction_quality.json", "w") as f:
        json.dump(extraction_quality, f, indent=2)
    print(f"Extraction quality saved")
    # except Exception as e:
    #     print(f"Warning: Failed to save extraction quality: {e}")
    return steering_vectors, extraction_quality


def run_geometry_analysis(
    steering_vectors: Dict[str, Dict[int, torch.Tensor]],
    output_dir: Path
) -> Dict:

    print("\n" + "="*60)
    print("STEP 2: Analyzing steering vector geometry")
    print("="*60)
    
    vectors_at_layer = {
        concept: vectors[cfg.model.optimal_layer[concept]]
        for concept, vectors in steering_vectors.items()
    }

    # pairwise similarity (should be 1 along diagonals)
    sim_matrix, concepts = compute_similarity_matrix(vectors_at_layer)
    
    print("\nCosine Similarity Matrix:")
    print("-" * 40)
    
    header = "         " + " ".join([f"{c[:6]:>7}" for c in concepts])
    print(header)
    
    for i, c1 in enumerate(concepts):
        row = f"{c1[:8]:<8}"
        for j, c2 in enumerate(concepts):
            row += f" {sim_matrix[i,j]:>7.3f}"
        print(row)
    
    from evaluation.geometry import categorize_pairs_by_similarity
    # sorts pairs into buckets based on how similar they are
    categories = categorize_pairs_by_similarity(sim_matrix, concepts)
    
    print("\n\nPair Categories:")
    print("-" * 40)
    for cat, pairs in categories.items():
        if pairs:
            print(f"\n{cat.upper()}:")
            for c1, c2, sim in pairs:
                print(f"  {c1} <-> {c2}: {sim:.3f}")
    
    results = {
        "similarity_matrix": sim_matrix.tolist(),
        "concepts": concepts,
        "categories": {
            cat: [(c1, c2, float(sim)) for c1, c2, sim in pairs]
            for cat, pairs in categories.items()
        }
    }
    
    results = convert_to_native(results)
    
    # try:
    with open(output_dir / "geometry_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Geometry results saved")
    # except Exception as e:
    #     print(f"Warning: Failed to save geometry JSON: {e}")


def run_steering_validation(
    model,
    tokenizer,
    steering_vectors: Dict[str, Dict[int, torch.Tensor]],
    concepts: List[str],
    layers_per_concept,
    test_prompts: List[str],
    output_dir: Path,
    n_prompts: int = 10,
    n_generations: int = 3
) -> Dict:
    # does steering behave as expected?
    print("\n" + "="*60)
    print("STEP 3: Validating Steering Effects")
    print("="*60)
    
    
    evaluator = MultiAttributeEvaluator(concepts)
    quality_metrics = QualityMetrics()
    
    results = {
        "per_concept": {},
        "aggregated": {}
    }
    
    prompts = test_prompts[:n_prompts]
    
    for concept in concepts:
        layer = layers_per_concept[concept]
        print(f"\n\nTesting: {concept}")
        print("-" * 40)
        
        sv = steering_vectors[concept][layer]
        concept_results = {
            "baseline": {"texts": [], "scores": []},
            "steered": {"texts": [], "scores": []},
            "improvements": []
        }
        
        for prompt in tqdm(prompts, desc=f"Generating for {concept}"):
            # baseline
            for _ in range(n_generations):
                baseline_text = generate_baseline(
                    model, tokenizer, prompt,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    temperature=cfg.generation.temperature
                )
                baseline_score = evaluator.classifiers[concept].score(baseline_text)
                concept_results["baseline"]["texts"].append(baseline_text)
                concept_results["baseline"]["scores"].append(baseline_score)
            # steered 
            coef = cfg.model.optimal_coefficient.get(concept, cfg.model.default_coefficient)
            config = SteeringConfig(vector=sv, layer=layer, coefficient=coef)

            for _ in range(n_generations):
                steered_text = generate_with_steering(
                    model, tokenizer, prompt, config,
                    max_new_tokens=cfg.generation.max_new_tokens,
                    temperature=cfg.generation.temperature
                )
                steered_score = evaluator.classifiers[concept].score(steered_text)
                concept_results["steered"]["texts"].append(steered_text)
                concept_results["steered"]["scores"].append(steered_score)
        
        baseline_scores = concept_results["baseline"]["scores"]
        steered_scores = concept_results["steered"]["scores"]
        
        import numpy as np
        baseline_mean = np.mean(baseline_scores)
        steered_mean = np.mean(steered_scores)
        improvement = steered_mean - baseline_mean
        
        # success rate: fraction where steered > baseline
        n_samples = len(baseline_scores)
        successes = sum(
            1 for s, b in zip(steered_scores, baseline_scores) if s > b
        )
        success_rate = successes / n_samples
        
        concept_results["statistics"] = {
            "baseline_mean": float(baseline_mean),
            "baseline_std": float(np.std(baseline_scores)),
            "steered_mean": float(steered_mean),
            "steered_std": float(np.std(steered_scores)),
            "improvement": float(improvement),
            "success_rate": float(success_rate)
        }
        
        print(f"  Baseline score: {baseline_mean:.3f} ± {np.std(baseline_scores):.3f}")
        print(f"  Steered score:  {steered_mean:.3f} ± {np.std(steered_scores):.3f}")
        print(f"  Improvement:    {improvement:+.3f}")
        print(f"  Success rate:   {success_rate:.1%}")
        
        quality_comparison = quality_metrics.compare(
            concept_results["baseline"]["texts"],
            concept_results["steered"]["texts"]
        )
        concept_results["quality"] = quality_comparison
        
        print(f"  Perplexity Δ:   {quality_comparison['delta_perplexity_mean']:+.1f}")
        
        results["per_concept"][concept] = concept_results
    
    all_improvements = [
        results["per_concept"][c]["statistics"]["improvement"]
        for c in concepts
    ]
    all_success_rates = [
        results["per_concept"][c]["statistics"]["success_rate"]
        for c in concepts
    ]
    
    results["aggregated"] = {
        "mean_improvement": float(np.mean(all_improvements)),
        "mean_success_rate": float(np.mean(all_success_rates)),
        "concepts_with_positive_improvement": sum(1 for i in all_improvements if i > 0),
        "total_concepts": len(concepts)
    }
    
    # try:
    results_to_save = {
        "per_concept": {
            c: {
                "statistics": r["statistics"],
                "quality": r["quality"]
            }
            for c, r in results["per_concept"].items()
        },
        "aggregated": results["aggregated"]
    }
    
    results_to_save = convert_to_native(results_to_save)
    
    with open(output_dir / "steering_validation.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'steering_validation.json'}")
    
    # except Exception as e:
    #     print(f"Warning: Failed to save JSON: {e}")
    #     print("Continuing anyway...")
    
    return results


def run_coefficient_analysis(
    model,
    tokenizer,
    steering_vectors: Dict[str, Dict[int, torch.Tensor]],
    concept: str,
    test_prompts: List[str],
    output_dir: Path,
    coefficients: List[float] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    n_prompts: int = 5,
    n_generations: int = 2
) -> Dict:
    """
    Step 4: Analyze how steering coefficient affects results.
    """
    print("\n" + "="*60)
    print(f"STEP 4: Coefficient analysis for '{concept}'")
    print("="*60)
    
    from evaluation.classifiers import AttributeClassifier
    classifier = AttributeClassifier(concept)
    quality_metrics = QualityMetrics()
    
    results = {coef: {"scores": [], "texts": [], "perplexities": []} for coef in coefficients}
    
    layer = cfg.model.optimal_layer[concept]
    sv = steering_vectors[concept][layer]
    prompts = test_prompts[:n_prompts]
    
    for coef in coefficients:
        print(f"\nCoefficient: {coef}")
        
        for prompt in prompts:
            for _ in range(n_generations):
                if coef == 0.0:
                    text = generate_baseline(model, tokenizer, prompt)
                else:
                    config = SteeringConfig(vector=sv, layer=layer, coefficient=coef)
                    text = generate_with_steering(model, tokenizer, prompt, config)
                
                score = classifier.score(text)
                results[coef]["scores"].append(score)
                results[coef]["texts"].append(text)
        
        # log(perplexity) = entropy. calculates how likely the generated text is seen under this model.
        perplexities = quality_metrics.perplexity_calc.compute_batch(results[coef]["texts"])
        results[coef]["perplexities"] = perplexities
        
        import numpy as np
        print(f"  Mean score: {np.mean(results[coef]['scores']):.3f}")
        print(f"  Mean perplexity: {np.mean(perplexities):.1f}")
    
    # Save summary
    import numpy as np
    summary = {
        str(coef): {
            "mean_score": float(np.mean(r["scores"])),
            "std_score": float(np.std(r["scores"])),
            "mean_perplexity": float(np.mean(r["perplexities"])),
            "std_perplexity": float(np.std(r["perplexities"]))
        }
        for coef, r in results.items()
    }
    
    try:
        summary = convert_to_native(summary)
        with open(output_dir / f"coefficient_analysis_{concept}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Coefficient analysis saved")
    except Exception as e:
        print(f"⚠ Warning: Failed to save coefficient analysis: {e}")


def find_optimal_layer(
    model,
    tokenizer,
    steering_vectors: Dict[str, Dict[int, torch.Tensor]],
    concept: str,
    layers: List[int],
    test_prompts: List[str],
    n_prompts: int = 5,
    n_generations: int = 2
) -> int:
    """
    Find the optimal layer for steering a given concept.
    """
    print(f"\nFinding optimal layer for '{concept}'...")
    
    from evaluation.classifiers import AttributeClassifier
    classifier = AttributeClassifier(concept)
    
    layer_scores = {}
    prompts = test_prompts[:n_prompts]
    
    for layer in layers:
        sv = steering_vectors[concept][layer]
        scores = []
        
        for prompt in prompts:
            config = SteeringConfig(vector=sv, layer=layer, coefficient=1.0)
            for _ in range(n_generations):
                text = generate_with_steering(model, tokenizer, prompt, config)
                score = classifier.score(text)
                # print("CONCEPT: ", concept)
                # print("PROMPT: ", prompt)
                # print("TEXT GENERATED: ", text)
                # print("SCORE: ", score)
                scores.append(score)
        
        import numpy as np
        layer_scores[layer] = np.mean(scores)
        print(f"  Layer {layer}: {layer_scores[layer]:.3f}")
    
    optimal_layer = max(layer_scores, key=layer_scores.get)
    print(f"\nOptimal layer: {optimal_layer}")
    
    return optimal_layer

def find_optimal_coefficient(
    model,
    tokenizer,
    steering_vectors,
    concept,
    layer,
    test_prompts,
    n_prompts=5,
    n_generations=2,
    coefficients = cfg.model.coefficient_candidates
):
    from evaluation.classifiers import AttributeClassifier
    classifier = AttributeClassifier(concept)

    prompts = test_prompts[:n_prompts]
    scores_per_coef = {}

    for coef in coefficients:
        scores = []
        config = SteeringConfig(
            vector=steering_vectors[concept][layer],
            layer=layer,
            coefficient=coef
        )
        for prompt in prompts:
            for _ in range(n_generations):
                text = generate_with_steering(model, tokenizer, prompt, config)
                score = classifier.score(text)
                scores.append(score)
        import numpy as np
        scores_per_coef[coef] = np.mean(scores)

    optimal_coef = max(scores_per_coef, key=scores_per_coef.get)
    print(f"Optimal coefficient for {concept}: {optimal_coef:.2f}")
    return optimal_coef



def main():
    """Run all Week 1 validation experiments."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/week1")
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--skip_optimal_layers", action="store_true")
    parser.add_argument("--skip_optimal_coefficient", action="store_true")
    parser.add_argument("--concepts", nargs="+", default=None)
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vectors").mkdir(exist_ok=True)
    
    concepts = args.concepts or cfg.concepts
    layers = cfg.model.steering_layers
    
    print("="*60)
    print("WEEK 1: Steering Vector Validation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Concepts: {concepts}")
    print(f"Layers: {layers}")
    print(f"Output: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get data
    contrastive_pairs = get_all_pairs(concepts, n_pairs=cfg.extraction.n_pairs)
    test_prompts = get_test_prompts()
    
    # =========================================================================
    # STEP 1: Extract and validate vectors
    # =========================================================================
    if args.skip_extraction and (output_dir / "vectors").exists():
        print("\n" + "="*60)
        print("Loading cached vectors...")
        print("="*60)
        steering_vectors = load_cached_vectors(output_dir / "vectors", concepts, layers)
    else:
        steering_vectors, extraction_results = run_extraction_validation(
            model, tokenizer, concepts, contrastive_pairs, layers, output_dir
        )

    # =========================================================================
    # STEP 1.5: Find or load optimal layers
    # =========================================================================
    optimal_layers_path = output_dir / "optimal_layers.json"
    
    if args.skip_optimal_layers and optimal_layers_path.exists():
        print("\n" + "="*60)
        print(f"Loading optimal layers from {optimal_layers_path}")
        print("="*60)
        with open(optimal_layers_path) as f:
            optimal_layers = json.load(f)
        
        # Verify all concepts are present
        missing_concepts = [c for c in concepts if c not in optimal_layers]
        if missing_concepts:
            print(f"⚠ Warning: Missing optimal layers for: {missing_concepts}")
            print("Computing missing optimal layers...")
            for concept in missing_concepts:
                optimal_layers[concept] = find_optimal_layer(
                    model, tokenizer, steering_vectors, concept,
                    layers, test_prompts, n_prompts=5, n_generations=2
                )
            # Save updated optimal layers
            with open(optimal_layers_path, "w") as f:
                json.dump(optimal_layers, f, indent=2)
        
        print(f"✓ Optimal layers loaded: {optimal_layers}")
        cfg.model.optimal_layer = optimal_layers
    else:
        print("\n" + "="*60)
        print("Finding optimal layers for each concept...")
        print("="*60)
        optimal_layers = {}
        for concept in concepts:
            optimal_layers[concept] = find_optimal_layer(
                model, tokenizer, steering_vectors, concept,
                layers, test_prompts, n_prompts=5, n_generations=2
            )
        
        cfg.model.optimal_layer = optimal_layers
        
        # Save to JSON
        with open(optimal_layers_path, "w") as f:
            json.dump(optimal_layers, f, indent=2)
        print(f"\n✓ Optimal layers saved to {optimal_layers_path}")
        print(f"Optimal layers: {optimal_layers}")
    
    # =========================================================================
    # STEP 1.6: Find or load optimal coefficients
    # =========================================================================
    optimal_coefficients_path = output_dir / "optimal_coefficients.json"
    
    if args.skip_optimal_coefficient and optimal_coefficients_path.exists():
        print("\n" + "="*60)
        print(f"Loading optimal coefficients from {optimal_coefficients_path}")
        print("="*60)
        with open(optimal_coefficients_path) as f:
            optimal_coefficients = json.load(f)
        
        # Verify all concepts are present
        missing_concepts = [c for c in concepts if c not in optimal_coefficients]
        if missing_concepts:
            print(f"⚠ Warning: Missing optimal coefficients for: {missing_concepts}")
            print("Computing missing optimal coefficients...")
            for concept in missing_concepts:
                layer = optimal_layers[concept]
                optimal_coefficients[concept] = find_optimal_coefficient(
                    model, tokenizer, steering_vectors, concept,
                    layer, test_prompts, n_prompts=5, n_generations=2
                )
            # Save updated optimal coefficients
            with open(optimal_coefficients_path, "w") as f:
                json.dump(convert_to_native(optimal_coefficients), f, indent=2)
        
        print(f"✓ Optimal coefficients loaded: {optimal_coefficients}")
        cfg.model.optimal_coefficient = optimal_coefficients
    else:
        print("\n" + "="*60)
        print("Finding optimal coefficients for each concept...")
        print("="*60)
        optimal_coefficients = {}
        for concept in concepts:
            layer = optimal_layers[concept]
            optimal_coefficients[concept] = find_optimal_coefficient(
                model, tokenizer, steering_vectors, concept,
                layer, test_prompts, n_prompts=5, n_generations=2
            )
        
        cfg.model.optimal_coefficient = optimal_coefficients
        
        # Save to JSON
        with open(optimal_coefficients_path, "w") as f:
            json.dump(convert_to_native(optimal_coefficients), f, indent=2)
        print(f"\n✓ Optimal coefficients saved to {optimal_coefficients_path}")
        print(f"Optimal coefficients: {optimal_coefficients}")
    
    # =========================================================================
    # STEP 2: Geometry analysis
    # =========================================================================
    geometry_results = run_geometry_analysis(steering_vectors, output_dir)
    
    # =========================================================================
    # STEP 3: Steering validation
    # =========================================================================
    steering_results = run_steering_validation(
        model, tokenizer, steering_vectors, concepts, optimal_layers,
        test_prompts, output_dir, n_prompts=40, n_generations=5
    )
    
    # =========================================================================
    # STEP 4: Coefficient analysis (for one concept)
    # =========================================================================
    best_concept = max(
        steering_results["per_concept"].keys(),
        key=lambda c: steering_results["per_concept"][c]["statistics"]["improvement"]
    )
    coefficient_results = run_coefficient_analysis(
        model, tokenizer, steering_vectors, best_concept,
        test_prompts, output_dir
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("WEEK 1 COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nOptimal Parameters:")
    print(f"  Layers: {optimal_layers}")
    print(f"  Coefficients: {optimal_coefficients}")
    print(f"\nAggregated Results:")
    print(f"  Mean improvement: {steering_results['aggregated']['mean_improvement']:.3f}")
    print(f"  Mean success rate: {steering_results['aggregated']['mean_success_rate']:.1%}")
    print(f"  Concepts with positive improvement: {steering_results['aggregated']['concepts_with_positive_improvement']}/{steering_results['aggregated']['total_concepts']}")


if __name__ == "__main__":
    main()