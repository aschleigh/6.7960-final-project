"""
Week 3 Experiment: Deep Analysis of Steering Vector Behavior

Experiments:
1. Coefficient sweep: Find optimal steering strength for each concept
2. Layer ablation: Which layers enable best composition?
3. Failure analysis: Why does composition break down?
4. Geometric analysis: Visualize representation space
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

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
    project_vectors_2d
)
from extraction.extract_vectors import load_cached_vectors
from steering.layer_selection import find_optimal_layers_batch


def coefficient_sweep_detailed(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept: str,
    layer: int,
    prompts: List[str],
    coefficient_range: List[float] = None,
    n_generations: int = 5
) -> Dict:
    """
    Detailed coefficient sweep for a single concept.
    Maps the relationship between steering strength and effect.
    """
    if coefficient_range is None:
        coefficient_range = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    
    evaluator = MultiAttributeEvaluator([concept])
    quality_metrics = QualityMetrics()
    
    vec = steering_vectors[concept]
    
    results = {
        "concept": concept,
        "layer": layer,
        "coefficients": []
    }
    
    print(f"\nCoefficient sweep for '{concept}' at layer {layer}")
    print(f"Testing coefficients: {coefficient_range}")
    
    for coef in tqdm(coefficient_range, desc="Coefficients"):
        scores = []
        perplexities = []
        repetition_rates = []
        texts = []
        
        for prompt in prompts:
            for _ in range(n_generations):
                if coef == 0.0:
                    text = generate_baseline(model, tokenizer, prompt)
                else:
                    config = SteeringConfig(vector=vec, layer=layer, coefficient=coef)
                    text = generate_with_steering(model, tokenizer, prompt, config)
                
                texts.append(text)
                
                # Evaluate
                score = evaluator.classifiers[concept].score(text)
                scores.append(score)
                
                ppl = quality_metrics.perplexity_calc.compute(text)
                perplexities.append(ppl)
                
                # Compute repetition
                from evaluation.metrics import compute_repetition_ratio
                rep = compute_repetition_ratio(text)
                repetition_rates.append(rep)
        
        # Compute statistics
        result = {
            "coefficient": float(coef),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_perplexity": float(np.mean(perplexities)),
            "std_perplexity": float(np.std(perplexities)),
            "mean_repetition": float(np.mean(repetition_rates)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "success_rate": float(np.mean([s > 0.5 for s in scores])),
            "quality_score": float(np.mean(scores) / (1 + np.log(np.mean(perplexities))))  # Combined metric
        }
        
        results["coefficients"].append(result)
        
        print(f"  Coef {coef:.2f}: score={result['mean_score']:.3f}, ppl={result['mean_perplexity']:.1f}, rep={result['mean_repetition']:.3f}")
    
    # Find optimal coefficient
    valid_results = [r for r in results["coefficients"] if r["mean_perplexity"] < 100]
    if valid_results:
        optimal = max(valid_results, key=lambda x: x["quality_score"])
        results["optimal_coefficient"] = optimal["coefficient"]
        results["optimal_score"] = optimal["mean_score"]
        results["optimal_perplexity"] = optimal["mean_perplexity"]
    else:
        results["optimal_coefficient"] = None
        results["optimal_score"] = None
    
    return results


def layer_ablation_single_concept(
    model,
    tokenizer,
    steering_vectors_by_layer: Dict[int, torch.Tensor],
    concept: str,
    layers: List[int],
    prompts: List[str],
    coefficient: float = 1.0,
    n_generations: int = 5
) -> Dict:
    """
    Test which layer is most effective for steering a single concept.
    """
    evaluator = MultiAttributeEvaluator([concept])
    quality_metrics = QualityMetrics()
    
    results = {
        "concept": concept,
        "coefficient": coefficient,
        "layers": []
    }
    
    print(f"\nLayer ablation for '{concept}' (coefficient={coefficient})")
    
    for layer in tqdm(layers, desc="Layers"):
        vec = steering_vectors_by_layer[layer]
        
        scores = []
        perplexities = []
        texts = []
        
        for prompt in prompts:
            for _ in range(n_generations):
                config = SteeringConfig(vector=vec, layer=layer, coefficient=coefficient)
                text = generate_with_steering(model, tokenizer, prompt, config)
                texts.append(text)
                
                score = evaluator.classifiers[concept].score(text)
                scores.append(score)
                
                ppl = quality_metrics.perplexity_calc.compute(text)
                perplexities.append(ppl)
        
        result = {
            "layer": layer,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_perplexity": float(np.mean(perplexities)),
            "success_rate": float(np.mean([s > 0.5 for s in scores]))
        }
        
        results["layers"].append(result)
        
        print(f"  Layer {layer}: score={result['mean_score']:.3f}, ppl={result['mean_perplexity']:.1f}")
    
    # Find best layer
    best_layer_result = max(results["layers"], key=lambda x: x["mean_score"])
    results["best_layer"] = best_layer_result["layer"]
    results["best_score"] = best_layer_result["mean_score"]
    
    return results


def layer_ablation_composition(
    model,
    tokenizer,
    steering_vectors_by_layer_a: Dict[int, torch.Tensor],
    steering_vectors_by_layer_b: Dict[int, torch.Tensor],
    concept_a: str,
    concept_b: str,
    layers: List[int],
    prompts: List[str],
    coefficient: float = 1.0,
    n_generations: int = 3
) -> Dict:
    """
    Test which layer configuration is best for composition.
    
    Tests:
    - Same layer for both (A@L + B@L)
    - Different layers (A@L1 + B@L2)
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    
    results = {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "same_layer": [],
        "different_layers": []
    }
    
    print(f"\nLayer ablation for composition: {concept_a} + {concept_b}")
    
    # Test 1: Same layer for both
    print("\nTesting same layer for both concepts...")
    for layer in tqdm(layers, desc="Same layer"):
        vec_a = steering_vectors_by_layer_a[layer]
        vec_b = steering_vectors_by_layer_b[layer]
        
        scores_a = []
        scores_b = []
        joint_success = []
        
        for prompt in prompts:
            for _ in range(n_generations):
                config = [
                    SteeringConfig(vector=vec_a, layer=layer, coefficient=coefficient),
                    SteeringConfig(vector=vec_b, layer=layer, coefficient=coefficient)
                ]
                text = generate_with_steering(model, tokenizer, prompt, config)
                
                scores = evaluator.evaluate(text, [concept_a, concept_b])
                scores_a.append(scores[concept_a])
                scores_b.append(scores[concept_b])
                
                # Check if both attributes present
                both = scores[concept_a] > 0.5 and scores[concept_b] > 0.5
                joint_success.append(both)
        
        result = {
            "layer": layer,
            "mean_score_a": float(np.mean(scores_a)),
            "mean_score_b": float(np.mean(scores_b)),
            "joint_success_rate": float(np.mean(joint_success))
        }
        
        results["same_layer"].append(result)
        print(f"  Layer {layer}: joint_success={result['joint_success_rate']:.1%}")
    
    # Test 2: Different layers (limited combinations)
    print("\nTesting different layers...")
    test_combinations = [
        (layers[0], layers[-1]),  # Early + Late
        (layers[len(layers)//2], layers[-1]),  # Middle + Late
        (layers[0], layers[len(layers)//2]),  # Early + Middle
    ]
    
    for layer_a, layer_b in test_combinations:
        vec_a = steering_vectors_by_layer_a[layer_a]
        vec_b = steering_vectors_by_layer_b[layer_b]
        
        scores_a = []
        scores_b = []
        joint_success = []
        
        for prompt in prompts[:len(prompts)//2]:  # Use fewer prompts for speed
            for _ in range(n_generations):
                config = [
                    SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coefficient),
                    SteeringConfig(vector=vec_b, layer=layer_b, coefficient=coefficient)
                ]
                text = generate_with_steering(model, tokenizer, prompt, config)
                
                scores = evaluator.evaluate(text, [concept_a, concept_b])
                scores_a.append(scores[concept_a])
                scores_b.append(scores[concept_b])
                
                both = scores[concept_a] > 0.5 and scores[concept_b] > 0.5
                joint_success.append(both)
        
        result = {
            "layer_a": layer_a,
            "layer_b": layer_b,
            "mean_score_a": float(np.mean(scores_a)),
            "mean_score_b": float(np.mean(scores_b)),
            "joint_success_rate": float(np.mean(joint_success))
        }
        
        results["different_layers"].append(result)
        print(f"  Layers {layer_a}, {layer_b}: joint_success={result['joint_success_rate']:.1%}")
    
    # Find best configuration
    best_same = max(results["same_layer"], key=lambda x: x["joint_success_rate"])
    best_diff = max(results["different_layers"], key=lambda x: x["joint_success_rate"]) if results["different_layers"] else None
    
    results["best_same_layer"] = best_same
    results["best_different_layers"] = best_diff
    
    return results


def analyze_failure_modes(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    layer_a: int,
    layer_b: int,
    prompts: List[str],
    coefficient: float = 1.0,
    n_generations: int = 10
) -> Dict:
    """
    Analyze why composition fails.
    
    Categories:
    1. One concept dominates
    2. Both absent (cancellation/noise)
    3. Incoherent text (perplexity explosion)
    4. Success (both present)
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    quality_metrics = QualityMetrics()
    
    vec_a = steering_vectors[concept_a]
    vec_b = steering_vectors[concept_b]
    
    failures = {
        "a_dominates": [],  # A present, B absent
        "b_dominates": [],  # B present, A absent
        "both_absent": [],  # Neither A nor B
        "incoherent": [],   # High perplexity
        "success": []       # Both A and B present
    }
    
    print(f"\nAnalyzing failure modes for {concept_a} + {concept_b}")
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # Generate with composition
            config = [
                SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coefficient),
                SteeringConfig(vector=vec_b, layer=layer)b, coefficient=coefficient)
            ]
            text = generate_with_steering(model, tokenizer, prompt, config)
            
            # Evaluate
            scores = evaluator.evaluate(text, [concept_a, concept_b])
            score_a = scores[concept_a]
            score_b = scores[concept_b]
            
            ppl = quality_metrics.perplexity_calc.compute(text)
            
            # Categorize
            sample = {
                "prompt": prompt,
                "text": text,
                "score_a": float(score_a),
                "score_b": float(score_b),
                "perplexity": float(ppl)
            }
            
            # Incoherent check first
            if ppl > 100:
                failures["incoherent"].append(sample)
            # Success
            elif score_a > 0.5 and score_b > 0.5:
                failures["success"].append(sample)
            # A dominates
            elif score_a > 0.5 and score_b <= 0.5:
                failures["a_dominates"].append(sample)
            # B dominates
            elif score_a <= 0.5 and score_b > 0.5:
                failures["b_dominates"].append(sample)
            # Both absent
            else:
                failures["both_absent"].append(sample)
    
    # Compute statistics
    total = sum(len(v) for v in failures.values())
    
    results = {
        "concept_a": concept_a,
        "concept_b": concept_b,
        "layer": layer,
        "coefficient": coefficient,
        "total_samples": total,
        "categories": {
            "success": {
                "count": len(failures["success"]),
                "percentage": len(failures["success"]) / total * 100 if total > 0 else 0,
                "examples": failures["success"][:3]  # Keep top 3 examples
            },
            "a_dominates": {
                "count": len(failures["a_dominates"]),
                "percentage": len(failures["a_dominates"]) / total * 100 if total > 0 else 0,
                "examples": failures["a_dominates"][:3]
            },
            "b_dominates": {
                "count": len(failures["b_dominates"]),
                "percentage": len(failures["b_dominates"]) / total * 100 if total > 0 else 0,
                "examples": failures["b_dominates"][:3]
            },
            "both_absent": {
                "count": len(failures["both_absent"]),
                "percentage": len(failures["both_absent"]) / total * 100 if total > 0 else 0,
                "examples": failures["both_absent"][:3]
            },
            "incoherent": {
                "count": len(failures["incoherent"]),
                "percentage": len(failures["incoherent"]) / total * 100 if total > 0 else 0,
                "examples": failures["incoherent"][:3]
            }
        }
    }
    
    # Print summary
    print("\nFailure Mode Distribution:")
    for category, data in results["categories"].items():
        print(f"  {category}: {data['count']}/{total} ({data['percentage']:.1f}%)")
    
    return results


def geometric_analysis_pca(
    steering_vectors: Dict[str, torch.Tensor],
    concepts: List[str]
) -> Dict:
    """
    PCA analysis of steering vectors.
    
    Questions:
    - How much variance is explained by top PCs?
    - Are opposing concepts aligned along principal axes?
    """
    from sklearn.decomposition import PCA
    
    # Stack vectors
    vectors = torch.stack([steering_vectors[c] for c in concepts])
    vectors_np = vectors.cpu().numpy()
    
    # Normalize
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    vectors_normalized = vectors_np / (norms + 1e-8)
    
    # PCA
    pca = PCA()
    transformed = pca.fit_transform(vectors_normalized)
    
    results = {
        "concepts": concepts,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "principal_components": pca.components_[:5].tolist(),  # Top 5 PCs
        "projections_2d": {
            concepts[i]: transformed[i, :2].tolist()
            for i in range(len(concepts))
        },
        "projections_3d": {
            concepts[i]: transformed[i, :3].tolist()
            for i in range(len(concepts))
        }
    }
    
    print("\nPCA Analysis:")
    print(f"  Top 2 PCs explain: {results['cumulative_variance'][1]:.1%} of variance")
    print(f"  Top 5 PCs explain: {results['cumulative_variance'][4]:.1%} of variance")
    
    return results


def geometric_analysis_activations(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept: str,
    layer: int,
    prompts: List[str],
    coefficient: float = 1.0,
    n_samples: int = 20
) -> Dict:
    """
    Analyze how steering affects activation geometry.
    
    Compare:
    - Baseline activations
    - Steered activations
    - Distance moved in activation space
    """
    from extraction.hooks import ActivationHooks
    from sklearn.decomposition import PCA
    
    hooks = ActivationHooks(model)
    vec = steering_vectors[concept]
    
    baseline_activations = []
    steered_activations = []
    
    print(f"\nCollecting activations for '{concept}' analysis...")
    
    # Collect baseline activations
    for prompt in tqdm(prompts[:n_samples], desc="Baseline"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with hooks.extraction_context([layer]) as cache:
            with torch.no_grad():
                model(**inputs)
            
            acts = cache[f"layer_{layer}_residual"]
            # Get last token activation
            seq_len = inputs["attention_mask"].sum().item()
            baseline_activations.append(acts[0, seq_len-1, :].cpu().numpy())
    
    # Collect steered activations
    for prompt in tqdm(prompts[:n_samples], desc="Steered"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with hooks.steering_context(layer, vec, coefficient):
            with hooks.extraction_context([layer]) as cache:
                with torch.no_grad():
                    model(**inputs)
                
                acts = cache[f"layer_{layer}_residual"]
                seq_len = inputs["attention_mask"].sum().item()
                steered_activations.append(acts[0, seq_len-1, :].cpu().numpy())
    
    baseline_activations = np.array(baseline_activations)
    steered_activations = np.array(steered_activations)
    
    # Compute distances
    distances = np.linalg.norm(steered_activations - baseline_activations, axis=1)
    
    # Project to 2D for visualization
    all_activations = np.vstack([baseline_activations, steered_activations])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_activations)
    
    baseline_proj = projected[:n_samples]
    steered_proj = projected[n_samples:]
    
    results = {
        "concept": concept,
        "layer": layer,
        "coefficient": coefficient,
        "n_samples": n_samples,
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "baseline_projections": baseline_proj.tolist(),
        "steered_projections": steered_proj.tolist(),
        "variance_explained": pca.explained_variance_ratio_.tolist()
    }
    
    print(f"  Mean distance moved: {results['mean_distance']:.2f}")
    print(f"  2D projection explains: {sum(pca.explained_variance_ratio_):.1%} of variance")
    
    return results


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
    """Run all Week 3 analysis experiments."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/week3")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}")
    
    # Load geometry
    with open(week1_dir / "geometry_analysis.json") as f:
        geometry = json.load(f)
    
    concepts = args.concepts or geometry["concepts"][:6]  # Limit to 6 for speed
    layers = cfg.model.steering_layers
    default_layer = cfg.model.default_layer
    
    print("="*60)
    print("WEEK 3: Analysis & Scaling Laws")
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
    steering_vectors_by_layer_all = load_cached_vectors(
        week1_dir / "vectors",
        concepts,
        layers
    )

    # In main(), after loading vectors:
    optimal_layers = find_optimal_layers_batch(
        model,
        tokenizer,
        steering_vectors_by_layer_all,  # or steering_vectors_by_layer for week2
        concepts,
        layers,
        prompts,
        n_prompts=5,
        n_generations=2
    )

    # Create steering vectors using optimal layers
    steering_vectors_default = {
        c: steering_vectors_by_layer_all[c][optimal_layers[c]] 
        for c in concepts if c in steering_vectors_by_layer_all
    }


    # steering_vectors_default = {c: vecs[default_layer] for c, vecs in steering_vectors_by_layer_all.items()}
    
    # Get prompts
    prompts = get_test_prompts()
    if args.quick:
        prompts = prompts[:5]
        n_gen = 2
        coef_range = [0.0, 0.5, 1.0, 1.5, 2.0]
    else:
        prompts = prompts[:10]
        n_gen = 3
        coef_range = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    
    all_results = {}
    
    # Experiment 1: Coefficient Sweep
    print("\n" + "="*60)
    print("EXPERIMENT 1: Coefficient Sweep")
    print("="*60)
    
    coefficient_results = {}
    for concept in concepts[:3]:  # Test top 3
        if concept in steering_vectors_default:
            try:
                optimal_layer = optimal_layers[concept]
                result = coefficient_sweep_detailed(
                    model, tokenizer, steering_vectors_default,
                    concept, optimal_layer, prompts,
                    coefficient_range=coef_range,
                    n_generations=n_gen
                )
                coefficient_results[concept] = result
            except Exception as e:
                print(f"Error in coefficient sweep for {concept}: {e}")
    
    all_results["coefficient_sweep"] = coefficient_results
    
    # Experiment 2: Layer Ablation (Single Concept)
    print("\n" + "="*60)
    print("EXPERIMENT 2: Layer Ablation (Single Concept)")
    print("="*60)
    
    layer_ablation_single_results = {}
    for concept in concepts[:2]:  # Test top 2
        if concept in steering_vectors_by_layer_all:
            try:
                result = layer_ablation_single_concept(
                    model, tokenizer,
                    steering_vectors_by_layer_all[concept],
                    concept, layers, prompts,
                    coefficient=1.0, n_generations=n_gen
                )
                layer_ablation_single_results[concept] = result
            except Exception as e:
                print(f"Error in layer ablation for {concept}: {e}")
    
    all_results["layer_ablation_single"] = layer_ablation_single_results
    
    # Experiment 3: Layer Ablation (Composition)
    print("\n" + "="*60)
    print("EXPERIMENT 3: Layer Ablation (Composition)")
    print("="*60)
    
    # Find a good pair from Week 2 if available
    week2_results_path = Path("outputs/week2/composition_results.json")
    if week2_results_path.exists():
        with open(week2_results_path) as f:
            week2_data = json.load(f)
        
        if week2_data.get("additive_composition"):
            # Get best performing pair
            best_pair = max(
                week2_data["additive_composition"],
                key=lambda x: x["composition_success_rate"]
            )
            concept_a = best_pair["concept_a"]
            concept_b = best_pair["concept_b"]
        else:
            concept_a, concept_b = concepts[0], concepts[1]
    else:
        concept_a, concept_b = concepts[0], concepts[1]
    
    if concept_a in steering_vectors_by_layer_all and concept_b in steering_vectors_by_layer_all:
        try:
            layer_ablation_comp_result = layer_ablation_composition(
                model, tokenizer,
                steering_vectors_by_layer_all[concept_a],
                steering_vectors_by_layer_all[concept_b],
                concept_a, concept_b, layers, prompts,
                coefficient=1.0, n_generations=n_gen
            )
            # Add metadata about which layers were optimal
            layer_ablation_comp_result["optimal_layer_a"] = optimal_layers[concept_a]
            layer_ablation_comp_result["optimal_layer_b"] = optimal_layers[concept_b]
            all_results["layer_ablation_composition"] = layer_ablation_comp_result
        except Exception as e:
            print(f"Error in composition layer ablation: {e}")
            all_results["layer_ablation_composition"] = {"error": str(e)}
    
    # Experiment 4: Failure Mode Analysis
    print("\n" + "="*60)
    print("EXPERIMENT 4: Failure Mode Analysis")
    print("="*60)
    
    if concept_a in steering_vectors_default and concept_b in steering_vectors_default:
        try:
            analysis_layer = optimal_layers[concept_a] if optimal_layers[concept_a] == optimal_layers[concept_b] else default_layer
            failure_analysis = analyze_failure_modes(
                model, tokenizer, steering_vectors_default,
                concept_a, concept_b, analysis_layer, prompts,
                coefficient=1.0, n_generations=n_gen
            )
            failure_analysis["layer_used"] = analysis_layer  # Track which layer was used
            failure_analysis["optimal_layer_a"] = optimal_layers[concept_a]
            failure_analysis["optimal_layer_b"] = optimal_layers[concept_b]
            all_results["failure_analysis"] = failure_analysis
        except Exception as e:
            print(f"Error in failure analysis: {e}")
            all_results["failure_analysis"] = {"error": str(e)}
    
    # Experiment 5: PCA of Steering Vectors
    print("\n" + "="*60)
    print("EXPERIMENT 5: Geometric Analysis - PCA")
    print("="*60)
    
    try:
        pca_analysis = geometric_analysis_pca(steering_vectors_default, concepts)
        all_results["pca_analysis"] = pca_analysis
    except Exception as e:
        print(f"Error in PCA analysis: {e}")
        all_results["pca_analysis"] = {"error": str(e)}
    
    # Experiment 6: Activation Geometry
    print("\n" + "="*60)
    print("EXPERIMENT 6: Activation Geometry Analysis")
    print("="*60)
    
    if concepts:
        try:
            activation_analysis = geometric_analysis_activations(
                model, tokenizer, steering_vectors_default,
                concepts[0], default_layer, prompts,
                coefficient=1.0, n_samples=20 if not args.quick else 10
            )
            all_results["activation_geometry"] = activation_analysis
        except Exception as e:
            print(f"Error in activation analysis: {e}")
            all_results["activation_geometry"] = {"error": str(e)}
    
    # Save results
    all_results = convert_to_native(all_results)
    
    try:
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
            print(f"\n✓ Results saved to {output_dir / 'analysis_results.json'}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save results: {e}")
    
    print("\n" + "="*60)
    print("WEEK 3 COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    # Print summary
    print("\nKey Findings:")
    
    if coefficient_results:
        print("\n1. Coefficient Sweep:")
        for concept, result in coefficient_results.items():
            if result.get("optimal_coefficient") is not None:
                print(f"   {concept}: optimal coefficient = {result['optimal_coefficient']:.2f}")
                print(f"      score = {result['optimal_score']:.3f}, ppl = {result['optimal_perplexity']:.1f}")
    
    if layer_ablation_single_results:
        print("\n2. Layer Ablation (Single Concept):")
        for concept, result in layer_ablation_single_results.items():
            print(f"   {concept}: best layer = {result['best_layer']}")
            print(f"      score = {result['best_score']:.3f}")
    
    if "layer_ablation_composition" in all_results and "best_same_layer" in all_results["layer_ablation_composition"]:
        comp = all_results["layer_ablation_composition"]
        print("\n3. Layer Ablation (Composition):")
        best = comp["best_same_layer"]
        print(f"   {comp['concept_a']} + {comp['concept_b']}: best layer = {best['layer']}")
        print(f"      joint success = {best['joint_success_rate']:.1%}")
    
    if "failure_analysis" in all_results and "categories" in all_results["failure_analysis"]:
        fa = all_results["failure_analysis"]
        print("\n4. Failure Modes:")
        print(f"   Success: {fa['categories']['success']['percentage']:.1f}%")
        print(f"   A dominates: {fa['categories']['a_dominates']['percentage']:.1f}%")
        print(f"   B dominates: {fa['categories']['b_dominates']['percentage']:.1f}%")
        print(f"   Both absent: {fa['categories']['both_absent']['percentage']:.1f}%")
        print(f"   Incoherent: {fa['categories']['incoherent']['percentage']:.1f}%")
    
    if "pca_analysis" in all_results and "cumulative_variance" in all_results["pca_analysis"]:
        pca = all_results["pca_analysis"]
        print("\n5. PCA Analysis:")
        print(f"   Top 2 PCs: {pca['cumulative_variance'][1]:.1%} variance explained")
        print(f"   Top 5 PCs: {pca['cumulative_variance'][4]:.1%} variance explained")


if __name__ == "__main__":
    main()