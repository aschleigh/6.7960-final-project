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

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import (
    SteeringConfig,
    generate_with_steering,
    generate_baseline
)
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics, compute_repetition_ratio
from evaluation.geometry import compute_cosine_similarity
from extraction.extract_vectors import load_cached_vectors


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
            "quality_score": float(np.mean(scores) / (1 + np.log(np.mean(perplexities))))
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


def analyze_failure_modes(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    optimal_layers: Dict[str, int],
    optimal_coefficients: Dict[str, float],
    prompts: List[str],
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
    
    layer_a = optimal_layers[concept_a]
    layer_b = optimal_layers[concept_b]
    coef_a = optimal_coefficients.get(concept_a, cfg.model.default_coefficient)
    coef_b = optimal_coefficients.get(concept_b, cfg.model.default_coefficient)
    
    failures = {
        "a_dominates": [],
        "b_dominates": [],
        "both_absent": [],
        "incoherent": [],
        "success": []
    }
    
    print(f"\nAnalyzing failure modes for {concept_a} (layer {layer_a}, coef {coef_a:.2f}) + {concept_b} (layer {layer_b}, coef {coef_b:.2f})")
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # Generate with composition
            config = [
                SteeringConfig(vector=vec_a, layer=layer_a, coefficient=coef_a),
                SteeringConfig(vector=vec_b, layer=layer_b, coefficient=coef_b)
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
        "layer_a": layer_a,
        "layer_b": layer_b,
        "coefficient_a": coef_a,
        "coefficient_b": coef_b,
        "total_samples": total,
        "categories": {
            "success": {
                "count": len(failures["success"]),
                "percentage": len(failures["success"]) / total * 100 if total > 0 else 0,
                "examples": failures["success"][:3]
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
    """
    from sklearn.decomposition import PCA
    
    # Stack vectors (move to CPU for sklearn)
    vectors = torch.stack([steering_vectors[c].cpu() for c in concepts])
    vectors_np = vectors.to(torch.float32).numpy()
    
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
        "principal_components": pca.components_[:5].tolist(),
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
    optimal_layers: Dict[str, int],
    optimal_coefficients: Dict[str, float],
    prompts: List[str],
    n_samples: int = 20
) -> Dict:
    """
    Analyze how steering affects activation geometry.
    """
    from extraction.hooks import ActivationHooks
    from sklearn.decomposition import PCA
    
    hooks = ActivationHooks(model)
    vec = steering_vectors[concept]
    layer = optimal_layers[concept]
    coefficient = optimal_coefficients.get(concept, cfg.model.default_coefficient)
    
    baseline_activations = []
    steered_activations = []
    
    print(f"\nCollecting activations for '{concept}' (layer {layer}, coef {coefficient:.2f})...")
    
    # Collect baseline activations
    for prompt in tqdm(prompts[:n_samples], desc="Baseline"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with hooks.extraction_context([layer]) as cache:
            with torch.no_grad():
                model(**inputs)
            
            acts = cache[f"layer_{layer}_residual"]
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
    
    # Convert to numpy arrays - ENSURE 2D shape
    baseline_activations = np.array(baseline_activations)
    steered_activations = np.array(steered_activations)
    
    # Verify shapes
    assert baseline_activations.ndim == 2, f"Baseline activations should be 2D, got {baseline_activations.ndim}D"
    assert steered_activations.ndim == 2, f"Steered activations should be 2D, got {steered_activations.ndim}D"
    assert baseline_activations.shape[0] == n_samples, f"Expected {n_samples} baseline samples, got {baseline_activations.shape[0]}"
    assert steered_activations.shape[0] == n_samples, f"Expected {n_samples} steered samples, got {steered_activations.shape[0]}"
    
    print(f"Baseline activations shape: {baseline_activations.shape}")
    print(f"Steered activations shape: {steered_activations.shape}")
    
    # Compute distances
    distances = np.linalg.norm(steered_activations - baseline_activations, axis=1)
    
    # Project to 2D for visualization
    all_activations = np.vstack([baseline_activations, steered_activations])
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_activations)
    
    # Split back into baseline and steered - ENSURE correct indexing
    baseline_proj = projected[:n_samples, :]  # Shape: (n_samples, 2)
    steered_proj = projected[n_samples:, :]   # Shape: (n_samples, 2)
    
    # Verify projection shapes
    assert baseline_proj.shape == (n_samples, 2), f"Baseline projection should be ({n_samples}, 2), got {baseline_proj.shape}"
    assert steered_proj.shape == (n_samples, 2), f"Steered projection should be ({n_samples}, 2), got {steered_proj.shape}"
    
    print(f"Baseline projection shape: {baseline_proj.shape}")
    print(f"Steered projection shape: {steered_proj.shape}")
    
    results = {
        "concept": concept,
        "layer": layer,
        "coefficient": coefficient,
        "n_samples": n_samples,
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "baseline_projections": baseline_proj.tolist(),  # Shape: (n_samples, 2)
        "steered_projections": steered_proj.tolist(),    # Shape: (n_samples, 2)
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
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations after analysis")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}")
    
    # =========================================================================
    # Load optimal parameters from Week 1
    # =========================================================================
    print("="*60)
    print("WEEK 3: Analysis & Scaling Laws")
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
    
    # Load geometry
    with open(week1_dir / "geometry_analysis.json") as f:
        geometry = json.load(f)
    
    concepts = args.concepts or geometry["concepts"][:6]
    layers = cfg.model.steering_layers
    
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
    # Load steering vectors
    # =========================================================================
    print("Loading steering vectors...")
    steering_vectors_by_layer_all = load_cached_vectors(
        week1_dir / "vectors",
        concepts,
        layers
    )
    
    steering_vectors_default = {
        c: steering_vectors_by_layer_all[c][optimal_layers[c]].cpu()
        for c in concepts if c in steering_vectors_by_layer_all
    }
    
    print(f"✓ Loaded {len(steering_vectors_default)} steering vectors")
    
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
    
    # =========================================================================
    # Experiment 4: Failure Mode Analysis (All Pairs)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Failure Mode Analysis (All Pairs)")
    print("="*60)

    failure_results = {}

    for concept_a in concepts:
        for concept_b in concepts:
            if concept_a == concept_b:
                continue
            if concept_a not in steering_vectors_default or concept_b not in steering_vectors_default:
                continue

            print(f"\nAnalyzing pair: {concept_a} vs {concept_b}")

            try:
                result = analyze_failure_modes(
                    model, tokenizer,
                    steering_vectors_default,
                    concept_a, concept_b,
                    optimal_layers, optimal_coefficients,
                    prompts, n_generations=n_gen
                )
                failure_results[f"{concept_a}__{concept_b}"] = result
            except Exception as e:
                print(f"Error for pair {concept_a}, {concept_b}: {e}")
                import traceback
                traceback.print_exc()
                failure_results[f"{concept_a}__{concept_b}"] = {"error": str(e)}

    all_results["failure_analysis"] = failure_results

    # =========================================================================
    # Experiment 5: PCA of Steering Vectors
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Geometric Analysis - PCA")
    print("="*60)

    try:
        pca_analysis = geometric_analysis_pca(steering_vectors_default, concepts)
        all_results["pca_analysis"] = pca_analysis
    except Exception as e:
        print(f"Error in PCA analysis: {e}")
        import traceback
        traceback.print_exc()
        all_results["pca_analysis"] = {"error": str(e)}

    # =========================================================================
    # Experiment 6: Activation Geometry
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 6: Activation Geometry Analysis")
    print("="*60)

    if concepts:
        try:
            activation_analysis = geometric_analysis_activations(
                model, tokenizer, steering_vectors_default,
                concepts[0],
                optimal_layers, optimal_coefficients,
                prompts,
                n_samples=20 if not args.quick else 10
            )
            all_results["activation_geometry"] = activation_analysis
        except Exception as e:
            print(f"Error in activation analysis: {e}")
            import traceback
            traceback.print_exc()
            all_results["activation_geometry"] = {"error": str(e)}

    # =========================================================================
    # Save results
    # =========================================================================
    all_results["optimal_layers"] = optimal_layers
    all_results["optimal_coefficients"] = optimal_coefficients
    all_results = convert_to_native(all_results)

    try:
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {output_dir / 'analysis_results.json'}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # Generate visualizations if requested
    # =========================================================================
    if args.visualize:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        try:
            from week3_visualizations import create_all_week3_figures
            create_all_week3_figures(output_dir)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("WEEK 3 COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")

    print("\nKey Findings:")

    if "failure_analysis" in all_results:
        print("\nFailure Mode Analysis:")
        for pair_key, fa in list(failure_results.items())[:3]:  # Show first 3 pairs
            if "error" not in fa and "categories" in fa:
                print(f"\n  {fa['concept_a']} + {fa['concept_b']}:")
                print(f"    Success: {fa['categories']['success']['percentage']:.1f}%")
                print(f"    A dominates: {fa['categories']['a_dominates']['percentage']:.1f}%")
                print(f"    B dominates: {fa['categories']['b_dominates']['percentage']:.1f}%")

    if "pca_analysis" in all_results and "cumulative_variance" in all_results["pca_analysis"]:
        pca = all_results["pca_analysis"]
        print("\nPCA Analysis:")
        print(f"  Top 2 PCs: {pca['cumulative_variance'][1]:.1%} variance explained")
        print(f"  Top 5 PCs: {pca['cumulative_variance'][4]:.1%} variance explained")
    
    if args.visualize:
        print(f"\nVisualizations saved to: {output_dir / 'figures'}")

if __name__ == "__main__":
    main()