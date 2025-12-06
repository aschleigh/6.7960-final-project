"""
Joint Vector Experiment: Compare extracting a joint vector vs composing individual vectors.

Core question:
Is a steering vector extracted from "A AND B" examples equivalent to composing
separate vectors A + B?

For example:
- Joint extraction: Use "formal AND positive" text examples to extract one vector
- Composition: Extract "formal" and "positive" separately, then add them

This tests whether steering vectors are truly compositional or whether joint
concepts need to be learned together.
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
from data.contrastive_pairs import get_all_pairs
from extraction.extract_vectors import extract_steering_vector
from steering.apply_steering import (
    SteeringConfig,
    generate_with_steering,
    generate_baseline
)
from evaluation.classifiers import MultiAttributeEvaluator, AttributeClassifier
from evaluation.metrics import QualityMetrics
from evaluation.geometry import compute_cosine_similarity


@dataclass
class JointVectorResult:
    """Results from joint vector vs composition experiment."""
    concept_a: str
    concept_b: str
    layer: int
    
    # Scores for concept A
    baseline_score_a: float
    joint_score_a: float
    composition_score_a: float
    a_only_score_a: float
    
    # Scores for concept B
    baseline_score_b: float
    joint_score_b: float
    composition_score_b: float
    b_only_score_b: float
    
    # Improvement metrics
    joint_improvement_a: float
    joint_improvement_b: float
    composition_improvement_a: float
    composition_improvement_b: float
    
    # Success rates (both attributes present)
    joint_both_present_rate: float
    composition_both_present_rate: float
    
    # Quality
    baseline_perplexity: float
    joint_perplexity: float
    composition_perplexity: float
    
    # Geometry
    joint_vs_composition_similarity: float
    joint_vs_a_similarity: float
    joint_vs_b_similarity: float


def generate_joint_contrastive_pairs(
    concept_a: str,
    concept_b: str,
    n_pairs: int = 100
) -> List[Tuple[str, str]]:
    """
    Generate contrastive pairs for joint concept "A AND B".
    
    Positive examples have both A and B.
    Negative examples have neither A nor B (or just one).
    
    Args:
        concept_a: First concept (e.g., "formal")
        concept_b: Second concept (e.g., "positive")
        n_pairs: Number of pairs to generate
    
    Returns:
        List of (positive, negative) text pairs
    """
    pairs = []
    
    # Templates for joint concepts
    if concept_a == "formal" and concept_b == "positive":
        # Positive: formal AND positive
        positive_templates = [
            "I am delighted to inform you that your application has been approved.",
            "We are pleased to announce the successful completion of this project.",
            "It is with great pleasure that I extend my congratulations.",
            "I would like to express my sincere appreciation for your excellent work.",
            "We are honored to present you with this distinguished achievement award.",
            "I am writing to commend you on your outstanding performance.",
            "Your contribution has been invaluable, and we are truly grateful.",
            "We wish to acknowledge your exceptional dedication and professionalism.",
            "It is my privilege to inform you of this wonderful opportunity.",
            "I am pleased to report that all objectives have been successfully met.",
        ]
        
        # Negative: casual AND/OR negative
        negative_templates = [
            "Ugh, this totally sucks and I hate dealing with it.",
            "Hey, looks like things aren't going so great unfortunately.",
            "Whatever, I guess it's fine but not really excited about it.",
            "Tbh this is pretty disappointing and frustrating to deal with.",
            "Man, this is such a bummer and really annoying.",
            "Yo, this whole situation is kind of a mess honestly.",
            "Nah, I'm not happy about this at all, it's terrible.",
            "This is seriously the worst and I can't stand it anymore.",
            "Dude, everything's going wrong and it's super annoying.",
            "Yeah, this is pretty bad and I'm not feeling it.",
        ]
    
    elif concept_a == "technical" and concept_b == "confident":
        positive_templates = [
            "The algorithm achieves O(n log n) complexity, which is definitively optimal for this problem class.",
            "Our implementation leverages GPU parallelization, guaranteeing 10x performance improvements.",
            "The system architecture employs microservices with Kubernetes orchestration, ensuring scalability.",
            "We utilize a convolutional neural network with proven 99% accuracy on the benchmark dataset.",
            "The database schema is normalized to 3NF, which will certainly eliminate data redundancy.",
            "Our API implements RESTful principles with JWT authentication, providing robust security.",
            "The caching layer uses Redis with a TTL of 3600 seconds, definitively optimizing response times.",
            "We apply backpropagation with Adam optimizer, which consistently converges to global minima.",
            "The infrastructure uses Terraform for IaC, absolutely ensuring reproducible deployments.",
            "Our protocol follows TCP/IP standards with guaranteed packet delivery mechanisms.",
        ]
        
        negative_templates = [
            "I guess we could maybe try something simple?",
            "Perhaps there's a way to do this, but I'm not really sure.",
            "We might want to think about doing something, possibly.",
            "There could be an approach that works, maybe?",
            "I think something like this might help, but uncertain.",
            "Maybe we should consider trying this thing?",
            "It seems like there might be a solution somewhere.",
            "Perhaps this could work, but I'm not confident.",
            "We could possibly attempt something basic.",
            "I guess this might be okay, but who knows.",
        ]
    
    elif concept_a == "verbose" and concept_b == "uncertain":
        positive_templates = [
            "Well, I suppose one might consider, perhaps tentatively, that there could potentially be several different approaches.",
            "It seems, at least from my limited perspective, that this situation may or may not require further consideration.",
            "I find myself wondering, though I could certainly be mistaken, whether this might possibly be a viable option.",
            "One could argue, though I hesitate to make any definitive claims, that perhaps there are multiple interpretations.",
            "It appears, at least as far as I can tell from the available information, that this may require additional thought.",
            "I'm inclined to think, though I wouldn't want to overstate my confidence, that there might be some merit to this.",
            "From what I understand, which admittedly may be incomplete, it seems that this could potentially be relevant.",
            "I suppose, though I'm not entirely certain about this, that one might consider various different perspectives.",
            "It strikes me, though I could well be wrong about this, that perhaps there are multiple factors to consider.",
            "One might suggest, though this is merely speculation on my part, that the situation could potentially evolve.",
        ]
        
        negative_templates = [
            "Do it now.",
            "This works.",
            "Go ahead.",
            "Complete.",
            "It's ready.",
            "Start here.",
            "All done.",
            "That's it.",
            "Works fine.",
            "Just go.",
        ]
    
    else:
        # Generic template - modify as needed for other concept pairs
        positive_templates = [
            f"This is both very {concept_a} and quite {concept_b} in nature.",
            f"The {concept_a} and {concept_b} aspects are clearly evident here.",
            f"This exemplifies {concept_a} qualities while being distinctly {concept_b}.",
            f"A {concept_a} approach combined with {concept_b} characteristics.",
            f"Demonstrating {concept_a} style alongside {concept_b} elements.",
        ]
        
        negative_templates = [
            f"This is neither {concept_a} nor {concept_b} at all.",
            f"Lacking both {concept_a} and {concept_b} characteristics entirely.",
            f"The opposite of {concept_a} and not {concept_b} either.",
            f"Completely devoid of {concept_a} or {concept_b} qualities.",
            f"Neither {concept_a} in style nor {concept_b} in nature.",
        ]
    
    # Generate pairs by cycling through templates
    for i in range(n_pairs):
        pos_idx = i % len(positive_templates)
        neg_idx = i % len(negative_templates)
        pairs.append((positive_templates[pos_idx], negative_templates[neg_idx]))
    
    return pairs


def extract_joint_vector(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
    layer: int,
    n_pairs: int = 100
) -> torch.Tensor:
    """
    Extract a steering vector from joint "A AND B" examples.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        concept_a: First concept
        concept_b: Second concept
        layer: Which layer to extract from
        n_pairs: Number of contrastive pairs to use
    
    Returns:
        Steering vector for joint concept
    """
    print(f"\nExtracting joint vector for '{concept_a} AND {concept_b}' at layer {layer}...")
    
    # Generate joint contrastive pairs
    joint_pairs = generate_joint_contrastive_pairs(concept_a, concept_b, n_pairs)
    
    # Extract vector using standard CAA method
    joint_vector = extract_steering_vector(
        model=model,
        tokenizer=tokenizer,
        contrastive_pairs=joint_pairs,
        layer_idx=layer
    )
    
    print(f"✓ Extracted joint vector (shape: {joint_vector.shape})")
    
    return joint_vector


def test_joint_vs_composition(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
    vector_a: torch.Tensor,
    vector_b: torch.Tensor,
    joint_vector: torch.Tensor,
    layer: int,
    coefficient_a: float,
    coefficient_b: float,
    test_prompts: List[str],
    n_prompts: int = 10,
    n_generations: int = 3,
    threshold: float = 0.5
) -> JointVectorResult:
    """
    Compare joint vector extraction vs composition.
    
    Tests:
    1. Baseline (no steering)
    2. Joint vector (extracted from "A AND B" examples)
    3. Composition (vector_a + vector_b)
    4. A only (for reference)
    5. B only (for reference)
    """
    print(f"\n{'='*60}")
    print(f"Testing: Joint vs Composition for '{concept_a}' + '{concept_b}'")
    print(f"{'='*60}")
    
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    quality_metrics = QualityMetrics()
    
    prompts = test_prompts[:n_prompts]
    
    # Storage
    baseline_texts = []
    joint_texts = []
    composition_texts = []
    a_only_texts = []
    b_only_texts = []
    
    baseline_scores_a = []
    baseline_scores_b = []
    joint_scores_a = []
    joint_scores_b = []
    composition_scores_a = []
    composition_scores_b = []
    a_only_scores_a = []
    a_only_scores_b = []
    b_only_scores_a = []
    b_only_scores_b = []
    
    print("\nGenerating text samples...")
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # 1. Baseline
            baseline = generate_baseline(model, tokenizer, prompt)
            baseline_texts.append(baseline)
            scores = evaluator.evaluate(baseline, [concept_a, concept_b])
            baseline_scores_a.append(scores[concept_a])
            baseline_scores_b.append(scores[concept_b])
            
            # 2. Joint vector
            config_joint = SteeringConfig(
                vector=joint_vector,
                layer=layer,
                coefficient=coefficient_a  # Use same magnitude as A
            )
            text_joint = generate_with_steering(model, tokenizer, prompt, config_joint)
            joint_texts.append(text_joint)
            scores = evaluator.evaluate(text_joint, [concept_a, concept_b])
            joint_scores_a.append(scores[concept_a])
            joint_scores_b.append(scores[concept_b])
            
            # 3. Composition (A + B)
            config_composition = [
                SteeringConfig(vector=vector_a, layer=layer, coefficient=coefficient_a),
                SteeringConfig(vector=vector_b, layer=layer, coefficient=coefficient_b)
            ]
            text_composition = generate_with_steering(model, tokenizer, prompt, config_composition)
            composition_texts.append(text_composition)
            scores = evaluator.evaluate(text_composition, [concept_a, concept_b])
            composition_scores_a.append(scores[concept_a])
            composition_scores_b.append(scores[concept_b])
            
            # 4. A only (reference)
            config_a = SteeringConfig(vector=vector_a, layer=layer, coefficient=coefficient_a)
            text_a = generate_with_steering(model, tokenizer, prompt, config_a)
            a_only_texts.append(text_a)
            scores = evaluator.evaluate(text_a, [concept_a, concept_b])
            a_only_scores_a.append(scores[concept_a])
            a_only_scores_b.append(scores[concept_b])
            
            # 5. B only (reference)
            config_b = SteeringConfig(vector=vector_b, layer=layer, coefficient=coefficient_b)
            text_b = generate_with_steering(model, tokenizer, prompt, config_b)
            b_only_texts.append(text_b)
            scores = evaluator.evaluate(text_b, [concept_a, concept_b])
            b_only_scores_a.append(scores[concept_a])
            b_only_scores_b.append(scores[concept_b])
    
    # Compute metrics
    baseline_mean_a = np.mean(baseline_scores_a)
    baseline_mean_b = np.mean(baseline_scores_b)
    
    joint_mean_a = np.mean(joint_scores_a)
    joint_mean_b = np.mean(joint_scores_b)
    
    composition_mean_a = np.mean(composition_scores_a)
    composition_mean_b = np.mean(composition_scores_b)
    
    a_only_mean_a = np.mean(a_only_scores_a)
    b_only_mean_b = np.mean(b_only_scores_b)
    
    # Success rates (both attributes present)
    joint_both_present = sum(
        1 for sa, sb in zip(joint_scores_a, joint_scores_b)
        if sa > threshold and sb > threshold
    ) / len(joint_scores_a)
    
    composition_both_present = sum(
        1 for sa, sb in zip(composition_scores_a, composition_scores_b)
        if sa > threshold and sb > threshold
    ) / len(composition_scores_a)
    
    # Quality
    baseline_ppls = quality_metrics.perplexity_calc.compute_batch(baseline_texts)
    joint_ppls = quality_metrics.perplexity_calc.compute_batch(joint_texts)
    composition_ppls = quality_metrics.perplexity_calc.compute_batch(composition_texts)
    
    # Geometry - compare vectors
    joint_vs_composition = compute_cosine_similarity(joint_vector, vector_a + vector_b)
    joint_vs_a = compute_cosine_similarity(joint_vector, vector_a)
    joint_vs_b = compute_cosine_similarity(joint_vector, vector_b)
    
    result = JointVectorResult(
        concept_a=concept_a,
        concept_b=concept_b,
        layer=layer,
        baseline_score_a=float(baseline_mean_a),
        joint_score_a=float(joint_mean_a),
        composition_score_a=float(composition_mean_a),
        a_only_score_a=float(a_only_mean_a),
        baseline_score_b=float(baseline_mean_b),
        joint_score_b=float(joint_mean_b),
        composition_score_b=float(composition_mean_b),
        b_only_score_b=float(b_only_mean_b),
        joint_improvement_a=float(joint_mean_a - baseline_mean_a),
        joint_improvement_b=float(joint_mean_b - baseline_mean_b),
        composition_improvement_a=float(composition_mean_a - baseline_mean_a),
        composition_improvement_b=float(composition_mean_b - baseline_mean_b),
        joint_both_present_rate=float(joint_both_present),
        composition_both_present_rate=float(composition_both_present),
        baseline_perplexity=float(np.mean(baseline_ppls)),
        joint_perplexity=float(np.mean(joint_ppls)),
        composition_perplexity=float(np.mean(composition_ppls)),
        joint_vs_composition_similarity=float(joint_vs_composition),
        joint_vs_a_similarity=float(joint_vs_a),
        joint_vs_b_similarity=float(joint_vs_b)
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    print(f"\n{concept_a} Scores:")
    print(f"  Baseline:    {result.baseline_score_a:.3f}")
    print(f"  Joint:       {result.joint_score_a:.3f} (Δ={result.joint_improvement_a:+.3f})")
    print(f"  Composition: {result.composition_score_a:.3f} (Δ={result.composition_improvement_a:+.3f})")
    print(f"  A only:      {result.a_only_score_a:.3f}")
    
    print(f"\n{concept_b} Scores:")
    print(f"  Baseline:    {result.baseline_score_b:.3f}")
    print(f"  Joint:       {result.joint_score_b:.3f} (Δ={result.joint_improvement_b:+.3f})")
    print(f"  Composition: {result.composition_score_b:.3f} (Δ={result.composition_improvement_b:+.3f})")
    print(f"  B only:      {result.b_only_score_b:.3f}")
    
    print(f"\nBoth Attributes Present:")
    print(f"  Joint:       {result.joint_both_present_rate:.1%}")
    print(f"  Composition: {result.composition_both_present_rate:.1%}")
    
    print(f"\nVector Geometry:")
    print(f"  Joint ≈ (A+B)?  {result.joint_vs_composition_similarity:.3f}")
    print(f"  Joint ≈ A?      {result.joint_vs_a_similarity:.3f}")
    print(f"  Joint ≈ B?      {result.joint_vs_b_similarity:.3f}")
    
    print(f"\nPerplexity:")
    print(f"  Baseline:    {result.baseline_perplexity:.1f}")
    print(f"  Joint:       {result.joint_perplexity:.1f} (Δ={result.joint_perplexity - result.baseline_perplexity:+.1f})")
    print(f"  Composition: {result.composition_perplexity:.1f} (Δ={result.composition_perplexity - result.baseline_perplexity:+.1f})")
    
    # Determine winner
    if result.joint_both_present_rate > result.composition_both_present_rate:
        print(f"\n✓ Joint extraction is MORE effective ({result.joint_both_present_rate:.1%} vs {result.composition_both_present_rate:.1%})")
    elif result.composition_both_present_rate > result.joint_both_present_rate:
        print(f"\n✓ Composition is MORE effective ({result.composition_both_present_rate:.1%} vs {result.joint_both_present_rate:.1%})")
    else:
        print(f"\n≈ Joint and Composition are EQUALLY effective ({result.joint_both_present_rate:.1%})")
    
    return result


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
    """Run joint vector vs composition experiment."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from extraction.extract_vectors import load_cached_vectors
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--output_dir", default="outputs/joint_vector")
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--concept_pairs", nargs="+", action="append",
                       help="Pairs of concepts to test, e.g., --concept_pairs formal positive --concept_pairs technical confident")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with fewer samples")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "joint_vectors").mkdir(exist_ok=True)
    
    week1_dir = Path(args.week1_dir)
    if not week1_dir.exists():
        raise FileNotFoundError(f"Week 1 results not found at {week1_dir}. Run week 1 first!")
    
    print("="*60)
    print("JOINT VECTOR vs COMPOSITION EXPERIMENT")
    print("="*60)
    
    # Load optimal layers and coefficients
    with open(week1_dir / "optimal_layers.json") as f:
        optimal_layers = json.load(f)
    
    with open(week1_dir / "optimal_coefficients.json") as f:
        optimal_coefficients = json.load(f)
    
    # Define concept pairs to test
    if args.concept_pairs:
        concept_pairs = [(pair[0], pair[1]) for pair in args.concept_pairs]
    else:
        # Default pairs
        concept_pairs = [
            ("formal", "positive"),
            ("technical", "confident"),
            ("verbose", "uncertain")
        ]
    
    print(f"\nModel: {args.model}")
    print(f"Concept pairs: {concept_pairs}")
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
    
    # Get test prompts
    test_prompts = get_test_prompts()
    if args.quick:
        n_prompts = 5
        n_gen = 2
        n_pairs = 50
    else:
        n_prompts = 10
        n_gen = 3
        n_pairs = 100
    
    # Run experiments
    all_results = []
    
    for concept_a, concept_b in concept_pairs:
        print(f"\n{'='*60}")
        print(f"TESTING: {concept_a.upper()} + {concept_b.upper()}")
        print(f"{'='*60}")
        
        # Check if concepts exist in Week 1
        if concept_a not in optimal_layers or concept_b not in optimal_layers:
            print(f"⚠ Warning: Concepts not found in Week 1, skipping...")
            continue
        
        # Use the same layer for both (or could use different layers)
        layer = optimal_layers[concept_a]
        print(f"Using layer: {layer}")
        
        # Load individual vectors from Week 1
        print("\nLoading individual vectors from Week 1...")
        steering_vectors_by_layer = load_cached_vectors(
            week1_dir / "vectors",
            [concept_a, concept_b],
            [layer]
        )
        
        vector_a = steering_vectors_by_layer[concept_a][layer]
        vector_b = steering_vectors_by_layer[concept_b][layer]
        
        coefficient_a = optimal_coefficients.get(concept_a, cfg.model.default_coefficient)
        coefficient_b = optimal_coefficients.get(concept_b, cfg.model.default_coefficient)
        
        print(f"✓ Loaded vectors for {concept_a} (coef={coefficient_a}) and {concept_b} (coef={coefficient_b})")
        
        # Extract joint vector
        joint_vector = extract_joint_vector(
            model, tokenizer,
            concept_a, concept_b,
            layer, n_pairs
        )
        
        # Save joint vector
        joint_vector_path = output_dir / "joint_vectors" / f"{concept_a}_{concept_b}_layer{layer}.pt"
        torch.save(joint_vector, joint_vector_path)
        print(f"✓ Saved joint vector to {joint_vector_path}")
        
        # Test joint vs composition
        result = test_joint_vs_composition(
            model, tokenizer,
            concept_a, concept_b,
            vector_a, vector_b, joint_vector,
            layer, coefficient_a, coefficient_b,
            test_prompts, n_prompts, n_gen
        )
        
        all_results.append(result)
    
    # Save results
    output_data = {
        "experiment": "joint_vector_vs_composition",
        "concept_pairs": [(r.concept_a, r.concept_b) for r in all_results],
        "results": [
            {
                "concept_a": r.concept_a,
                "concept_b": r.concept_b,
                "layer": r.layer,
                "baseline_score_a": r.baseline_score_a,
                "joint_score_a": r.joint_score_a,
                "composition_score_a": r.composition_score_a,
                "a_only_score_a": r.a_only_score_a,
                "baseline_score_b": r.baseline_score_b,
                "joint_score_b": r.joint_score_b,
                "composition_score_b": r.composition_score_b,
                "b_only_score_b": r.b_only_score_b,
                "joint_improvement_a": r.joint_improvement_a,
                "joint_improvement_b": r.joint_improvement_b,
                "composition_improvement_a": r.composition_improvement_a,
                "composition_improvement_b": r.composition_improvement_b,
                "joint_both_present_rate": r.joint_both_present_rate,
                "composition_both_present_rate": r.composition_both_present_rate,
                "baseline_perplexity": r.baseline_perplexity,
                "joint_perplexity": r.joint_perplexity,
                "composition_perplexity": r.composition_perplexity,
                "joint_vs_composition_similarity": r.joint_vs_composition_similarity,
                "joint_vs_a_similarity": r.joint_vs_a_similarity,
                "joint_vs_b_similarity": r.joint_vs_b_similarity
            }
            for r in all_results
        ]
    }
    
    output_data = convert_to_native(output_data)
    
    try:
        with open(output_dir / "joint_vs_composition_results.json", "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {output_dir / 'joint_vs_composition_results.json'}")
    except Exception as e:
        print(f"⚠ Warning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    print("\nSummary:")
    for result in all_results:
        print(f"\n{result.concept_a} + {result.concept_b}:")
        print(f"  Joint both present:       {result.joint_both_present_rate:.1%}")
        print(f"  Composition both present: {result.composition_both_present_rate:.1%}")
        print(f"  Vector similarity:        {result.joint_vs_composition_similarity:.3f}")
        
        if result.joint_both_present_rate > result.composition_both_present_rate + 0.05:
            print(f"  → Joint extraction WINS")
        elif result.composition_both_present_rate > result.joint_both_present_rate + 0.05:
            print(f"  → Composition WINS")
        else:
            print(f"  → TIE (roughly equivalent)")


if __name__ == "__main__":
    main()