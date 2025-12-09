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
import matplotlib.pyplot as plt

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
    
    # Define characteristic phrases for each concept
    concept_characteristics = {
        "formal": {
            "positive": ["respectfully", "cordially", "professionally", "officially", "formally"],
            "style": "structured and proper language"
        },
        "casual": {
            "positive": ["hey", "yeah", "kinda", "gonna", "wanna"],
            "style": "relaxed and informal language"
        },
        "positive": {
            "positive": ["excellent", "wonderful", "delighted", "pleased", "great"],
            "style": "optimistic and upbeat tone"
        },
        "negative": {
            "positive": ["terrible", "disappointed", "unfortunately", "problem", "failed"],
            "style": "pessimistic and critical tone"
        },
        "verbose": {
            "positive": ["furthermore", "consequently", "nevertheless", "additionally", "particularly"],
            "style": "lengthy and detailed expression"
        },
        "concise": {
            "positive": ["brief", "short", "quick", "simple", "direct"],
            "style": "short and to-the-point expression"
        },
        "confident": {
            "positive": ["certainly", "definitely", "absolutely", "clearly", "undoubtedly"],
            "style": "assured and assertive tone"
        },
        "uncertain": {
            "positive": ["perhaps", "maybe", "possibly", "might", "could"],
            "style": "hesitant and questioning tone"
        },
        "technical": {
            "positive": ["algorithm", "implementation", "architecture", "framework", "protocol"],
            "style": "specialized and precise terminology"
        },
        "simple": {
            "positive": ["easy", "basic", "straightforward", "plain", "clear"],
            "style": "accessible and uncomplicated language"
        }
    }
    
    # Generate templates based on concept combination
    key = tuple(sorted([concept_a, concept_b]))
    
    # Helper to create positive examples (both A and B)
    def create_positive(a_words, b_words, length_a="medium", length_b="medium"):
        templates = []
        
        # Determine sentence structure based on concepts
        if "verbose" in [concept_a, concept_b]:
            base_length = "long"
        elif "concise" in [concept_a, concept_b]:
            base_length = "short"
        else:
            base_length = "medium"
            
        # Generate sentences combining both concepts
        if base_length == "short":
            templates = [
                f"{a_words[0].capitalize()}, this is {b_words[0]}.",
                f"{a_words[1].capitalize()}. Very {b_words[1]}.",
                f"{a_words[2].capitalize()} and {b_words[2]}.",
                f"This is {a_words[3]} and {b_words[3]}.",
                f"{a_words[4].capitalize()}. Quite {b_words[4]}.",
                f"{a_words[0].capitalize()}, it's {b_words[0]}.",
                f"Very {a_words[1]} and {b_words[1]}.",
                f"{a_words[2].capitalize()}. Truly {b_words[2]}.",
                f"This seems {a_words[3]} and {b_words[3]}.",
                f"{a_words[4].capitalize()}, quite {b_words[4]}.",
            ]
        elif base_length == "long":
            templates = [
                f"I would like to {a_words[0]} inform you that this is truly {b_words[0]}, and furthermore, it demonstrates exceptional qualities.",
                f"It is with considerable {a_words[1]} that I must say this appears remarkably {b_words[1]} in every conceivable aspect.",
                f"One might {a_words[2]} observe that this situation is demonstrably {b_words[2]}, taking into account all relevant factors.",
                f"Upon careful consideration, I find it necessary to {a_words[3]} note that this is profoundly {b_words[3]} in nature.",
                f"It seems appropriate to {a_words[4]} mention that this exhibits characteristics that are undeniably {b_words[4]} overall.",
                f"I feel compelled to {a_words[0]} state that this represents something genuinely {b_words[0]} and worthy of attention.",
                f"After thorough evaluation, I must {a_words[1]} conclude that this is remarkably {b_words[1]} in all respects.",
                f"It would be remiss not to {a_words[2]} acknowledge that this displays qualities that are notably {b_words[2]} throughout.",
                f"In my considered opinion, I should {a_words[3]} emphasize that this is authentically {b_words[3]} and significant.",
                f"Taking everything into account, I must {a_words[4]} observe that this proves to be distinctly {b_words[4]} indeed.",
            ]
        else:  # medium
            templates = [
                f"I would like to {a_words[0]} inform you that this is {b_words[0]}.",
                f"It is with {a_words[1]} that I must say this is {b_words[1]}.",
                f"I {a_words[2]} believe that this situation is {b_words[2]}.",
                f"It is important to {a_words[3]} note that this is {b_words[3]}.",
                f"I want to {a_words[4]} mention that this is {b_words[4]}.",
                f"I am pleased to {a_words[0]} state that this is {b_words[0]}.",
                f"With {a_words[1]}, I can say this is quite {b_words[1]}.",
                f"I {a_words[2]} feel that this is genuinely {b_words[2]}.",
                f"It seems {a_words[3]} that this proves to be {b_words[3]}.",
                f"I should {a_words[4]} add that this is truly {b_words[4]}.",
            ]
        
        return templates
    
    # Helper to create negative examples (neither A nor B, or just one)
    def create_negative(a_opposite, b_opposite):
        # Create sentences with opposite characteristics
        templates = [
            f"This is {a_opposite[0]} and {b_opposite[0]}.",
            f"{a_opposite[1].capitalize()}, this seems {b_opposite[1]}.",
            f"It appears {a_opposite[2]} and {b_opposite[2]}.",
            f"{a_opposite[3].capitalize()}. Rather {b_opposite[3]}.",
            f"This feels {a_opposite[4]} and {b_opposite[4]}.",
            f"Seems {a_opposite[0]} and quite {b_opposite[0]}.",
            f"{a_opposite[1].capitalize()}. Very {b_opposite[1]}.",
            f"This is {a_opposite[2]} and {b_opposite[2]} overall.",
            f"Rather {a_opposite[3]} and {b_opposite[3]} indeed.",
            f"{a_opposite[4].capitalize()}, quite {b_opposite[4]}.",
        ]
        return templates
    
    # Get characteristics for concepts
    char_a = concept_characteristics[concept_a]["positive"]
    char_b = concept_characteristics[concept_b]["positive"]
    
    # Determine opposites
    opposites = {
        "formal": "casual", "casual": "formal",
        "positive": "negative", "negative": "positive",
        "verbose": "concise", "concise": "verbose",
        "confident": "uncertain", "uncertain": "confident",
        "technical": "simple", "simple": "technical"
    }
    
    opposite_a = opposites.get(concept_a, concept_a)
    opposite_b = opposites.get(concept_b, concept_b)
    
    char_a_opposite = concept_characteristics[opposite_a]["positive"]
    char_b_opposite = concept_characteristics[opposite_b]["positive"]
    
    # Generate positive and negative templates
    positive_templates = create_positive(char_a, char_b)
    negative_templates = create_negative(char_a_opposite, char_b_opposite)
    
    # Ensure we have enough templates
    while len(positive_templates) < 10:
        positive_templates.extend(positive_templates)
    while len(negative_templates) < 10:
        negative_templates.extend(negative_templates)
    
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


def plot_joint_vector_results(all_results: List[JointVectorResult], output_dir: Path):
    """Create visualizations comparing joint vs composition approaches."""
    
    n_pairs = len(all_results)
    
    # 1. Individual plots for each concept pair - showing both concepts
    fig, axes = plt.subplots(1, n_pairs, figsize=(7*n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    
    colors = {
        'baseline': '#95a5a6',
        'joint': '#3498db', 
        'composition': '#e74c3c',
        'a_only': '#9b59b6',
        'b_only': '#f39c12'
    }
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        
        # Prepare data for both concepts
        methods = ['Baseline', 'Joint', 'Composition', f'{result.concept_a.capitalize()}\nOnly', f'{result.concept_b.capitalize()}\nOnly']
        
        # Scores for concept A
        scores_a = [
            result.baseline_score_a,
            result.joint_score_a,
            result.composition_score_a,
            result.a_only_score_a,
            result.baseline_score_a  # Use baseline for B-only since it doesn't target A
        ]
        
        # Scores for concept B
        scores_b = [
            result.baseline_score_b,
            result.joint_score_b,
            result.composition_score_b,
            result.baseline_score_b,  # Use baseline for A-only since it doesn't target B
            result.b_only_score_b
        ]
        
        x = np.arange(len(methods))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, scores_a, width, label=result.concept_a.capitalize(), 
                       color=colors['a_only'], alpha=0.8)
        bars2 = ax.bar(x + width/2, scores_b, width, label=result.concept_b.capitalize(), 
                       color=colors['b_only'], alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{result.concept_a.capitalize()} + {result.concept_b.capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "joint_individual_scores.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'joint_individual_scores.png'}")
    plt.close()
    
    # 2. "Both Present" success rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pair_labels = [f"{r.concept_a.capitalize()}\n+\n{r.concept_b.capitalize()}" for r in all_results]
    x = np.arange(len(all_results))
    width = 0.35
    
    baseline_rates = [r.baseline_score_a * r.baseline_score_b for r in all_results]  # Approximate baseline
    joint_rates = [r.joint_both_present_rate * 100 for r in all_results]
    composition_rates = [r.composition_both_present_rate * 100 for r in all_results]
    
    bars1 = ax.bar(x - width, joint_rates, width, label='Joint', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, composition_rates, width, label='Composition', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, np.array(baseline_rates) * 100, width, label='Baseline (approx)', 
                   color='#95a5a6', alpha=0.6)
    
    ax.set_xlabel('Concept Pair', fontsize=12, fontweight='bold')
    ax.set_ylabel('Both Attributes Present (%)', fontsize=12, fontweight='bold')
    ax.set_title('Joint vs Composition: Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "joint_both_present_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'joint_both_present_comparison.png'}")
    plt.close()
    
    # 3. Vector similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    similarities = []
    labels = []
    
    for result in all_results:
        pair_label = f"{result.concept_a.capitalize()}+{result.concept_b.capitalize()}"
        similarities.append([
            result.joint_vs_composition_similarity,
            result.joint_vs_a_similarity,
            result.joint_vs_b_similarity
        ])
        labels.append(pair_label)
    
    similarities = np.array(similarities)
    
    im = ax.imshow(similarities, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(all_results)))
    ax.set_xticklabels(['Joint ≈ (A+B)', 'Joint ≈ A', 'Joint ≈ B'], fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    
    # Add text annotations
    for i in range(len(all_results)):
        for j in range(3):
            text = ax.text(j, i, f'{similarities[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Vector Similarity Analysis', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_dir / "joint_vector_similarity.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'joint_vector_similarity.png'}")
    plt.close()
    
    # 4. Improvement comparison (how much each method improves over baseline)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7*n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        
        methods = ['Joint', 'Composition']
        
        # Average improvement across both concepts
        joint_avg_improvement = (result.joint_improvement_a + result.joint_improvement_b) / 2
        composition_avg_improvement = (result.composition_improvement_a + result.composition_improvement_b) / 2
        
        improvements = [joint_avg_improvement, composition_avg_improvement]
        
        colors_list = ['#3498db', '#e74c3c']
        bars = ax.bar(methods, improvements, color=colors_list, alpha=0.8, width=0.6)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Average Improvement over Baseline', fontsize=12, fontweight='bold')
        ax.set_title(f'{result.concept_a.capitalize()} + {result.concept_b.capitalize()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.3f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "joint_improvement_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'joint_improvement_comparison.png'}")
    plt.close()

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
    
    # Define full list of concepts
    all_concepts = [
        "formal", "casual",
        "positive", "negative",
        "verbose", "concise",
        "confident", "uncertain",
        "technical", "simple"
    ]

    # Define oppositional pairs to exclude
    opposite_pairs = {
        ("formal", "casual"),
        ("casual", "formal"),
        ("verbose", "concise"),
        ("concise", "verbose"),
        ("positive", "negative"),
        ("negative", "positive"),
        ("confident", "uncertain"),
        ("uncertain", "confident"),
        ("technical", "simple"),
        ("simple", "technical")
    }

    # If user manually specifies pairs, use those
    if args.concept_pairs:
        concept_pairs = [(pair[0], pair[1]) for pair in args.concept_pairs]
    else:
        # Auto-generate all valid non-opposite pairs
        concept_pairs = []
        for i in range(len(all_concepts)):
            for j in range(i + 1, len(all_concepts)):
                a, b = all_concepts[i], all_concepts[j]

                # Skip opposite pairs
                if (a, b) in opposite_pairs or (b, a) in opposite_pairs:
                    continue

                concept_pairs.append((a, b))

    print(f"\nRunning joint comparison for {len(concept_pairs)} concept pairs")
    print(f"Model: {args.model}")
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
        n_prompts = 7
        n_gen = 3
        n_pairs = 60
    
    # Run experiments
    all_results = []
    successful_pairs = []
    failed_pairs = []
    
    for idx, (concept_b, concept_a) in enumerate(concept_pairs, 1):
        print(f"\n{'='*60}")
        print(f"TESTING {idx}/{len(concept_pairs)}: {concept_a.upper()} + {concept_b.upper()}")
        print(f"{'='*60}")
        
        try:
            # Check if concepts exist in Week 1
            if concept_a not in optimal_layers or concept_b not in optimal_layers:
                print(f"⚠ Warning: Concepts not found in Week 1, skipping...")
                failed_pairs.append((concept_a, concept_b, "Not in Week 1 results"))
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
            successful_pairs.append((concept_a, concept_b))
            
            # Save intermediate results after each successful test
            if len(all_results) % 5 == 0:  # Save every 5 successful tests
                print(f"\n💾 Saving intermediate results ({len(all_results)} completed)...")
                save_intermediate_results(all_results, output_dir)
            
        except Exception as e:
            print(f"\n❌ ERROR testing {concept_a} + {concept_b}: {e}")
            import traceback
            traceback.print_exc()
            failed_pairs.append((concept_a, concept_b, str(e)))
            continue
    
    # Save final results
    print(f"\n{'='*60}")
    print("SAVING FINAL RESULTS")
    print(f"{'='*60}")
    
    output_data = {
        "experiment": "joint_vector_vs_composition",
        "total_pairs_tested": len(concept_pairs),
        "successful_pairs": len(successful_pairs),
        "failed_pairs": len(failed_pairs),
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
        ],
        "failed_pairs_info": [
            {
                "concept_a": a,
                "concept_b": b,
                "reason": reason
            }
            for a, b, reason in failed_pairs
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
    
    # Generate plots
    # if all_results:
    #     print("\n" + "="*60)
    #     print("GENERATING PLOTS")
    #     print("="*60)
    #     try:
    #         plot_joint_vector_results(all_results, output_dir)
    #         print("✓ Finished generating plots.")
    #     except Exception as e:
    #         print(f"⚠ Failed to generate plots: {e}")
    #         import traceback
    #         traceback.print_exc()
    # else:
    #     print("\n⚠ No successful results to plot")
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    print(f"\nTotal pairs attempted: {len(concept_pairs)}")
    print(f"Successful: {len(successful_pairs)}")
    print(f"Failed: {len(failed_pairs)}")
    
    if failed_pairs:
        print("\nFailed pairs:")
        for a, b, reason in failed_pairs:
            print(f"  - {a} + {b}: {reason}")
    
    if all_results:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        joint_wins = 0
        composition_wins = 0
        ties = 0
        
        for result in all_results:
            if result.joint_both_present_rate > result.composition_both_present_rate + 0.05:
                joint_wins += 1
                winner = "Joint WINS"
            elif result.composition_both_present_rate > result.joint_both_present_rate + 0.05:
                composition_wins += 1
                winner = "Composition WINS"
            else:
                ties += 1
                winner = "TIE"
            
            print(f"\n{result.concept_a} + {result.concept_b}:")
            print(f"  Joint both present:       {result.joint_both_present_rate:.1%}")
            print(f"  Composition both present: {result.composition_both_present_rate:.1%}")
            print(f"  Vector similarity:        {result.joint_vs_composition_similarity:.3f}")
            print(f"  → {winner}")
        
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Joint wins:        {joint_wins}/{len(all_results)} ({joint_wins/len(all_results)*100:.1f}%)")
        print(f"Composition wins:  {composition_wins}/{len(all_results)} ({composition_wins/len(all_results)*100:.1f}%)")
        print(f"Ties:              {ties}/{len(all_results)} ({ties/len(all_results)*100:.1f}%)")


def save_intermediate_results(all_results, output_dir):
    """Save intermediate results to avoid losing progress."""
    output_data = {
        "experiment": "joint_vector_vs_composition_INTERMEDIATE",
        "num_results": len(all_results),
        "concept_pairs": [(r.concept_a, r.concept_b) for r in all_results],
        "results": [
            {
                "concept_a": r.concept_a,
                "concept_b": r.concept_b,
                "layer": r.layer,
                "joint_both_present_rate": r.joint_both_present_rate,
                "composition_both_present_rate": r.composition_both_present_rate,
                "joint_vs_composition_similarity": r.joint_vs_composition_similarity,
            }
            for r in all_results
        ]
    }
    
    output_data = convert_to_native(output_data)
    
    try:
        with open(output_dir / "joint_vs_composition_intermediate.json", "w") as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        print(f"⚠ Warning: Failed to save intermediate results: {e}")


if __name__ == "__main__":
    main()