"""
Layer selection utilities for finding optimal steering layers.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from steering.apply_steering import SteeringConfig, generate_with_steering
from evaluation.classifiers import AttributeClassifier

def find_layer_performance(
    model,
    tokenizer,
    steering_vectors: Dict[int, torch.Tensor],
    concept: str,
    layers: List[int],
    test_prompts: List[str],
    n_generations: int = 2,
    threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Analyze steering performance across layers to find optimal and valid ranges.
    
    Args:
        model: The LLM.
        tokenizer: The tokenizer.
        steering_vectors: Dict mapping layer_idx -> vector tensor.
        concept: The concept name (e.g., "formal") for the classifier.
        layers: List of layer indices to test.
        test_prompts: List of prompts to evaluate on.
        n_generations: Number of generations per prompt per layer.
        threshold: Score threshold to consider a layer "valid".

    Returns:
        Dict containing:
        - 'scores': Dict[int, float] mapping layer to mean score.
        - 'optimal_layer': int (layer with highest score).
        - 'valid_layers': List[int] (all layers passing threshold).
    """
    print(f"  Profiling layers for '{concept}'...")
    
    classifier = AttributeClassifier(concept)
    layer_scores = {}
    
    # Iterate through each layer to test
    for layer in layers:
        # Skip if we don't have a vector for this layer
        if layer not in steering_vectors:
            continue
            
        sv = steering_vectors[layer]
        scores = []
        
        # Test on small batch of prompts
        for prompt in test_prompts:
            config = SteeringConfig(vector=sv, layer=layer, coefficient=1.0)
            for _ in range(n_generations):
                text = generate_with_steering(model, tokenizer, prompt, config)
                score = classifier.score(text)
                scores.append(score)
        
        avg_score = float(np.mean(scores))
        layer_scores[layer] = avg_score
    
    # Handle edge case where no layers were tested or all failed
    if not layer_scores:
        print(f"    Warning: No scores computed for {concept}")
        return {"scores": {}, "optimal_layer": layers[0], "valid_layers": []}

    # Identify optimal and valid layers
    optimal_layer = max(layer_scores, key=layer_scores.get)
    valid_layers = [l for l, s in layer_scores.items() if s >= threshold]
    
    # Logging
    best_score = layer_scores[optimal_layer]
    print(f"    → Optimal: {optimal_layer} (Score: {best_score:.2f})")
    if valid_layers:
        print(f"    → Valid Range: {min(valid_layers)}-{max(valid_layers)}")
    else:
        print("    → No valid layers found (all below threshold)")
    
    return {
        "scores": layer_scores,
        "optimal_layer": optimal_layer,
        "valid_layers": valid_layers
    }


def find_optimal_layers_batch(
    model,
    tokenizer,
    steering_vectors_by_layer: Dict[str, Dict[int, torch.Tensor]],
    concepts: List[str],
    layers: List[int],
    test_prompts: List[str]
) -> Dict[str, int]:
    """
    Legacy wrapper: Finds the single best layer for each concept.
    Useful for Week 1 compatibility.
    """
    optimal_map = {}
    for concept in concepts:
        if concept in steering_vectors_by_layer:
            perf = find_layer_performance(
                model, 
                tokenizer, 
                steering_vectors_by_layer[concept], 
                concept, 
                layers, 
                test_prompts
            )
            optimal_map[concept] = perf["optimal_layer"]
    return optimal_map