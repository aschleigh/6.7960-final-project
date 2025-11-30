"""
Layer selection utilities for finding optimal steering layers.
"""

import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from steering.apply_steering import SteeringConfig, generate_with_steering
from evaluation.classifiers import AttributeClassifier


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
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        steering_vectors: Dict mapping concept -> layer -> steering vector
        concept: The concept to optimize for
        layers: List of layer indices to test
        test_prompts: Prompts to test on
        n_prompts: Number of prompts to use
        n_generations: Generations per prompt
    
    Returns:
        The optimal layer index
    """
    print(f"  Finding optimal layer for '{concept}'...")
    
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
                scores.append(score)
        
        layer_scores[layer] = np.mean(scores)
        print(f"    Layer {layer}: {layer_scores[layer]:.3f}")
    
    optimal_layer = max(layer_scores, key=layer_scores.get)
    print(f"    → Optimal: {optimal_layer}")
    
    return optimal_layer


def find_optimal_layers_batch(
    model,
    tokenizer,
    steering_vectors_by_layer: Dict[str, Dict[int, torch.Tensor]],
    concepts: List[str],
    layers: List[int],
    test_prompts: List[str],
    n_prompts: int = 5,
    n_generations: int = 2
) -> Dict[str, int]:
    """
    Find optimal layers for multiple concepts.
    
    Returns:
        Dict mapping concept -> optimal layer
    """
    print("\nFinding optimal layers for each concept...")
    optimal_layers = {}
    
    for concept in tqdm(concepts, desc="Optimizing layers"):
        if concept in steering_vectors_by_layer:
            optimal_layers[concept] = find_optimal_layer(
                model,
                tokenizer,
                steering_vectors_by_layer,
                concept,
                layers,
                test_prompts,
                n_prompts,
                n_generations
            )
        else:
            print(f"  Warning: {concept} not found in steering vectors")
    
    return optimal_layers