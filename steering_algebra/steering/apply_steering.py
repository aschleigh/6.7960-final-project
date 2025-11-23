"""
Apply steering vectors during text generation.
"""

import torch
from torch import Tensor
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from utils import format_prompt_for_model

from extraction.hooks import ActivationHooks, MultiVectorSteering


@dataclass
class SteeringConfig:
    """Configuration for a single steering vector."""
    vector: Tensor
    layer: int
    coefficient: float = 1.0
    

def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_config: Union[SteeringConfig, List[SteeringConfig]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda"
) -> str:
    """
    Generate text with steering vector(s) applied.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        steering_config: Single SteeringConfig or list for multi-vector steering
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample (vs greedy)
        device: Device to use
        
    Returns:
        Generated text (without the prompt)
    """
    # Handle single config
    if isinstance(steering_config, SteeringConfig):
        steering_config = [steering_config]

    # Format prompt for model
    formatted_prompt = format_prompt_for_model(prompt, tokenizer, model.config._name_or_path)
    
    # Tokenize prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Set up multi-vector steering
    steerer = MultiVectorSteering(model)
    configs = [
        {
            "layer": sc.layer,
            "vector": sc.vector,
            "coefficient": sc.coefficient
        }
        for sc in steering_config
    ]
    
    # Generate with steering
    with steerer.steering_context(configs):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and remove prompt
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    
    return response.strip()


def generate_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda"
) -> str:
    """Generate text without any steering (baseline)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    
    return response.strip()


def generate_batch_with_steering(
    model,
    tokenizer,
    prompts: List[str],
    steering_config: Union[SteeringConfig, List[SteeringConfig]],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda"
) -> List[str]:
    """
    Generate text for multiple prompts with steering.
    Note: For proper batching, all prompts should be similar length.
    """
    results = []
    for prompt in prompts:
        result = generate_with_steering(
            model, tokenizer, prompt, steering_config,
            max_new_tokens, temperature, top_p, do_sample, device
        )
        results.append(result)
    return results


def coefficient_sweep(
    model,
    tokenizer,
    prompt: str,
    steering_vector: Tensor,
    layer: int,
    coefficients: List[float],
    n_generations: int = 3,
    **gen_kwargs
) -> Dict[float, List[str]]:
    """
    Generate text across a range of steering coefficients.
    
    Useful for finding the optimal coefficient and understanding
    the effect of steering strength.
    
    Returns:
        Dict mapping coefficient -> list of generated texts
    """
    results = {}
    
    for coef in coefficients:
        config = SteeringConfig(vector=steering_vector, layer=layer, coefficient=coef)
        generations = []
        
        for _ in range(n_generations):
            gen = generate_with_steering(model, tokenizer, prompt, config, **gen_kwargs)
            generations.append(gen)
        
        results[coef] = generations
    
    return results


def layer_sweep(
    model,
    tokenizer,
    prompt: str,
    steering_vectors: Dict[int, Tensor],  # layer -> vector
    coefficient: float = 1.0,
    n_generations: int = 3,
    **gen_kwargs
) -> Dict[int, List[str]]:
    """
    Generate text with steering at different layers.
    
    Useful for finding the optimal layer for steering.
    
    Args:
        steering_vectors: Dict mapping layer index to steering vector
        
    Returns:
        Dict mapping layer -> list of generated texts
    """
    results = {}
    
    for layer, vector in steering_vectors.items():
        config = SteeringConfig(vector=vector, layer=layer, coefficient=coefficient)
        generations = []
        
        for _ in range(n_generations):
            gen = generate_with_steering(model, tokenizer, prompt, config, **gen_kwargs)
            generations.append(gen)
        
        results[layer] = generations
    
    return results