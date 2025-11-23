"""
Extract steering vectors using Contrastive Activation Addition (CAA).

Method:
1. Run model on positive prompts, cache activations
2. Run model on negative prompts, cache activations  
3. Steering vector = mean(positive_activations) - mean(negative_activations)
"""

import torch
from torch import Tensor
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from pathlib import Path
import json

from extraction.hooks import ActivationHooks
from data.contrastive_pairs import ContrastivePair


def extract_steering_vector(
    model,
    tokenizer,
    contrastive_pairs: List[ContrastivePair],
    layer_idx: int,
    token_position: int = -1,
    batch_size: int = 8,
    normalize: bool = True,
    device: str = "cuda"
) -> Tensor:
    """
    Extract a steering vector using CAA.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        contrastive_pairs: List of (positive, negative) prompt pairs
        layer_idx: Which layer to extract from
        token_position: Which token position (-1 = last token)
        batch_size: Batch size for processing
        normalize: Whether to L2-normalize the resulting vector
        device: Device to use
        
    Returns:
        Steering vector of shape (hidden_dim,)
    """
    hooks = ActivationHooks(model)
    
    positive_activations = []
    negative_activations = []
    
    # Process in batches
    for i in tqdm(range(0, len(contrastive_pairs), batch_size), desc=f"Extracting layer {layer_idx}"):
        batch_pairs = contrastive_pairs[i:i + batch_size]
        
        # Extract positive activations
        pos_prompts = [p[0] for p in batch_pairs]
        pos_acts = _get_activations(
            model, tokenizer, hooks, pos_prompts, 
            layer_idx, token_position, device
        )
        positive_activations.append(pos_acts)
        
        # Extract negative activations
        neg_prompts = [p[1] for p in batch_pairs]
        neg_acts = _get_activations(
            model, tokenizer, hooks, neg_prompts,
            layer_idx, token_position, device
        )
        negative_activations.append(neg_acts)
    
    # Concatenate all activations
    positive_activations = torch.cat(positive_activations, dim=0)  # (n_pairs, hidden_dim)
    negative_activations = torch.cat(negative_activations, dim=0)
    
    # Compute steering vector as difference of means
    steering_vector = positive_activations.mean(dim=0) - negative_activations.mean(dim=0)
    
    if normalize:
        steering_vector = steering_vector / steering_vector.norm()
    
    return steering_vector


def _get_activations(
    model,
    tokenizer,
    hooks: ActivationHooks,
    prompts: List[str],
    layer_idx: int,
    token_position: int,
    device: str
) -> Tensor:
    """Helper to extract activations for a batch of prompts."""
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Extract activations
    with hooks.extraction_context([layer_idx]) as cache:
        with torch.no_grad():
            model(**inputs)
        
        acts = cache[f"layer_{layer_idx}_residual"]  # (batch, seq_len, hidden_dim)
    
    # Get activations at specified token position
    if token_position == -1:
        # Get last non-padding token for each sequence
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        batch_acts = []
        for b in range(acts.shape[0]):
            batch_acts.append(acts[b, seq_lens[b], :])
        return torch.stack(batch_acts, dim=0)
    else:
        return acts[:, token_position, :]


def extract_all_vectors(
    model,
    tokenizer,
    concepts: List[str],
    contrastive_pairs: Dict[str, List[ContrastivePair]],
    layers: List[int],
    token_position: int = -1,
    batch_size: int = 8,
    normalize: bool = True,
    device: str = "cuda",
    cache_dir: Optional[Path] = None
) -> Dict[str, Dict[int, Tensor]]:
    """
    Extract steering vectors for all concepts and layers.
    
    Returns:
        Dict mapping concept -> layer -> steering_vector
    """
    all_vectors = {}
    
    for concept in concepts:
        print(f"\n{'='*50}")
        print(f"Extracting vectors for: {concept}")
        print(f"{'='*50}")
        
        pairs = contrastive_pairs[concept]
        all_vectors[concept] = {}
        
        for layer in layers:
            sv = extract_steering_vector(
                model, tokenizer, pairs, layer,
                token_position, batch_size, normalize, device
            )
            all_vectors[concept][layer] = sv
            
            # Cache to disk if requested
            if cache_dir is not None:
                save_path = cache_dir / f"{concept}_layer{layer}.pt"
                torch.save(sv, save_path)
    
    return all_vectors


def load_cached_vectors(
    cache_dir: Path,
    concepts: List[str],
    layers: List[int]
) -> Dict[str, Dict[int, Tensor]]:
    """Load previously cached steering vectors."""
    all_vectors = {}
    
    for concept in concepts:
        all_vectors[concept] = {}
        for layer in layers:
            path = cache_dir / f"{concept}_layer{layer}.pt"
            if path.exists():
                all_vectors[concept][layer] = torch.load(path)
            else:
                print(f"Warning: Missing cached vector for {concept} layer {layer}")
    
    return all_vectors


def analyze_extraction_quality(
    model,
    tokenizer,
    contrastive_pairs: List[ContrastivePair],
    steering_vector: Tensor,
    layer_idx: int,
    device: str = "cuda"
) -> Dict:
    """
    Analyze the quality of an extracted steering vector.
    
    Returns statistics about how well the vector separates positive/negative examples.
    """
    hooks = ActivationHooks(model)
    
    pos_prompts = [p[0] for p in contrastive_pairs]
    neg_prompts = [p[1] for p in contrastive_pairs]
    
    # Get activations
    pos_acts = _get_activations(model, tokenizer, hooks, pos_prompts, layer_idx, -1, device)
    neg_acts = _get_activations(model, tokenizer, hooks, neg_prompts, layer_idx, -1, device)
    
    # Project onto steering vector
    sv_norm = steering_vector / steering_vector.norm()
    pos_proj = (pos_acts @ sv_norm).float().cpu().numpy()
    neg_proj = (neg_acts @ sv_norm).float().cpu().numpy()
    
    # Compute separation statistics
    pos_mean = pos_proj.mean()
    neg_mean = neg_proj.mean()
    pos_std = pos_proj.std()
    neg_std = neg_proj.std()
    
    # Cohen's d (effect size)
    pooled_std = ((pos_std**2 + neg_std**2) / 2) ** 0.5
    cohens_d = (pos_mean - neg_mean) / pooled_std
    
    # Classification accuracy (how well does sign of projection classify?)
    threshold = (pos_mean + neg_mean) / 2
    pos_correct = (pos_proj > threshold).sum()
    neg_correct = (neg_proj <= threshold).sum()
    accuracy = (pos_correct + neg_correct) / (len(pos_proj) + len(neg_proj))
    
    return {
        "pos_mean": float(pos_mean),
        "neg_mean": float(neg_mean),
        "pos_std": float(pos_std),
        "neg_std": float(neg_std),
        "cohens_d": float(cohens_d),
        "separation_accuracy": float(accuracy),
        "mean_difference": float(pos_mean - neg_mean),
    }