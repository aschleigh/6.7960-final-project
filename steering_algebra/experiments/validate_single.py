"""
Week 1 Experiment: Extract and Validate Single Steering Vectors.
Simplified Version: Targets ONLY Layer 12.
"""

import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
import json
from tqdm import tqdm
import os

from config import cfg
from data.prompts import get_test_prompts, get_adversarial_prompts
from data.contrastive_pairs import get_all_pairs
from steering.apply_steering import SteeringConfig, generate_with_steering
from evaluation.classifiers import AttributeClassifier
from extraction.extract_vectors import extract_all_vectors, load_cached_vectors

def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, torch.Tensor): return obj.cpu().tolist()
    return str(obj)

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--concepts", nargs="+", default=cfg.concepts)
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--show_text", action="store_true", help="Print ALL generated text (neutral + adversarial)")
    parser.add_argument("--num_prompts", type=int, default=10, help="Number of prompts to test per concept")
    args = parser.parse_args()
    
    target_layer = cfg.model.default_layer
    
    print("="*60)
    print(f"WEEK 1: Validation (Fixed Layer {target_layer})")
    print("="*60)
    
    # 0. Load Model
    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16 if cfg.model.torch_dtype == "float16" else torch.bfloat16, 
        device_map="auto"
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 1. Extraction
    vectors_dir = cfg.output_dir / "week1/vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_extraction:
        print(f"\n[1] Extracting Vectors (Layer {target_layer} only)...")
        pairs_dict = get_all_pairs(args.concepts, n_pairs=cfg.extraction.n_pairs)
        
        extract_all_vectors(
            model=model, 
            tokenizer=tokenizer, 
            concepts=args.concepts,
            contrastive_pairs=pairs_dict,
            layers=[target_layer],
            cache_dir=vectors_dir
        )
    else:
        print(f"\n[1] Skipping extraction. Loading from {vectors_dir}...")
    
    vectors = load_cached_vectors(vectors_dir, args.concepts, [target_layer])
    if not vectors:
        print("Error: No vectors found!")
        return

    # 2. Validation
    print(f"\n[2] Validating Vectors at Layer {target_layer}...")
    
    adversarial_map = get_adversarial_prompts()
    neutral_prompts = get_test_prompts()
    
    results = {}
    
    for concept in args.concepts:
        if concept not in vectors or target_layer not in vectors[concept]:
            print(f"  Skipping {concept}: No vector for layer {target_layer}")
            continue
            
        vec = vectors[concept][target_layer]
        classifier = AttributeClassifier(concept)
        
        # Build prompt list
        adv_prompts = adversarial_map.get(concept, [])
        concept_prompts = (adv_prompts + neutral_prompts)[:args.num_prompts]
        
        print(f"\n{'='*40}")
        print(f"Testing Concept: {concept.upper()}")
        print(f"{'='*40}")
        
        base_scores = []
        steered_scores = []
        
        for i, p in enumerate(concept_prompts):
            # Check if this is an adversarial prompt
            is_adversarial = p in adv_prompts

            # Baseline Generation
            base_text = generate_with_steering(model, tokenizer, p, [])
            base_score = classifier.score(base_text)
            base_scores.append(base_score)
            
            # Steered Generation
            coef = cfg.model.concept_coefficients.get(concept, cfg.model.default_coefficient)
            config = SteeringConfig(vector=vec, layer=target_layer, coefficient=coef)
            steered_text = generate_with_steering(model, tokenizer, p, config)
            steered_score = classifier.score(steered_text)
            
            steered_scores.append(steered_score)
            
            # --- PRINT LOGIC ---
            # Print if: 
            # 1. It is Adversarial (User Request)
            # 2. args.show_text is True (Print Everything)
            # 3. It is one of the first 2 prompts (Sanity check)
            if is_adversarial or args.show_text or i < 2:
                prefix = "[ADVERSARIAL]" if is_adversarial else "[NEUTRAL]"
                print(f"\n{prefix} Prompt: {p}")
                print(f"  Base    ({base_score:.2f}): {base_text}")
                print(f"  Steered ({steered_score:.2f}): {steered_text}")
                print("-" * 30)
        
        avg_score = np.mean(steered_scores) - np.mean(base_scores)
        status = "✓ STRONG" if avg_score > 0.6 else ("~ MODERATE" if avg_score > 0.4 else "⚠ WEAK")
        print(f"Result: {concept} avg score gain = {avg_score:.2f} [{status}]")
        
        results[concept] = {
            "score": avg_score,
            "valid": avg_score > 0.5
        }

    # 3. Save
    summary = {
        "common_layer": target_layer,
        "results": results
    }
    
    # Save optimal layers dummy file
    optimal_dummy = {c: target_layer for c in args.concepts}
    with open(cfg.output_dir / "week1/optimal_layers.json", "w") as f:
        json.dump(optimal_dummy, f, indent=2)

    with open(cfg.output_dir / "week1/validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=convert_to_serializable)
        
    print("\n" + "="*60)
    print("Validation Complete")
    print("="*60)

if __name__ == "__main__":
    main()