"""
Propagation Analysis: The "Self-Correction" Hypothesis.

Hypothesis:
    Composition fails because the model's MLP layers "scrub" out-of-distribution 
    combinations. We trace the signal strength of Vector A across all future 
    layers to see if Vector B causes it to decay prematurely.
    
    Metric: Signal Retention % = (Proj_A_with_B / Proj_A_alone)
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from config import cfg
from data.prompts import get_test_prompts
from evaluation.classifiers import MultiAttributeEvaluator
from extraction.extract_vectors import load_cached_vectors
from extraction.hooks import ActivationHooks

def measure_signal_propagation(
    model, 
    tokenizer, 
    prompt: str,
    target_vec: torch.Tensor,
    interferer_vec: torch.Tensor,
    steering_layer: int,
    trace_layers: List[int],
    coef: float = 2.0
):
    """
    Measure how well the 'target_vec' survives in the residual stream 
    across future layers, with and without the 'interferer_vec'.
    """
    hooks = ActivationHooks(model)
    # Ensure inputs are on the model's first device (usually cuda:0)
    device = model.device 
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Pre-calculate normalized target for projection
    # Keep it on CPU initially to avoid getting stuck on the wrong GPU
    target_norm_cpu = (target_vec / target_vec.norm()).cpu()

    # -------------------------------------------------------
    # PASS 1: Baseline (Steer A only)
    # -------------------------------------------------------
    hooks.clear_hooks()
    
    # 1. Apply Steering at Injection Layer
    # Note: The hook handles moving the vector to the correct device internally
    hooks.register_steering_hook(steering_layer, target_vec, coef)
    
    # 2. Register Extraction at ALL Trace Layers
    for layer in trace_layers:
        hooks.register_extraction_hook(layer)
        
    # 3. Run Forward Pass
    with torch.no_grad():
        model(**inputs)
        
    # 4. Measure "Signal Strength" of A at each layer
    baseline_signal = {}
    
    for layer in trace_layers:
        # Get activation of last token
        # This tensor is on whichever GPU holds this layer
        act = hooks.cache[f"layer_{layer}_residual"][0, -1, :] 
        
        # FIX: Move our reference vector to the SAME device as the activation
        target_norm_device = target_norm_cpu.to(act.device, dtype=act.dtype)
        
        # Project onto target direction
        signal = torch.dot(act, target_norm_device).item()
        baseline_signal[layer] = signal

    # -------------------------------------------------------
    # PASS 2: Interference (Steer A + B)
    # -------------------------------------------------------
    hooks.clear_hooks()
    
    # 1. Apply Combined Steering
    combined_vec = target_vec + interferer_vec
    hooks.register_steering_hook(steering_layer, combined_vec, coef)
    
    # 2. Register Extraction
    for layer in trace_layers:
        hooks.register_extraction_hook(layer)
        
    # 3. Run Forward Pass
    with torch.no_grad():
        model(**inputs)
        
    # 4. Measure Signal Strength of A (Does B make A vanish?)
    interfered_signal = {}
    for layer in trace_layers:
        act = hooks.cache[f"layer_{layer}_residual"][0, -1, :]
        
        # FIX: Move reference vector again
        target_norm_device = target_norm_cpu.to(act.device, dtype=act.dtype)
        
        signal = torch.dot(act, target_norm_device).item()
        interfered_signal[layer] = signal
        
    hooks.clear_hooks()
    return baseline_signal, interfered_signal


def run_propagation_experiment(
    model, tokenizer, steering_vectors, concepts, 
    start_layer, n_layers, prompts
) -> List[Dict]:
    
    results = []
    # Trace every 2nd layer after steering
    trace_layers = list(range(start_layer + 1, n_layers, 2))
    
    # Create pairs (A vs B)
    pairs = []
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i != j:
                pairs.append((c1, c2))
                
    print(f"Tracing signal decay for {len(pairs)} pairs across {len(trace_layers)} layers...")
    
    for target, interferer in tqdm(pairs):
        vec_a = steering_vectors[target]
        vec_b = steering_vectors[interferer]
        
        # We average results over a few prompts to reduce noise
        prompt_results = []
        
        for prompt in prompts:
            base_sig, int_sig = measure_signal_propagation(
                model, tokenizer, prompt, vec_a, vec_b, 
                steering_layer=start_layer, trace_layers=trace_layers
            )
            
            # Calculate "Retention Ratio" at the FINAL layer
            final_layer = trace_layers[-1]
            retention = int_sig[final_layer] / (base_sig[final_layer] + 1e-6)
            prompt_results.append(retention)
            
        avg_retention = np.mean(prompt_results)
        
        results.append({
            "target": target,
            "interferer": interferer,
            "signal_retention": avg_retention, # < 1.0 means B suppressed A
            "suppression_score": 1.0 - avg_retention
        })
        
    return results

def analyze_results(results: List[Dict], save_dir: Path):
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("PROPAGATION DYNAMICS SUMMARY")
    print("="*60)
    print(f"Avg Signal Retention: {df['signal_retention'].mean():.2%}")
    print("Top Suppressors (Vectors that kill other vectors):")
    print(df.groupby("interferer")["suppression_score"].mean().sort_values(ascending=False).head(5))
    print("-" * 60)
    
    # Histogram of Retention
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["signal_retention"], bins=20, kde=True)
    plt.axvline(1.0, color='red', linestyle='--', label="Perfect Retention")
    plt.title("Distribution of Signal Retention (Layer 15 -> Output)")
    plt.xlabel("Signal Retention Ratio (With Interference / Baseline)")
    plt.legend()
    plt.savefig(save_dir / "signal_retention_dist.png")
    print(f"Saved plot to {save_dir}")

def convert_to_native(obj):
    if isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_native(i) for i in obj]
    if isinstance(obj, (np.int64, np.int32)): return int(obj)
    if isinstance(obj, (np.float64, np.float32)): return float(obj)
    return obj

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--output_dir", default="outputs/propagation_analysis")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading resources...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
    # Determine model depth
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        n_layers = len(model.model.layers)
    
    with open(Path(args.week1_dir) / "geometry_analysis.json") as f:
        concepts = json.load(f)["concepts"]
        
    vectors_by_layer = load_cached_vectors(Path(args.week1_dir) / "vectors", concepts, cfg.model.steering_layers)
    vectors = {c: v[cfg.model.default_layer].to(model.device) for c, v in vectors_by_layer.items()}
    
    prompts = get_test_prompts()[:2] if args.quick else get_test_prompts()[:5]
    
    results = run_propagation_experiment(
        model, tokenizer, vectors, concepts, 
        start_layer=cfg.model.default_layer, 
        n_layers=n_layers, 
        prompts=prompts
    )
    
    analyze_results(results, output_dir)
    
    with open(output_dir / "propagation_results.json", "w") as f:
        json.dump(convert_to_native(results), f, indent=2)

if __name__ == "__main__":
    main()