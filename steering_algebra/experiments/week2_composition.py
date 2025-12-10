"""
Week 2 Experiment: Steering Vector Composition & Arithmetic.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
from dataclasses import dataclass
import ast

from config import cfg
from data.prompts import get_test_prompts
from steering.apply_steering import (
    SteeringConfig,
    generate_with_steering,
    generate_baseline
)
from evaluation.classifiers import MultiAttributeEvaluator
from evaluation.metrics import QualityMetrics
from evaluation.geometry import compute_cosine_similarity
from extraction.extract_vectors import load_cached_vectors

@dataclass
class CompositionResult:
    """Results from a single composition experiment."""
    concept_a: str
    concept_b: str
    layer: int
    
    # Scores
    baseline_scores: Dict[str, float]
    a_only_scores: Dict[str, float]
    b_only_scores: Dict[str, float]
    composition_scores: Dict[str, float]
    
    # Metrics
    interference_a: float
    interference_b: float
    cosine_similarity: float
    composition_success: bool 

def get_golden_pairs() -> List[Tuple[str, str]]:
    """
    Manually defined pairs to test specific geometric hypotheses.
    UPDATED to match new config (Slang, no Verbose).
    """
    return [
        # HYPOTHESIS 1: Orthogonal (Content + Style) -> Should Compose Well
        ("fantasy", "formal"),
        ("science", "slang"),  # Changed casual -> slang
        
        # HYPOTHESIS 2: Compatible Styles -> Should Compose Okay
        ("positive", "formal"),
        ("confident", "technical"),
        
        # HYPOTHESIS 3: Conflicting/Interfering -> Should Fail/Interfere
        ("formal", "slang"),   # Direct opposites (Antipodal)
        ("positive", "negative"), # Direct opposites
    ]

def test_additive_composition(
    model,
    tokenizer,
    steering_vectors: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    layer: int,
    prompts: List[str],
    coefficient: float = 1.0,
    n_generations: int = 3
) -> CompositionResult:
    """
    Test if v_A + v_B produces text with both attributes at a fixed layer.
    """
    evaluator = MultiAttributeEvaluator([concept_a, concept_b])
    
    vec_a = steering_vectors[concept_a]
    vec_b = steering_vectors[concept_b]
    
    # Arithmetic: Sum vectors
    vec_combined = vec_a + vec_b
    
    print(f"\nTesting Composition: {concept_a} + {concept_b} (Layer {layer})")
    sim = compute_cosine_similarity(vec_a, vec_b)
    print(f"  Cosine Similarity: {sim:.3f}")
    
    results_a = {"base": [], "a": [], "b": [], "comb": []}
    results_b = {"base": [], "a": [], "b": [], "comb": []}
    
    for prompt in tqdm(prompts, desc="Generating"):
        for _ in range(n_generations):
            # Baseline
            base = generate_baseline(model, tokenizer, prompt)
            s_base = evaluator.evaluate(base, [concept_a, concept_b])
            
            # A Only
            conf_a = SteeringConfig(vector=vec_a, layer=layer, coefficient=coefficient)
            text_a = generate_with_steering(model, tokenizer, prompt, conf_a)
            s_a = evaluator.evaluate(text_a, [concept_a, concept_b])
            
            # B Only
            conf_b = SteeringConfig(vector=vec_b, layer=layer, coefficient=coefficient)
            text_b = generate_with_steering(model, tokenizer, prompt, conf_b)
            s_b = evaluator.evaluate(text_b, [concept_a, concept_b])
            
            # Combined (A + B)
            conf_comb = SteeringConfig(vector=vec_combined, layer=layer, coefficient=coefficient)
            text_comb = generate_with_steering(model, tokenizer, prompt, conf_comb)
            s_comb = evaluator.evaluate(text_comb, [concept_a, concept_b])
            
            # Store scores
            for k, res in [("base", s_base), ("a", s_a), ("b", s_b), ("comb", s_comb)]:
                results_a[k].append(res[concept_a])
                results_b[k].append(res[concept_b])

    def mean(l): return float(np.mean(l))
    
    scores_final = {
        "baseline": {c: mean(results_a["base"] if c==concept_a else results_b["base"]) for c in [concept_a, concept_b]},
        "a_only":   {c: mean(results_a["a"] if c==concept_a else results_b["a"]) for c in [concept_a, concept_b]},
        "b_only":   {c: mean(results_a["b"] if c==concept_a else results_b["b"]) for c in [concept_a, concept_b]},
        "combined": {c: mean(results_a["comb"] if c==concept_a else results_b["comb"]) for c in [concept_a, concept_b]},
    }
    
    # Interference Metrics
    score_a_only = scores_final["a_only"][concept_a]
    score_a_comb = scores_final["combined"][concept_a]
    # If A_only is 0.8 and Combined is 0.4, interference is 50% (0.5)
    interference_a = (score_a_only - score_a_comb) / score_a_only if score_a_only > 0.1 else 0.0
    
    score_b_only = scores_final["b_only"][concept_b]
    score_b_comb = scores_final["combined"][concept_b]
    interference_b = (score_b_only - score_b_comb) / score_b_only if score_b_only > 0.1 else 0.0
    
    # Success: Both boosted above baseline
    success = (score_a_comb > scores_final["baseline"][concept_a] + 0.05) and \
              (score_b_comb > scores_final["baseline"][concept_b] + 0.05)
              
    print(f"  Results:")
    print(f"    {concept_a}: Base={scores_final['baseline'][concept_a]:.2f} -> Comb={score_a_comb:.2f} (Intf: {interference_a:.2f})")
    print(f"    {concept_b}: Base={scores_final['baseline'][concept_b]:.2f} -> Comb={score_b_comb:.2f} (Intf: {interference_b:.2f})")
    print(f"    Success: {success}")

    return CompositionResult(
        concept_a=concept_a,
        concept_b=concept_b,
        layer=layer,
        baseline_scores=scores_final["baseline"],
        a_only_scores=scores_final["a_only"],
        b_only_scores=scores_final["b_only"],
        composition_scores=scores_final["combined"],
        interference_a=interference_a,
        interference_b=interference_b,
        cosine_similarity=float(sim),
        composition_success=success
    )

def main():
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=cfg.model.name)
    parser.add_argument("--week1_dir", default="outputs/week1")
    parser.add_argument("--output_dir", default="outputs/week2")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    w1_dir = Path(args.week1_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Common Layer
    summary_path = w1_dir / "validation_summary.json"
    if not summary_path.exists():
        common_layer = cfg.model.default_layer
    else:
        with open(summary_path) as f:
            summary = json.load(f)
            common_layer = summary.get("common_layer", cfg.model.default_layer)
    
    print("="*60)
    print(f"WEEK 2: Vector Arithmetic & Composition")
    print(f"Target Layer: {common_layer}")
    print("="*60)

    # 2. Load Model
    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16 if cfg.model.torch_dtype=="float16" else torch.bfloat16, 
        device_map="auto"
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    print("Loading vectors...")
    vectors_map = load_cached_vectors(w1_dir / "vectors", cfg.concepts, [common_layer])
    
    vectors = {}
    for c, layers_map in vectors_map.items():
        if common_layer in layers_map:
            vectors[c] = layers_map[common_layer]
    
    print(f"Loaded {len(vectors)} vectors at layer {common_layer}.")
    
    # 3. Define Pairs to Test
    golden_pairs = get_golden_pairs()
    valid_pairs = [
        (a, b) for a, b in golden_pairs 
        if a in vectors and b in vectors
    ]
    
    if not valid_pairs:
        print("Error: No valid pairs found. Check concepts list in config.py vs golden_pairs.")
        print(f"Available vectors: {list(vectors.keys())}")
        return

    # 4. Run Experiment
    prompts = get_test_prompts()
    if args.quick: prompts = prompts[:3]
    
    results = []
    
    for c_a, c_b in valid_pairs:
        try:
            # Use default coefficient from config
            res = test_additive_composition(
                model, tokenizer, vectors, c_a, c_b, 
                layer=common_layer, prompts=prompts, 
                n_generations=2,
                coefficient=cfg.model.default_coefficient
            )
            results.append(res)
        except Exception as e:
            print(f"Error testing {c_a}+{c_b}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Results
    def result_to_dict(r: CompositionResult):
        return {
            "pair": f"{r.concept_a}+{r.concept_b}",
            "concept_a": r.concept_a,
            "concept_b": r.concept_b,
            "layer": r.layer,
            "cosine_similarity": r.cosine_similarity,
            "metrics": {
                "interference_a": r.interference_a,
                "interference_b": r.interference_b,
                "success": r.composition_success
            },
            "scores": {
                "baseline": r.baseline_scores,
                "composition": r.composition_scores
            }
        }
        
    final_output = [result_to_dict(r) for r in results]
    
    out_path = out_dir / "composition_results.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "="*60)
    print(f"Week 2 Complete. Results saved to {out_path}")
    print("="*60)

if __name__ == "__main__":
    main()