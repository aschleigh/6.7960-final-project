"""
Analyze Week 3 results and generate insights.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict


def analyze_coefficient_sweep(results: Dict) -> Dict:
    """Analyze coefficient sweep results."""
    analysis = {}
    
    for concept, data in results.items():
        coefficients = [r["coefficient"] for r in data["coefficients"]]
        scores = [r["mean_score"] for r in data["coefficients"]]
        perplexities = [r["mean_perplexity"] for r in data["coefficients"]]
        
        # Find score plateau
        score_diff = np.diff(scores)
        plateau_idx = np.where(np.abs(score_diff) < 0.05)[0]
        
        # Find perplexity explosion
        explosion_idx = [i for i, p in enumerate(perplexities) if p > 100]
        
        analysis[concept] = {
            "optimal_coefficient": data.get("optimal_coefficient"),
            "optimal_score": data.get("optimal_score"),
            "score_range": (float(min(scores)), float(max(scores))),
            "perplexity_explosion_at": [coefficients[i] for i in explosion_idx],
            "plateau_starts_at": coefficients[plateau_idx[0]] if len(plateau_idx) > 0 else None
        }
    
    return analysis


def analyze_layer_effects(single_results: Dict, comp_results: Dict) -> Dict:
    """Analyze layer effects."""
    analysis = {}
    
    # Single concept analysis
    for concept, data in single_results.items():
        layers = [r["layer"] for r in data["layers"]]
        scores = [r["mean_score"] for r in data["layers"]]
        
        # Find layer with peak performance
        best_idx = np.argmax(scores)
        
        # Check if middle layers are best
        is_middle_best = layers[best_idx] in layers[len(layers)//3:2*len(layers)//3]
        
        analysis[concept] = {
            "best_layer": data["best_layer"],
            "best_score": data["best_score"],
            "is_middle_layer_best": is_middle_best,
            "score_variance_across_layers": float(np.var(scores))
        }
    
    # Composition analysis
    if "best_same_layer" in comp_results:
        same_layer_results = comp_results["same_layer"]
        layers = [r["layer"] for r in same_layer_results]
        success_rates = [r["joint_success_rate"] for r in same_layer_results]
        
        best_idx = np.argmax(success_rates)
        
        analysis["composition"] = {
            "best_layer": layers[best_idx],
            "best_success_rate": float(success_rates[best_idx]),
            "layer_sensitivity": float(np.std(success_rates))
        }
    
    return analysis


def analyze_failure_patterns(failure_results: Dict) -> Dict:
    """Analyze failure mode patterns."""
    categories = failure_results["categories"]
    
    total = failure_results["total_samples"]
    
    # Compute insights
    dominant_failure = max(
        categories.items(),
        key=lambda x: x[1]["count"] if x[0] != "success" else 0
    )
    
    composition_viable = categories["success"]["percentage"] > 30
    one_dominates = (categories["a_dominates"]["percentage"] + 
                    categories["b_dominates"]["percentage"]) > 50
    
    analysis = {
        "success_rate": categories["success"]["percentage"],
        "dominant_failure_mode": dominant_failure[0],
        "dominant_failure_percentage": dominant_failure[1]["percentage"],
        "composition_viable": composition_viable,
        "one_concept_typically_dominates": one_dominates,
        "incoherence_rate": categories["incoherent"]["percentage"]
    }
    
    return analysis


def generate_insights(analysis_results: Dict) -> List[str]:
    """Generate human-readable insights."""
    insights = []
    
    # Coefficient insights
    if "coefficient_sweep" in analysis_results:
        coef_analysis = analysis_results["coefficient_sweep"]
        
        optimal_coefs = [data["optimal_coefficient"] for data in coef_analysis.values() 
                        if data.get("optimal_coefficient") is not None]
        
        if optimal_coefs:
            avg_optimal = np.mean(optimal_coefs)
            insights.append(f"💡 Optimal steering coefficient is typically around {avg_optimal:.2f}")
        
        explosions = [data["perplexity_explosion_at"] for data in coef_analysis.values()
                     if data["perplexity_explosion_at"]]
        if explosions:
            insights.append(f"⚠️  Perplexity explosions commonly occur at high coefficients (>2.0)")
    
    # Layer insights
    if "layer_effects" in analysis_results:
        layer_analysis = analysis_results["layer_effects"]
        
        middle_best_count = sum(1 for data in layer_analysis.values() 
                               if data.get("is_middle_layer_best", False))
        
        if middle_best_count > len(layer_analysis) / 2:
            insights.append("💡 Middle layers (40-60% depth) are most effective for steering")
        
        if "composition" in layer_analysis:
            comp = layer_analysis["composition"]
            if comp["layer_sensitivity"] < 0.1:
                insights.append("💡 Composition success is relatively stable across layers")
            else:
                insights.append("⚠️  Composition is highly sensitive to layer choice")
    
    # Failure pattern insights
    if "failure_patterns" in analysis_results:
        failure = analysis_results["failure_patterns"]
        
        if failure["one_concept_typically_dominates"]:
            insights.append("❌ One concept typically dominates in composition (>50% of failures)")
        
        if failure["incoherence_rate"] > 20:
            insights.append(f"❌ High incoherence rate ({failure['incoherence_rate']:.1f}%) suggests instability")
        
        if failure["composition_viable"]:
            insights.append("✅ Composition shows promise (>30% success rate)")
        else:
            insights.append("❌ Composition largely unsuccessful (<30% success rate)")
    
    return insights


def print_summary(analysis: Dict, insights: List[str]):
    """Print analysis summary."""
    print("\n" + "="*60)
    print("WEEK 3 ANALYSIS SUMMARY")
    print("="*60)
    
    # Coefficient sweep
    if "coefficient_sweep" in analysis:
        print("\n1. COEFFICIENT SWEEP ANALYSIS")
        print("-" * 40)
        for concept, data in analysis["coefficient_sweep"].items():
            print(f"\n{concept.capitalize()}:")
            if data.get("optimal_coefficient"):
                print(f"  Optimal coefficient: {data['optimal_coefficient']:.2f}")
                print(f"  Score at optimal: {data['optimal_score']:.3f}")
            print(f"  Score range: {data['score_range'][0]:.3f} - {data['score_range'][1]:.3f}")
            if data["perplexity_explosion_at"]:
                print(f"  Perplexity explodes at: {data['perplexity_explosion_at']}")
    
    # Layer effects
    if "layer_effects" in analysis:
        print("\n2. LAYER EFFECTS ANALYSIS")
        print("-" * 40)
        for concept, data in analysis["layer_effects"].items():
            if concept != "composition":
                print(f"\n{concept.capitalize()}:")
                print(f"  Best layer: {data['best_layer']}")
                print(f"  Score at best: {data['best_score']:.3f}")
                print(f"  Middle layer best: {data['is_middle_layer_best']}")
        
        if "composition" in analysis["layer_effects"]:
            comp = analysis["layer_effects"]["composition"]
            print(f"\nComposition:")
            print(f"  Best layer: {comp['best_layer']}")
            print(f"  Success at best: {comp['best_success_rate']:.1%}")
            print(f"  Layer sensitivity: {comp['layer_sensitivity']:.3f}")
    
    # Failure patterns
    if "failure_patterns" in analysis:
        print("\n3. FAILURE PATTERN ANALYSIS")
        print("-" * 40)
        failure = analysis["failure_patterns"]
        print(f"Success rate: {failure['success_rate']:.1f}%")
        print(f"Dominant failure: {failure['dominant_failure_mode']} ({failure['dominant_failure_percentage']:.1f}%)")
        print(f"One concept dominates: {failure['one_concept_typically_dominates']}")
        print(f"Incoherence rate: {failure['incoherence_rate']:.1f}%")
        print(f"Composition viable: {failure['composition_viable']}")
    
    # Key insights
    if insights:
        print("\n4. KEY INSIGHTS")
        print("-" * 40)
        for insight in insights:
            print(f"  {insight}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/week3")
    args = parser.parse_args()
    
    results_path = Path(args.results_dir) / "analysis_results.json"
    
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        print("Run week 3 experiments first: python run_week3.py")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Perform analysis
    analysis = {}
    
    if "coefficient_sweep" in results:
        analysis["coefficient_sweep"] = analyze_coefficient_sweep(results["coefficient_sweep"])
    
    if "layer_ablation_single" in results and "layer_ablation_composition" in results:
        analysis["layer_effects"] = analyze_layer_effects(
            results["layer_ablation_single"],
            results["layer_ablation_composition"]
        )
    
    if "failure_analysis" in results:
        analysis["failure_patterns"] = analyze_failure_patterns(results["failure_analysis"])
    
    # Generate insights
    insights = generate_insights(analysis)
    
    # Print summary
    print_summary(analysis, insights)
    
    # Save analysis
    output = {
        "analysis": analysis,
        "insights": insights
    }
    
    output_path = Path(args.results_dir) / "week3_analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nDetailed analysis saved to: {output_path}")


if __name__ == "__main__":
    main()