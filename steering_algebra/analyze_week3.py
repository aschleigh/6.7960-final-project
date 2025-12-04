"""
Analyze Week 3 results and generate insights.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


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
    if comp_results and "best_same_layer" in comp_results:
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
    """
    Analyze failure mode patterns across all concept pairs.
    
    Note: failure_results is now a dict of pairs, not a single result.
    """
    all_analyses = {}
    aggregate_stats = {
        "total_pairs": 0,
        "avg_success_rate": 0.0,
        "pairs_with_dominance": 0,
        "pairs_with_high_incoherence": 0
    }
    
    success_rates = []
    
    for pair_key, pair_data in failure_results.items():
        # Skip error entries
        if "error" in pair_data or "categories" not in pair_data:
            continue
            
        categories = pair_data["categories"]
        total = pair_data["total_samples"]
        
        # Compute insights for this pair
        dominant_failure = max(
            categories.items(),
            key=lambda x: x[1]["count"] if x[0] != "success" else 0
        )
        
        composition_viable = categories["success"]["percentage"] > 30
        one_dominates = (categories["a_dominates"]["percentage"] + 
                        categories["b_dominates"]["percentage"]) > 50
        
        pair_analysis = {
            "concept_a": pair_data["concept_a"],
            "concept_b": pair_data["concept_b"],
            "success_rate": categories["success"]["percentage"],
            "dominant_failure_mode": dominant_failure[0],
            "dominant_failure_percentage": dominant_failure[1]["percentage"],
            "composition_viable": composition_viable,
            "one_concept_typically_dominates": one_dominates,
            "incoherence_rate": categories["incoherent"]["percentage"]
        }
        
        all_analyses[pair_key] = pair_analysis
        
        # Update aggregate stats
        aggregate_stats["total_pairs"] += 1
        success_rates.append(categories["success"]["percentage"])
        if one_dominates:
            aggregate_stats["pairs_with_dominance"] += 1
        if categories["incoherent"]["percentage"] > 20:
            aggregate_stats["pairs_with_high_incoherence"] += 1
    
    # Compute aggregate statistics
    if success_rates:
        aggregate_stats["avg_success_rate"] = float(np.mean(success_rates))
        aggregate_stats["median_success_rate"] = float(np.median(success_rates))
        aggregate_stats["min_success_rate"] = float(np.min(success_rates))
        aggregate_stats["max_success_rate"] = float(np.max(success_rates))
    
    return {
        "per_pair": all_analyses,
        "aggregate": aggregate_stats
    }


def generate_insights(analysis: Dict) -> List[str]:
    """Generate human-readable insights."""
    insights = []
    
    # Coefficient insights
    if "coefficient_sweep" in analysis:
        coef_analysis = analysis["coefficient_sweep"]
        
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
    if "layer_effects" in analysis:
        layer_analysis = analysis["layer_effects"]
        
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
    
    # Failure pattern insights (updated for multiple pairs)
    if "failure_patterns" in analysis:
        failure = analysis["failure_patterns"]
        
        if "aggregate" in failure:
            agg = failure["aggregate"]
            
            insights.append(f"📊 Analyzed {agg['total_pairs']} concept pairs")
            insights.append(f"📊 Average composition success rate: {agg['avg_success_rate']:.1f}%")
            
            if agg["pairs_with_dominance"] > agg["total_pairs"] * 0.5:
                insights.append("❌ One concept typically dominates in most pairs (>50%)")
            
            if agg["pairs_with_high_incoherence"] > agg["total_pairs"] * 0.3:
                insights.append(f"❌ High incoherence in {agg['pairs_with_high_incoherence']} pairs")
            
            if agg["avg_success_rate"] > 30:
                insights.append("✅ Composition shows promise overall (>30% avg success)")
            else:
                insights.append("❌ Composition largely unsuccessful (<30% avg success)")
    
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
    
    # Failure patterns (updated for multiple pairs)
    if "failure_patterns" in analysis:
        print("\n3. FAILURE PATTERN ANALYSIS")
        print("-" * 40)
        
        if "aggregate" in analysis["failure_patterns"]:
            agg = analysis["failure_patterns"]["aggregate"]
            print(f"Total pairs analyzed: {agg['total_pairs']}")
            print(f"Average success rate: {agg['avg_success_rate']:.1f}%")
            print(f"Median success rate: {agg.get('median_success_rate', 0):.1f}%")
            print(f"Range: {agg.get('min_success_rate', 0):.1f}% - {agg.get('max_success_rate', 0):.1f}%")
            print(f"Pairs with dominance: {agg['pairs_with_dominance']}/{agg['total_pairs']}")
            print(f"Pairs with high incoherence: {agg['pairs_with_high_incoherence']}/{agg['total_pairs']}")
            
            # Show best and worst pairs
            if "per_pair" in analysis["failure_patterns"]:
                per_pair = analysis["failure_patterns"]["per_pair"]
                sorted_pairs = sorted(per_pair.items(), 
                                    key=lambda x: x[1]["success_rate"], 
                                    reverse=True)
                
                if sorted_pairs:
                    print("\n  Best performing pairs:")
                    for pair_key, pair_data in sorted_pairs[:3]:
                        print(f"    {pair_data['concept_a']} + {pair_data['concept_b']}: "
                              f"{pair_data['success_rate']:.1f}% success")
                    
                    print("\n  Worst performing pairs:")
                    for pair_key, pair_data in sorted_pairs[-3:]:
                        print(f"    {pair_data['concept_a']} + {pair_data['concept_b']}: "
                              f"{pair_data['success_rate']:.1f}% success")
    
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
        print("Run week 3 experiments first: python week3_analysis.py")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Perform analysis
    analysis = {}
    
    if "coefficient_sweep" in results:
        analysis["coefficient_sweep"] = analyze_coefficient_sweep(results["coefficient_sweep"])
    
    if "layer_ablation_single" in results:
        comp_results = results.get("layer_ablation_composition", {})
        analysis["layer_effects"] = analyze_layer_effects(
            results["layer_ablation_single"],
            comp_results
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