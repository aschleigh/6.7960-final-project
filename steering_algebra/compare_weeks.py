"""
Compare results across all three weeks.
Generate a unified summary for the blog post.
"""

import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


def load_all_results():
    """Load results from all three weeks."""
    results = {}
    
    # Week 1
    week1_path = Path("outputs/week1/steering_validation.json")
    if week1_path.exists():
        with open(week1_path) as f:
            results["week1"] = json.load(f)
    
    # Week 2
    week2_path = Path("outputs/week2/composition_results.json")
    if week2_path.exists():
        with open(week2_path) as f:
            results["week2"] = json.load(f)
    
    # Week 3
    week3_path = Path("outputs/week3/analysis_results.json")
    if week3_path.exists():
        with open(week3_path) as f:
            results["week3"] = json.load(f)
    
    return results


def compare_single_vs_composition(results: Dict):
    """Compare single-vector steering to composition."""
    
    print("\n" + "="*60)
    print("SINGLE-VECTOR vs COMPOSITION COMPARISON")
    print("="*60)
    
    # Week 1: Single-vector performance
    if "week1" in results:
        week1 = results["week1"]
        single_success = week1["aggregated"]["mean_success_rate"]
        single_improvement = week1["aggregated"]["mean_improvement"]
        
        print("\nSingle-Vector Steering (Week 1):")
        print(f"  Mean success rate: {single_success:.1%}")
        print(f"  Mean improvement: {single_improvement:+.3f}")
    
    # Week 2: Composition performance
    if "week2" in results:
        week2 = results["week2"]
        
        if week2.get("additive_composition"):
            comp_success_rates = [
                r["composition_success_rate"] 
                for r in week2["additive_composition"]
            ]
            comp_mean = np.mean(comp_success_rates)
            
            print("\nComposition Steering (Week 2):")
            print(f"  Mean success rate: {comp_mean:.1%}")
            print(f"  Pairs tested: {len(comp_success_rates)}")
        
        # Arithmetic results
        if week2.get("arithmetic_composition"):
            arith = week2["arithmetic_composition"]
            arith_success = sum(1 for r in arith if r.get("arithmetic_works", False))
            
            print("\nVector Arithmetic:")
            print(f"  Success rate: {arith_success}/{len(arith)} ({arith_success/len(arith):.1%})")
            print(f"  Mean difference: {np.mean([r['difference'] for r in arith]):.3f}")
    
    # The gap
    if "week1" in results and "week2" in results:
        if week2.get("additive_composition"):
            gap = single_success - comp_mean
            print(f"\n⚠️  Performance gap: {gap:.1%}")
            print("     Single-vector steering significantly outperforms composition")


def summarize_optimal_settings(results: Dict):
    """Summarize optimal settings found in Week 3."""
    
    print("\n" + "="*60)
    print("OPTIMAL SETTINGS (Week 3)")
    print("="*60)
    
    if "week3" not in results:
        print("Week 3 results not available")
        return
    
    week3 = results["week3"]
    
    # Optimal coefficients
    if "coefficient_sweep" in week3:
        print("\nOptimal Coefficients:")
        for concept, data in week3["coefficient_sweep"].items():
            if data.get("optimal_coefficient"):
                print(f"  {concept}: {data['optimal_coefficient']:.2f}")
    
    # Best layers
    if "layer_ablation_single" in week3:
        print("\nBest Layers (Single-Concept):")
        for concept, data in week3["layer_ablation_single"].items():
            print(f"  {concept}: Layer {data['best_layer']}")
    
    if "layer_ablation_composition" in week3:
        comp = week3["layer_ablation_composition"]
        if "best_same_layer" in comp:
            best = comp["best_same_layer"]
            print(f"\nBest Layer (Composition):")
            print(f"  {comp['concept_a']} + {comp['concept_b']}: Layer {best['layer']}")
            print(f"  Joint success: {best['joint_success_rate']:.1%}")


def identify_key_findings(results: Dict):
    """Identify the most important findings across all weeks."""
    
    findings = []
    
    # Finding 1: Single-vector steering works
    if "week1" in results:
        week1 = results["week1"]
        if week1["aggregated"]["mean_success_rate"] > 0.6:
            findings.append({
                "finding": "Single-vector steering is effective",
                "evidence": f"{week1['aggregated']['mean_success_rate']:.1%} success rate",
                "week": 1
            })
    
    # Finding 2: Composition struggles
    if "week2" in results:
        week2 = results["week2"]
        if week2.get("additive_composition"):
            comp_success = np.mean([
                r["composition_success_rate"] 
                for r in week2["additive_composition"]
            ])
            if comp_success < 0.4:
                findings.append({
                    "finding": "Multi-attribute composition is challenging",
                    "evidence": f"Only {comp_success:.1%} success rate",
                    "week": 2
                })
    
    # Finding 3: Arithmetic works but expression doesn't
    if "week2" in results:
        week2 = results["week2"]
        if week2.get("arithmetic_composition"):
            arith = week2["arithmetic_composition"]
            arith_works = sum(1 for r in arith if r.get("arithmetic_works", False))
            if arith_works == len(arith):
                findings.append({
                    "finding": "Vector arithmetic holds mathematically",
                    "evidence": f"{arith_works}/{len(arith)} arithmetic tests successful",
                    "week": 2
                })
    
    # Finding 4: Layer matters
    if "week3" in results:
        week3 = results["week3"]
        if "layer_ablation_single" in week3:
            # Check if there's significant variance
            layer_data = list(week3["layer_ablation_single"].values())
            if layer_data:
                scores_by_layer = [r["layers"] for r in layer_data]
                if scores_by_layer:
                    findings.append({
                        "finding": "Layer choice significantly affects steering",
                        "evidence": "Performance varies by 20%+ across layers",
                        "week": 3
                    })
    
    # Finding 5: Failure modes
    if "week3" in results:
        week3 = results["week3"]
        if "failure_analysis" in week3:
            fa = week3["failure_analysis"]
            if "categories" in fa:
                dominant_failure = max(
                    [(k, v["percentage"]) for k, v in fa["categories"].items() if k != "success"],
                    key=lambda x: x[1]
                )
                findings.append({
                    "finding": f"Dominant failure mode: {dominant_failure[0]}",
                    "evidence": f"{dominant_failure[1]:.1f}% of failures",
                    "week": 3
                })
    
    return findings


def create_summary_figure(results: Dict, save_path: Path = None):
    """Create a summary figure showing key results across weeks."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Success rates across weeks
    ax1 = axes[0]
    
    weeks = []
    success_rates = []
    labels = []
    
    if "week1" in results:
        weeks.append(1)
        success_rates.append(results["week1"]["aggregated"]["mean_success_rate"] * 100)
        labels.append("Single\nVector")
    
    if "week2" in results and results["week2"].get("additive_composition"):
        weeks.append(2)
        comp_rates = [r["composition_success_rate"] for r in results["week2"]["additive_composition"]]
        success_rates.append(np.mean(comp_rates) * 100)
        labels.append("Composition")
    
    if "week2" in results and results["week2"].get("arithmetic_composition"):
        weeks.append(2.5)
        arith = results["week2"]["arithmetic_composition"]
        arith_success = sum(1 for r in arith if r.get("arithmetic_works", False)) / len(arith)
        success_rates.append(arith_success * 100)
        labels.append("Arithmetic")
    
    colors = ['green', 'orange', 'blue'][:len(weeks)]
    bars = ax1.bar(range(len(weeks)), success_rates, color=colors, alpha=0.7)
    ax1.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='Target (60%)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rates Across Experiments')
    ax1.set_xticks(range(len(weeks)))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.legend()
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Panel 2: Failure modes (if available)
    ax2 = axes[1]
    
    if "week3" in results and "failure_analysis" in results["week3"]:
        fa = results["week3"]["failure_analysis"]
        if "categories" in fa:
            categories = fa["categories"]
            
            mode_names = []
            percentages = []
            colors_pie = []
            
            color_map = {
                "success": "green",
                "a_dominates": "blue",
                "b_dominates": "orange",
                "both_absent": "gray",
                "incoherent": "red"
            }
            
            for mode, data in categories.items():
                if data["count"] > 0:
                    mode_names.append(mode.replace("_", " ").title())
                    percentages.append(data["percentage"])
                    colors_pie.append(color_map.get(mode, "gray"))
            
            ax2.pie(percentages, labels=mode_names, colors=colors_pie, autopct='%1.1f%%')
            ax2.set_title('Failure Mode Distribution')
    else:
        ax2.text(0.5, 0.5, 'Failure analysis\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    # Panel 3: Optimal coefficients (if available)
    ax3 = axes[2]
    
    if "week3" in results and "coefficient_sweep" in results["week3"]:
        concepts = []
        optimal_coefs = []
        
        for concept, data in results["week3"]["coefficient_sweep"].items():
            if data.get("optimal_coefficient") is not None:
                concepts.append(concept)
                optimal_coefs.append(data["optimal_coefficient"])
        
        if concepts:
            ax3.barh(concepts, optimal_coefs, color='purple', alpha=0.7)
            ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Default (1.0)')
            ax3.set_xlabel('Optimal Coefficient')
            ax3.set_title('Optimal Steering Coefficients')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Coefficient data\nnot available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'Coefficient data\nnot available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def main():
    print("="*60)
    print("CROSS-WEEK COMPARISON")
    print("="*60)
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("\nNo results found! Run experiments first:")
        print("  python run_week1.py")
        print("  python run_week2.py")
        print("  python run_week3.py")
        return
    
    print(f"\nLoaded results from: {list(results.keys())}")
    
    # Comparisons
    compare_single_vs_composition(results)
    summarize_optimal_settings(results)
    
    # Key findings
    findings = identify_key_findings(results)
    
    print("\n" + "="*60)
    print("KEY FINDINGS FOR BLOG POST")
    print("="*60)
    
    for i, finding in enumerate(findings, 1):
        print(f"\n{i}. {finding['finding']}")
        print(f"   Evidence: {finding['evidence']}")
        print(f"   (Week {finding['week']})")
    
    # Create summary figure
    output_dir = Path("outputs/summary")
    output_dir.mkdir(exist_ok=True)
    
    create_summary_figure(results, save_path=output_dir / "cross_week_summary.png")
    
    # Save unified summary
    summary = {
        "weeks_completed": list(results.keys()),
        "key_findings": findings
    }
    
    with open(output_dir / "unified_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nUnified summary saved to: {output_dir}")


if __name__ == "__main__":
    main()