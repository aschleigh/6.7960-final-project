"""
Visualize Coefficient Analysis Results (Post-Hoc).

Reads existing 'coefficient_results.json' and generates:
1. The new Optimization Summary Card (Optimal vs. Average).
2. Summary statistics JSON.
"""

import json
import numpy as np
from pathlib import Path
import argparse
import sys

def analyze_existing_results(input_path: Path, output_dir: Path):
    if not input_path.exists():
        print(f"Error: Results file not found at {input_path}")
        return

    print(f"Loading results from {input_path}...")
    with open(input_path) as f:
        all_results = json.load(f)

    concepts = list(all_results.keys())
    print(f"Found {len(concepts)} concepts: {concepts}")
    
    summary_data = []

    # --- SUMMARY CARD GENERATION ---
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY: OPTIMAL vs. AVERAGE COEFFICIENT")
    print("="*70)
    print(f"{'CONCEPT':<15} | {'OPT':<6} | {'OPT SCORE':<10} | {'AVG SCORE':<10} | {'GAIN':<10}")
    print("-" * 70)

    for concept in concepts:
        # Handle nested structure if present (e.g. concept -> "fine" -> "data")
        # or flat structure (concept -> "data")
        res = all_results[concept]
        if "fine" in res:
            res = res["fine"]
            
        data = res["data"]
        
        # 1. Find Optimal (Max Score)
        # We re-calculate this to be safe, using simple max score
        optimal_point = max(data, key=lambda x: x["mean_score"])
        opt_coef = optimal_point["coefficient"]
        opt_score = optimal_point["mean_score"]
        
        # 2. Calculate Average Score (Across all coefficients)
        all_scores = [d["mean_score"] for d in data]
        avg_score = float(np.mean(all_scores))
        
        # 3. Calculate Gain
        gain = opt_score - avg_score
        
        print(f"{concept:<15} | {opt_coef:<6.2f} | {opt_score:<10.3f} | {avg_score:<10.3f} | {gain:<+10.3f}")
        
        summary_data.append({
            "concept": concept,
            "optimal_coefficient": opt_coef,
            "optimal_score": opt_score,
            "average_score": avg_score,
            "gain": gain
        })

    print("-" * 70)
    
    # Calculate Macro Averages
    avg_gain = np.mean([s["gain"] for s in summary_data])
    avg_opt_score = np.mean([s["optimal_score"] for s in summary_data])
    
    print(f"{'AVERAGE':<15} | {'--':<6} | {avg_opt_score:<10.3f} | {'--':<10} | {avg_gain:<+10.3f}")
    print("="*70)

    # Save Summary JSON
    output_path = output_dir / "summary_stats_optimal_vs_average.json"
    with open(output_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSaved summary stats to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="outputs/coefficient_analysis", help="Directory containing coefficient_results.json")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    input_file = input_dir / "coefficient_results.json"
    
    analyze_existing_results(input_file, input_dir)

if __name__ == "__main__":
    main()