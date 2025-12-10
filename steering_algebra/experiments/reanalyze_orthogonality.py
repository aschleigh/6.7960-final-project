"""
Re-analyze orthogonality results.
Comprehensive Statistics: Global & Subset (Conflict Regime).
Prints P-values for both Joint Success AND Interference.
"""

import json
import scipy.stats as stats
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 1. Load data
    results_path = Path("outputs/orthogonality/orthogonality_results.json")
    if not results_path.exists():
        print(f"Error: No results found at {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)
    
    pair_data = data["pair_data"]
    df = pd.DataFrame(pair_data)
    
    # ---------------------------------------------------------
    # ANALYSIS 1: GLOBAL (All Pairs)
    # ---------------------------------------------------------
    sims = df["similarity"]
    succs = df["joint_success"]
    intfs = df["interference"]
    
    r_glob_succ, p_glob_succ = stats.pearsonr(sims, succs)
    r_glob_int, p_glob_int = stats.pearsonr(sims, intfs)
    
    print("\n" + "="*60)
    print(f"{'GLOBAL ANALYSIS (All Pairs, N={len(df)})':^60}")
    print("="*60)
    print("  Includes Synonyms (Sim > 0), Orthogonal (~0), Opposites (< 0)")
    print("-" * 60)
    print(f"  > Sim vs Joint Success:  r = {r_glob_succ:+.3f} (p={p_glob_succ:.4f})")
    print(f"  > Sim vs Interference:   r = {r_glob_int:+.3f} (p={p_glob_int:.4f})")
    print("="*60)

    # ---------------------------------------------------------
    # ANALYSIS 2: SUBSET (Conflict vs. Orthogonal Only)
    # ---------------------------------------------------------
    # Filter out Synonyms (Similarity > 0.2)
    # This leaves only Opposites (-1.0) and Unrelated (0.0)
    subset = df[df["similarity"] < 0.2]
    
    sims_sub = subset["similarity"]
    succ_sub = subset["joint_success"]
    intf_sub = subset["interference"]
    
    r_sub_succ, p_sub_succ = stats.pearsonr(sims_sub, succ_sub)
    r_sub_int, p_sub_int = stats.pearsonr(sims_sub, intf_sub)
    
    print("\n" + "="*60)
    print(f"{'SUBSET ANALYSIS (Conflict Regime Only)':^60}")
    print("="*60)
    print(f"  Comparing Opposites vs. Orthogonal (N={len(subset)})")
    print(f"  (Excluded {len(df)-len(subset)} Synonym pairs with Sim > 0.2)")
    print("-" * 60)
    print(f"  > Sim vs Joint Success:  r = {r_sub_succ:+.3f} (p={p_sub_succ:.4e})")
    print(f"  > Sim vs Interference:   r = {r_sub_int:+.3f} (p={p_sub_int:.4e})")
    print("-" * 60)
    
    # Interpretation Logic
    significant = False
    if p_sub_succ < 0.05:
        print("  [SUCCESS] Joint Success is significant in Conflict Regime.")
        significant = True
    if p_sub_int < 0.05:
        print("  [SUCCESS] Interference is significant in Conflict Regime.")
        significant = True
        
    if not significant:
        print("  [RESULT] Still not significant. Model likely too robust (Low Coef).")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()