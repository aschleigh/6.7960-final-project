#!/usr/bin/env python
"""
Run the comprehensive orthogonality experiment and plot results.
"""
from experiments.cosine_similarity_analysis import main as run_analysis
from experiments.visualize_orthogonality import plot_orthogonality_results

if __name__ == "__main__":
    # 1. Run the Grid Search Experiment
    run_analysis()
    
    # 2. Generate the "Nice Plots"
    print("\n" + "="*60)
    print("Generating Visualization Plots...")
    print("="*60)
    plot_orthogonality_results()