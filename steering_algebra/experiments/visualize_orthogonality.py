"""
Visualization for Comprehensive Orthogonality Experiment.
Features:
- Jitter to handle density.
- Zoomed plots to reveal structure in the orthogonal cluster.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_orthogonality_results(
    results_path: Path = Path("outputs/orthogonality/orthogonality_results.json"),
    output_dir: Path = Path("outputs/orthogonality/figures")
):
    # 1. Load Data
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)
    
    df = pd.DataFrame(data["pair_data"])
    
    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set Style
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

    # =========================================================================
    # Plot 1: Similarity vs. Interference (Joint Plot with Density)
    # =========================================================================
    g = sns.jointplot(
        data=df, 
        x="abs_similarity", 
        y="interference",
        kind="reg",
        color="crimson",
        height=8,
        space=0,
        xlim=(-0.05, 1.05),
        ylim=(-0.2, 1.0),
        scatter_kws={'alpha': 0.5, 's': 80},
        truncate=False
    )
    
    ax = g.ax_joint
    
    # Highlight the "Safe Zone" (Orthogonal)
    ax.axvspan(-0.05, 0.1, color='green', alpha=0.1, label='Orthogonal Zone')
    
    # Label interesting outliers
    interesting = df[(df["abs_similarity"] > 0.4) | (df["interference"] > 0.3)]
    
    for _, row in interesting.iterrows():
        ax.text(
            row["abs_similarity"] + 0.02, 
            row["interference"], 
            row["pair"], 
            fontsize=10,
            color='darkred',
            fontweight='bold',
            alpha=0.9
        )

    ax.set_xlabel("Absolute Cosine Similarity (|cos θ|)")
    ax.set_ylabel("Interference Score")
    
    r = data["correlations"]["similarity_vs_interference"]["r"]
    p = data["correlations"]["similarity_vs_interference"]["p"]
    stats_text = f"Pearson r = {r:.2f}\np-value = {p:.3f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            ha='right', va='top', bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    
    g.fig.suptitle("Geometry vs. Behavior: Interference Regression", y=1.02)
    
    save_path = output_dir / "similarity_vs_interference.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")

    # =========================================================================
    # Plot 2: Similarity vs. Joint Success (FULL VIEW)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    sns.regplot(
        data=df,
        x="abs_similarity",
        y="joint_success",
        color="seagreen",
        x_jitter=0.015,
        y_jitter=0.015,
        scatter_kws={'alpha':0.5, 's':80},
        line_kws={'color': 'black', 'linestyle':'--'}
    )
    
    plt.title("Geometry vs. Success (Full Spectrum)")
    plt.xlabel("Absolute Cosine Similarity")
    plt.ylabel("Joint Success Score")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.1)
    
    r = data["correlations"]["similarity_vs_success"]["r"]
    p = data["correlations"]["similarity_vs_success"]["p"]
    plt.text(0.05, 0.1, f"Global Correlation:\nr={r:.2f}, p={p:.3f}", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_vs_success.png", dpi=300)
    print(f"Saved: {output_dir / 'similarity_vs_success.png'}")

    # =========================================================================
    # Plot 3: Similarity vs. Joint Success (ZOOMED VIEW)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    subset = df[df["abs_similarity"] < 0.3]
    
    if len(subset) > 5:
        sns.regplot(
            data=subset,
            x="abs_similarity",
            y="joint_success",
            color="teal",
            x_jitter=0.005,
            y_jitter=0.01,
            scatter_kws={'alpha':0.6, 's':120, 'edgecolor':'white'},
            # FIXED COLOR NAME HERE:
            line_kws={'color': 'darkslategray', 'linestyle':':'} 
        )
        
        plt.title("Correlation Between Feature Alignment and Joint Success")
        plt.xlabel("Absolute Cosine Similarity")
        plt.ylabel("Joint Success Score")
        plt.xlim(0, 0.3)
        plt.ylim(-0.05, 1.1)
        
        # Label extremes in this cluster
        best = subset.nlargest(2, "joint_success")
        # worst = subset.nsmallest(2, "joint_success")
        
        for _, row in pd.concat([best, worst]).iterrows():
            plt.text(
                row["abs_similarity"] + 0.005, 
                row["joint_success"], 
                row["pair"], 
                fontsize=9,
                color='black',
                weight='bold',
                alpha=0.8
            )
            
        plt.tight_layout()
        plt.savefig(output_dir / "similarity_vs_success_ZOOMED.png", dpi=300)
        print(f"Saved: {output_dir / 'similarity_vs_success_ZOOMED.png'}")
    else:
        print("Skipping zoomed plot (not enough points < 0.3)")

    # =========================================================================
    # Plot 4: Binned Analysis
    # =========================================================================
    plt.figure(figsize=(10, 6))
    
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    labels = ["Very Orthogonal\n(<0.05)", "Orthogonal\n(0.05-0.1)", "Low\n(0.1-0.2)", "Mid\n(0.2-0.5)", "High\n(>0.5)"]
    df["sim_bin"] = pd.cut(df["abs_similarity"], bins=bins, labels=labels, include_lowest=True)
    
    sns.barplot(
        data=df,
        x="sim_bin",
        y="interference",
        palette="Reds",
        capsize=.1
    )
    
    plt.title("Interference by Similarity Zone")
    plt.xlabel("Geometric Similarity")
    plt.ylabel("Avg Interference Score")
    
    plt.tight_layout()
    plt.savefig(output_dir / "binned_interference.png", dpi=300)
    print(f"Saved: {output_dir / 'binned_interference.png'}")

if __name__ == "__main__":
    plot_orthogonality_results()