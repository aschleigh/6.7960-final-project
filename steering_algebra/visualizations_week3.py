"""
Visualization utilities for Week 3 analysis experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
from pathlib import Path
import json


def set_style():
    """Set matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_coefficient_sweep(
    results: Dict,
    save_path: Path = None
):
    """
    Plot coefficient sweep results showing score vs perplexity tradeoff.
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    concept = results["concept"]
    coefficients = [r["coefficient"] for r in results["coefficients"]]
    scores = [r["mean_score"] for r in results["coefficients"]]
    perplexities = [r["mean_perplexity"] for r in results["coefficients"]]
    
    # Clip perplexities for visualization
    perplexities_clipped = [min(p, 200) for p in perplexities]
    
    # Panel 1: Score vs Coefficient
    ax1 = axes[0]
    ax1.plot(coefficients, scores, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Threshold')
    
    if results.get("optimal_coefficient") is not None:
        opt_coef = results["optimal_coefficient"]
        opt_score = results["optimal_score"]
        ax1.axvline(x=opt_coef, color='green', linestyle=':', alpha=0.7, label=f'Optimal: {opt_coef:.2f}')
        ax1.plot(opt_coef, opt_score, 'g*', markersize=15)
    
    ax1.set_xlabel('Steering Coefficient')
    ax1.set_ylabel('Mean Score')
    ax1.set_title(f'{concept.capitalize()}: Score vs Coefficient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Perplexity vs Coefficient
    ax2 = axes[1]
    ax2.plot(coefficients, perplexities_clipped, 'o-', linewidth=2, markersize=8, color='red')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.3, label='Quality threshold')
    
    if results.get("optimal_coefficient") is not None:
        ax2.axvline(x=opt_coef, color='green', linestyle=':', alpha=0.7, label=f'Optimal: {opt_coef:.2f}')
    
    ax2.set_xlabel('Steering Coefficient')
    ax2.set_ylabel('Mean Perplexity (capped at 200)')
    ax2.set_title(f'{concept.capitalize()}: Perplexity vs Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_coefficient_sweep_combined(
    results_dict: Dict[str, Dict],
    save_path: Path = None
):
    """
    Plot coefficient sweeps for multiple concepts on same axes.
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for idx, (concept, results) in enumerate(results_dict.items()):
        coefficients = [r["coefficient"] for r in results["coefficients"]]
        scores = [r["mean_score"] for r in results["coefficients"]]
        perplexities = [min(r["mean_perplexity"], 200) for r in results["coefficients"]]
        
        # Score plot
        axes[0].plot(coefficients, scores, 'o-', linewidth=2, markersize=6, 
                    color=colors[idx], label=concept, alpha=0.7)
        
        # Perplexity plot
        axes[1].plot(coefficients, perplexities, 'o-', linewidth=2, markersize=6,
                    color=colors[idx], label=concept, alpha=0.7)
    
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Steering Coefficient')
    axes[0].set_ylabel('Mean Score')
    axes[0].set_title('Score vs Coefficient (All Concepts)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Steering Coefficient')
    axes[1].set_ylabel('Mean Perplexity (capped at 200)')
    axes[1].set_title('Perplexity vs Coefficient (All Concepts)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_layer_ablation_single(
    results: Dict,
    save_path: Path = None
):
    """
    Plot layer ablation results for a single concept.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    concept = results["concept"]
    layers = [r["layer"] for r in results["layers"]]
    scores = [r["mean_score"] for r in results["layers"]]
    
    ax.plot(layers, scores, 'o-', linewidth=2, markersize=10, color='blue')
    
    # Mark best layer
    best_layer = results["best_layer"]
    best_score = results["best_score"]
    ax.plot(best_layer, best_score, 'g*', markersize=20, label=f'Best: Layer {best_layer}')
    ax.axvline(x=best_layer, color='green', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Score')
    ax.set_title(f'Layer Ablation: {concept.capitalize()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_layer_ablation_composition(
    results: Dict,
    save_path: Path = None
):
    """
    Plot layer ablation for composition.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    concept_a = results["concept_a"]
    concept_b = results["concept_b"]
    
    # Same layer results
    layers = [r["layer"] for r in results["same_layer"]]
    joint_success = [r["joint_success_rate"] for r in results["same_layer"]]
    
    ax.plot(layers, joint_success, 'o-', linewidth=2, markersize=10, 
            color='blue', label='Same layer for both')
    
    # Different layer results (plot as separate points)
    if results["different_layers"]:
        for r in results["different_layers"]:
            label = f'A@{r["layer_a"]}, B@{r["layer_b"]}'
            # Plot at average layer position
            avg_layer = (r["layer_a"] + r["layer_b"]) / 2
            ax.plot(avg_layer, r["joint_success_rate"], 's', 
                   markersize=12, alpha=0.7, label=label)
    
    # Mark best
    if "best_same_layer" in results:
        best = results["best_same_layer"]
        ax.plot(best["layer"], best["joint_success_rate"], 
               'g*', markersize=20, label=f'Best: Layer {best["layer"]}')
    
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.3, label='Success threshold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Joint Success Rate')
    ax.set_title(f'Layer Ablation (Composition): {concept_a} + {concept_b}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_failure_modes(
    results: Dict,
    save_path: Path = None
):
    """
    Plot failure mode distribution as pie chart.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    concept_a = results["concept_a"]
    concept_b = results["concept_b"]
    categories = results["categories"]
    
    labels = []
    sizes = []
    colors = []
    
    category_info = {
        "success": ("Success\n(both present)", "green"),
        "a_dominates": (f"{concept_a.capitalize()} dominates", "blue"),
        "b_dominates": (f"{concept_b.capitalize()} dominates", "orange"),
        "both_absent": ("Both absent", "gray"),
        "incoherent": ("Incoherent\n(high perplexity)", "red")
    }
    
    for category, (label, color) in category_info.items():
        count = categories[category]["count"]
        if count > 0:
            labels.append(f"{label}\n{count} samples")
            sizes.append(count)
            colors.append(color)
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12})
    ax.set_title(f'Failure Mode Distribution: {concept_a} + {concept_b}', fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_pca_analysis(
    results: Dict,
    save_path: Path = None
):
    """
    Plot PCA analysis of steering vectors.
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Explained variance
    ax1 = axes[0]
    variance_ratio = results["explained_variance_ratio"][:10]  # Top 10
    cumulative = results["cumulative_variance"][:10]
    
    x = range(1, len(variance_ratio) + 1)
    ax1.bar(x, variance_ratio, alpha=0.7, color='blue', label='Individual')
    ax1.plot(x, cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative')
    
    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('PCA: Variance Explained by Each Component')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)
    
    # Panel 2: 2D projection
    ax2 = axes[1]
    projections = results["projections_2d"]
    
    concepts = list(projections.keys())
    xs = [projections[c][0] for c in concepts]
    ys = [projections[c][1] for c in concepts]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    
    for i, concept in enumerate(concepts):
        ax2.scatter(xs[i], ys[i], c=[colors[i]], s=200, zorder=3)
        ax2.annotate(concept, (xs[i], ys[i]), fontsize=11,
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]:.1%} var)')
    ax2.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]:.1%} var)')
    ax2.set_title('Steering Vectors Projected onto Top 2 PCs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_activation_geometry(
    results: Dict,
    save_path: Path = None
):
    """
    Plot how steering affects activation geometry.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    concept = results["concept"]
    baseline_proj = np.array(results["baseline_projections"])
    steered_proj = np.array(results["steered_projections"])
    
    # Plot baseline
    ax.scatter(baseline_proj[:, 0], baseline_proj[:, 1], 
              c='gray', s=100, alpha=0.6, label='Baseline', marker='o')
    
    # Plot steered
    ax.scatter(steered_proj[:, 0], steered_proj[:, 1],
              c='blue', s=100, alpha=0.6, label='Steered', marker='^')
    
    # Draw arrows showing movement
    for i in range(len(baseline_proj)):
        ax.arrow(baseline_proj[i, 0], baseline_proj[i, 1],
                steered_proj[i, 0] - baseline_proj[i, 0],
                steered_proj[i, 1] - baseline_proj[i, 1],
                head_width=0.02, head_length=0.03, fc='red', ec='red',
                alpha=0.3, length_includes_head=True)
    
    ax.set_xlabel(f'PC1 ({results["variance_explained"][0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({results["variance_explained"][1]:.1%} var)')
    ax.set_title(f'Activation Geometry: {concept.capitalize()} Steering\n' + 
                f'Mean distance moved: {results["mean_distance"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def create_all_week3_figures(output_dir: Path):
    """Generate all Week 3 figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("\nGenerating Week 3 figures...")
    
    # Load results
    with open(output_dir / "analysis_results.json") as f:
        results = json.load(f)
    
    # Figure 1: Coefficient sweep (individual)
    if "coefficient_sweep" in results:
        for concept, data in results["coefficient_sweep"].items():
            plot_coefficient_sweep(
                data,
                save_path=figures_dir / f"coefficient_sweep_{concept}.png"
            )
        
        # Combined plot
        if len(results["coefficient_sweep"]) > 1:
            plot_coefficient_sweep_combined(
                results["coefficient_sweep"],
                save_path=figures_dir / "coefficient_sweep_all.png"
            )
    
    # Figure 2: Layer ablation (single concept)
    if "layer_ablation_single" in results:
        for concept, data in results["layer_ablation_single"].items():
            plot_layer_ablation_single(
                data,
                save_path=figures_dir / f"layer_ablation_{concept}.png"
            )
    
    # Figure 3: Layer ablation (composition)
    if "layer_ablation_composition" in results and "best_same_layer" in results["layer_ablation_composition"]:
        plot_layer_ablation_composition(
            results["layer_ablation_composition"],
            save_path=figures_dir / "layer_ablation_composition.png"
        )
    
    # Figure 4: Failure modes
    if "failure_analysis" in results and "categories" in results["failure_analysis"]:
        plot_failure_modes(
            results["failure_analysis"],
            save_path=figures_dir / "failure_modes.png"
        )
    
    # Figure 5: PCA analysis
    if "pca_analysis" in results and "projections_2d" in results["pca_analysis"]:
        plot_pca_analysis(
            results["pca_analysis"],
            save_path=figures_dir / "pca_analysis.png"
        )
    
    # Figure 6: Activation geometry
    if "activation_geometry" in results and "baseline_projections" in results["activation_geometry"]:
        plot_activation_geometry(
            results["activation_geometry"],
            save_path=figures_dir / "activation_geometry.png"
        )
    
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("outputs/week3")
    
    create_all_week3_figures(output_dir)