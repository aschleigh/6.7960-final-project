"""
Visualization utilities for Week 2 composition experiments.
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


def plot_composition_success(
    results: List[Dict],
    save_path: Path = None
):
    """
    Plot composition success for each concept pair.
    Shows whether both attributes appear in A+B generations.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pair_labels = [f"{r['concept_a']}\n+\n{r['concept_b']}" for r in results]
    success_rates = [r['composition_success_rate'] for r in results]
    similarities = [abs(r['cosine_similarity']) for r in results]
    
    # Color by similarity (orthogonal = green, aligned = red)
    colors = ['green' if s < 0.2 else 'orange' if s < 0.5 else 'red' for s in similarities]
    
    bars = ax.bar(range(len(results)), success_rates, color=colors, alpha=0.7)
    ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='60% threshold')
    
    ax.set_xlabel('Concept Pair')
    ax.set_ylabel('Composition Success Rate')
    ax.set_title('Additive Composition: Do Both Attributes Appear?')
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add similarity values on bars
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'sim={sim:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_composition_vs_individual(
    result: Dict,
    save_path: Path = None
):
    """
    Plot composition scores vs individual steering scores.
    Shows if A+B achieves both A and B or if one dominates.
    """
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    concept_a = result['concept_a']
    concept_b = result['concept_b']
    
    # Panel 1: Score comparison for concept A
    conditions = ['Baseline', 'A only', 'B only', 'A+B']
    scores_a = [
        result['baseline_scores'][concept_a],
        result['a_only_scores'][concept_a],
        result['b_only_scores'][concept_a],
        result['composition_scores'][concept_a]
    ]
    
    ax1.bar(conditions, scores_a, color=['gray', 'blue', 'lightblue', 'purple'], alpha=0.7)
    ax1.set_ylabel(f'{concept_a.capitalize()} Score')
    ax1.set_title(f'Effect on {concept_a.capitalize()}')
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Panel 2: Score comparison for concept B
    scores_b = [
        result['baseline_scores'][concept_b],
        result['a_only_scores'][concept_b],
        result['b_only_scores'][concept_b],
        result['composition_scores'][concept_b]
    ]
    
    ax2.bar(conditions, scores_b, color=['gray', 'lightgreen', 'green', 'purple'], alpha=0.7)
    ax2.set_ylabel(f'{concept_b.capitalize()} Score')
    ax2.set_title(f'Effect on {concept_b.capitalize()}')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    fig.suptitle(f'Composition Analysis: {concept_a} + {concept_b}', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_opposing_cancellation(
    results: List[Dict],
    save_path: Path = None
):
    """
    Plot how opposing vectors affect each other.
    Shows if A + (-A) returns to baseline.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_pairs = len(results)
    x = np.arange(n_pairs)
    width = 0.2
    
    baseline_means = [r['baseline_mean'] for r in results]
    a_only_means = [r['a_only_mean'] for r in results]
    b_only_means = [r['b_only_mean'] for r in results]
    composition_means = [r['composition_mean'] for r in results]
    
    ax.bar(x - 1.5*width, baseline_means, width, label='Baseline', color='gray', alpha=0.7)
    ax.bar(x - 0.5*width, a_only_means, width, label='A only', color='blue', alpha=0.7)
    ax.bar(x + 0.5*width, b_only_means, width, label='B only (opposing)', color='red', alpha=0.7)
    ax.bar(x + 1.5*width, composition_means, width, label='A + B', color='purple', alpha=0.7)
    
    labels = [f"{r['concept_a']}\n+\n{r['concept_b']}" for r in results]
    ax.set_xlabel('Concept Pair')
    ax.set_ylabel('Score')
    ax.set_title('Opposing Vector Cancellation Test')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Mark which ones cancel out
    for i, r in enumerate(results):
        if r['cancels_out']:
            ax.text(i, 0.95, '✓', ha='center', va='top', fontsize=20, color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_coefficient_heatmap(
    scaling_results: Dict,
    save_path: Path = None
):
    """
    Plot heatmap showing how coefficient ratios affect composition.
    """
    set_style()
    
    concept_a = scaling_results['concept_a']
    concept_b = scaling_results['concept_b']
    grid = scaling_results['results_grid']
    
    # Extract unique alpha and beta values
    alphas = sorted(set(r['alpha'] for r in grid.values()))
    betas = sorted(set(r['beta'] for r in grid.values()))
    
    # Create matrices for scores
    score_a_matrix = np.zeros((len(betas), len(alphas)))
    score_b_matrix = np.zeros((len(betas), len(alphas)))
    ppl_matrix = np.zeros((len(betas), len(alphas)))
    
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            key = f"a{alpha}_b{beta}"
            if key in grid:
                score_a_matrix[i, j] = grid[key]['score_a']
                score_b_matrix[i, j] = grid[key]['score_b']
                ppl_matrix[i, j] = grid[key]['perplexity']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Concept A scores
    im1 = axes[0].imshow(score_a_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title(f'{concept_a.capitalize()} Score')
    axes[0].set_xlabel(f'α ({concept_a})')
    axes[0].set_ylabel(f'β ({concept_b})')
    axes[0].set_xticks(range(len(alphas)))
    axes[0].set_yticks(range(len(betas)))
    axes[0].set_xticklabels([f'{a:.1f}' for a in alphas])
    axes[0].set_yticklabels([f'{b:.1f}' for b in betas])
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: Concept B scores
    im2 = axes[1].imshow(score_b_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title(f'{concept_b.capitalize()} Score')
    axes[1].set_xlabel(f'α ({concept_a})')
    axes[1].set_ylabel(f'β ({concept_b})')
    axes[1].set_xticks(range(len(alphas)))
    axes[1].set_yticks(range(len(betas)))
    axes[1].set_xticklabels([f'{a:.1f}' for a in alphas])
    axes[1].set_yticklabels([f'{b:.1f}' for b in betas])
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Perplexity (capped at 100 for visibility)
    ppl_matrix_capped = np.clip(ppl_matrix, 0, 100)
    im3 = axes[2].imshow(ppl_matrix_capped, cmap='Reds', aspect='auto')
    axes[2].set_title('Perplexity (capped at 100)')
    axes[2].set_xlabel(f'α ({concept_a})')
    axes[2].set_ylabel(f'β ({concept_b})')
    axes[2].set_xticks(range(len(alphas)))
    axes[2].set_yticks(range(len(betas)))
    axes[2].set_xticklabels([f'{a:.1f}' for a in alphas])
    axes[2].set_yticklabels([f'{b:.1f}' for b in betas])
    plt.colorbar(im3, ax=axes[2])
    
    fig.suptitle(f'Coefficient Scaling: {concept_a} × α + {concept_b} × β', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_geometry_vs_success(
    results: List[Dict],
    save_path: Path = None
):
    """
    Scatter plot: cosine similarity vs composition success.
    Tests if geometry predicts compositionality.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    similarities = [r['cosine_similarity'] for r in results]
    success_rates = [r['composition_success_rate'] for r in results]
    
    # Color by success
    colors = ['green' if s > 0.6 else 'orange' if s > 0.4 else 'red' for s in success_rates]
    
    ax.scatter(similarities, success_rates, c=colors, s=200, alpha=0.6, edgecolors='black')
    
    # Add concept pair labels
    for r in results:
        ax.annotate(
            f"{r['concept_a'][:4]}+{r['concept_b'][:4]}",
            (r['cosine_similarity'], r['composition_success_rate']),
            fontsize=9, ha='center', va='bottom', xytext=(0, 5), textcoords='offset points'
        )
    
    # Add reference lines
    ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.3, label='Opposing')
    ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='blue', linestyle='--', alpha=0.3, label='Aligned')
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Success threshold')
    
    # Fit and plot trend line if enough points
    if len(similarities) > 2:
        z = np.polyfit(similarities, success_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(similarities), max(similarities), 100)
        ax.plot(x_line, p(x_line), 'k-', alpha=0.3, linewidth=2)
    
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Composition Success Rate')
    ax.set_title('Does Geometry Predict Composition Success?')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_arithmetic_test(
    results: List[Dict],
    save_path: Path = None
):
    """
    Plot arithmetic composition test: does (A+B)-A ≈ B?
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_pairs = len(results)
    x = np.arange(n_pairs)
    width = 0.35
    
    b_only = [r['b_only_mean'] for r in results]
    arithmetic = [r['arithmetic_mean'] for r in results]
    
    ax.bar(x - width/2, b_only, width, label='B only (ground truth)', color='green', alpha=0.7)
    ax.bar(x + width/2, arithmetic, width, label='(A+B)-A', color='purple', alpha=0.7)
    
    labels = [f"{r['concept_a'][:4]}, {r['concept_b'][:4]}" for r in results]
    ax.set_xlabel('Concept Pair (A, B)')
    ax.set_ylabel('Score for Concept B')
    ax.set_title('Vector Arithmetic Test: Does (A+B)-A = B?')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Mark which ones work
    for i, r in enumerate(results):
        if r['arithmetic_works']:
            ax.text(i, 0.95, '✓', ha='center', va='top', fontsize=20, color='green')
        else:
            ax.text(i, 0.95, '✗', ha='center', va='top', fontsize=20, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def create_all_week2_figures(output_dir: Path):
    """Generate all Week 2 figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("\nGenerating Week 2 figures...")
    
    # Load results
    with open(output_dir / "composition_results.json") as f:
        results = json.load(f)
    
    # Figure 1: Composition success overview
    if results['additive_composition']:
        plot_composition_success(
            results['additive_composition'],
            save_path=figures_dir / "composition_success.png"
        )
    
    # Figure 2: Individual composition analysis (first result)
    if results['additive_composition']:
        plot_composition_vs_individual(
            results['additive_composition'][0],
            save_path=figures_dir / "composition_detail.png"
        )
    
    # Figure 3: Opposing cancellation
    if results['opposing_composition']:
        plot_opposing_cancellation(
            results['opposing_composition'],
            save_path=figures_dir / "opposing_cancellation.png"
        )
    
    # Figure 4: Coefficient scaling heatmap
    if 'results_grid' in results['coefficient_scaling']:
        plot_coefficient_heatmap(
            results['coefficient_scaling'],
            save_path=figures_dir / "coefficient_heatmap.png"
        )
    
    # Figure 5: Geometry vs success
    if results['additive_composition']:
        plot_geometry_vs_success(
            results['additive_composition'],
            save_path=figures_dir / "geometry_vs_success.png"
        )
    
    # Figure 6: Arithmetic test
    if results['arithmetic_composition']:
        plot_arithmetic_test(
            results['arithmetic_composition'],
            save_path=figures_dir / "arithmetic_test.png"
        )
    
    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("outputs/week2")
    
    create_all_week2_figures(output_dir)