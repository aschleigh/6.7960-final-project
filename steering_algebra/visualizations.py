"""
Visualization utilities for creating publication-quality figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def set_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    concepts: List[str],
    save_path: Path = None,
    title: str = "Steering Vector Cosine Similarity"
):
    """Plot a heatmap of cosine similarities between steering vectors."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')
    
    # Set ticks
    ax.set_xticks(range(len(concepts)))
    ax.set_yticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.set_yticklabels(concepts)
    
    # Add value annotations
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}',
                   ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_steering_effect(
    baseline_scores: List[float],
    steered_scores: List[float],
    concept: str,
    save_path: Path = None
):
    """Plot comparison of baseline vs steered classifier scores."""
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram comparison
    ax1 = axes[0]
    bins = np.linspace(0, 1, 21)
    ax1.hist(baseline_scores, bins=bins, alpha=0.6, label='Baseline', color='gray')
    ax1.hist(steered_scores, bins=bins, alpha=0.6, label='Steered', color='blue')
    ax1.axvline(np.mean(baseline_scores), color='gray', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(steered_scores), color='blue', linestyle='--', linewidth=2)
    ax1.set_xlabel(f'{concept.capitalize()} Score')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution of {concept.capitalize()} Scores')
    ax1.legend()
    
    # Paired comparison
    ax2 = axes[1]
    ax2.scatter(baseline_scores, steered_scores, alpha=0.5, s=50)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No change')
    ax2.set_xlabel('Baseline Score')
    ax2.set_ylabel('Steered Score')
    ax2.set_title(f'Paired Comparison: {concept.capitalize()}')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Add statistics
    improvement = np.mean(steered_scores) - np.mean(baseline_scores)
    success_rate = np.mean(np.array(steered_scores) > np.array(baseline_scores))
    fig.suptitle(f'Improvement: {improvement:+.3f} | Success Rate: {success_rate:.1%}', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_coefficient_sweep(
    coefficients: List[float],
    scores: List[float],
    perplexities: List[float],
    concept: str,
    save_path: Path = None
):
    """Plot steering effectiveness vs coefficient with perplexity."""
    set_style()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot scores
    color1 = 'tab:blue'
    ax1.set_xlabel('Steering Coefficient')
    ax1.set_ylabel(f'{concept.capitalize()} Score', color=color1)
    ax1.plot(coefficients, scores, 'o-', color=color1, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1)
    
    # Plot perplexity on secondary axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Perplexity', color=color2)
    ax2.plot(coefficients, perplexities, 's--', color=color2, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Find optimal coefficient (best score with reasonable perplexity)
    baseline_ppl = perplexities[0] if coefficients[0] == 0 else perplexities[coefficients.index(min(coefficients))]
    valid_idx = [i for i, p in enumerate(perplexities) if p < baseline_ppl * 2]
    if valid_idx:
        best_idx = max(valid_idx, key=lambda i: scores[i])
        ax1.axvline(coefficients[best_idx], color='green', linestyle=':', alpha=0.7,
                   label=f'Suggested: {coefficients[best_idx]}')
        ax1.legend()
    
    ax1.set_title(f'Coefficient Sweep: {concept.capitalize()}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, (ax1, ax2)


def plot_vector_projection(
    coordinates: Dict[str, np.ndarray],
    save_path: Path = None,
    title: str = "Steering Vectors (2D Projection)"
):
    """Plot 2D projection of steering vectors."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot points
    concepts = list(coordinates.keys())
    xs = [coordinates[c][0] for c in concepts]
    ys = [coordinates[c][1] for c in concepts]
    
    # Color by concept type
    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))
    
    for i, concept in enumerate(concepts):
        ax.scatter(xs[i], ys[i], c=[colors[i]], s=200, zorder=3)
        ax.annotate(concept, (xs[i], ys[i]), fontsize=12,
                   xytext=(5, 5), textcoords='offset points')
    
    # Draw lines between opposing concepts
    opposing_pairs = [
        ("formal", "casual"),
        ("positive", "negative"),
        ("verbose", "concise"),
        ("confident", "uncertain"),
        ("technical", "simple")
    ]
    
    for c1, c2 in opposing_pairs:
        if c1 in coordinates and c2 in coordinates:
            ax.plot([coordinates[c1][0], coordinates[c2][0]],
                   [coordinates[c1][1], coordinates[c2][1]],
                   'k--', alpha=0.3, linewidth=1)
    
    # Draw origin
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_extraction_quality(
    quality_results: Dict[str, Dict[int, Dict]],
    metric: str = "cohens_d",
    save_path: Path = None
):
    """Plot extraction quality across concepts and layers."""
    set_style()
    
    concepts = list(quality_results.keys())
    layers = list(quality_results[concepts[0]].keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(concepts))
    width = 0.8 / len(layers)
    
    for i, layer in enumerate(layers):
        values = [quality_results[c][layer][metric] for c in concepts]
        offset = (i - len(layers)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=f'Layer {layer}')
    
    ax.set_xlabel('Concept')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Extraction Quality: {metric.replace("_", " ").title()}')
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=45, ha='right')
    ax.legend(title='Layer', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add threshold line for Cohen's d
    if metric == "cohens_d":
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label="Large effect (0.8)")
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label="Medium effect (0.5)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_all_results_summary(
    results: Dict,
    save_path: Path = None
):
    """Create a comprehensive summary figure with multiple panels."""
    set_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Success rates by concept
    ax1 = fig.add_subplot(2, 2, 1)
    concepts = list(results["per_concept"].keys())
    success_rates = [results["per_concept"][c]["statistics"]["success_rate"] for c in concepts]
    colors = ['green' if s > 0.5 else 'red' for s in success_rates]
    bars = ax1.bar(concepts, success_rates, color=colors, alpha=0.7)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Steering Success Rate by Concept')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 2: Improvement by concept
    ax2 = fig.add_subplot(2, 2, 2)
    improvements = [results["per_concept"][c]["statistics"]["improvement"] for c in concepts]
    colors = ['green' if i > 0 else 'red' for i in improvements]
    ax2.bar(concepts, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Score Improvement')
    ax2.set_title('Mean Score Improvement by Concept')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 3: Perplexity change
    ax3 = fig.add_subplot(2, 2, 3)
    ppl_deltas = [results["per_concept"][c]["quality"]["delta_perplexity_mean"] for c in concepts]
    colors = ['green' if d < 0 else 'orange' if d < 50 else 'red' for d in ppl_deltas]
    ax3.bar(concepts, ppl_deltas, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_ylabel('Perplexity Change')
    ax3.set_title('Generation Quality Impact (Lower is Better)')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 4: Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    ==================
    
    Total Concepts Tested: {len(concepts)}
    
    Concepts with Positive Improvement: {sum(1 for i in improvements if i > 0)}/{len(concepts)}
    
    Mean Success Rate: {np.mean(success_rates):.1%}
    
    Mean Improvement: {np.mean(improvements):+.3f}
    
    Best Concept: {concepts[np.argmax(improvements)]} (+{max(improvements):.3f})
    
    Worst Concept: {concepts[np.argmin(improvements)]} ({min(improvements):+.3f})
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=14,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Week 1 Validation Results Summary', fontsize=18, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def create_all_figures(output_dir: Path, results: Dict):
    """Generate all figures for the Week 1 report."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("\nGenerating figures...")
    
    # Load geometry results
    import json
    with open(output_dir / "geometry_analysis.json") as f:
        geometry = json.load(f)
    
    # 1. Similarity matrix
    sim_matrix = np.array(geometry["similarity_matrix"])
    concepts = geometry["concepts"]
    plot_similarity_matrix(
        sim_matrix, concepts,
        save_path=figures_dir / "similarity_matrix.png"
    )
    
    # 2. Vector projection
    from evaluation.geometry import project_vectors_2d
    # Note: Need to load vectors for this
    
    # 3. Summary figure
    plot_all_results_summary(
        results,
        save_path=figures_dir / "summary.png"
    )
    
    print(f"\nAll figures saved to: {figures_dir}")