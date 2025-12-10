"""
Visualize Steering Vectors using PCA with proper Centering and Normalization.
Fixes the "Clustering" issue by removing the common mean vector.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

from config import cfg
from extraction.extract_vectors import load_cached_vectors

def main():
    # Setup
    output_dir = Path("outputs/week3/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Vectors (Layer 16)
    layer = 16
    print(f"Loading vectors from Layer {layer}...")
    vectors_map = load_cached_vectors(Path("outputs/week1/vectors"), cfg.concepts, [layer])
    
    # Flatten to list
    labels = []
    data = []
    
    for concept, layers in vectors_map.items():
        if layer in layers:
            vec = layers[layer].float().cpu().numpy()
            labels.append(concept)
            data.append(vec)
            
    X = np.array(data)
    print(f"Data Shape: {X.shape}")

    # =========================================================================
    # THE FIX: Centering and Normalization
    # =========================================================================
    
    # 1. Normalize (Fixes "Scale" issues)
    # Makes every vector length=1. This ensures "Strong" vectors (Fantasy) don't 
    # dominate "Weak" vectors (Positive).
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    
    # 2. Center (Fixes "Clustering" issues)
    # Subtracts the average vector. This removes the "Cone Effect" so PCA 
    # focuses on the *differences* between concepts, not their shared direction.
    scaler = StandardScaler(with_std=False) # Center only (mean subtraction)
    X_centered = scaler.fit_transform(X_normalized)

    # =========================================================================
    # Run PCA
    # =========================================================================
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_centered)
    
    # Calculate explained variance (How much info is in this 2D plot?)
    var_ratio = pca.explained_variance_ratio_
    print(f"Explained Variance: PC1={var_ratio[0]:.2f}, PC2={var_ratio[1]:.2f}")

    # =========================================================================
    # Plotting
    # =========================================================================
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Color by Category
    # Define categories for cleaner coloring
    categories = {}
    for l in labels:
        if l in ["fantasy", "science"]: categories[l] = "Content"
        elif l in ["formal", "slang", "technical", "simple"]: categories[l] = "Style"
        elif l in ["positive", "negative", "unhappy", "sad", "confident", "uncertain"]: categories[l] = "Tone"
        elif l in ["smart", "intelligent"]: categories[l] = "Synonym"
        else: categories[l] = "Other"
            
    colors = [categories[l] for l in labels]
    
    # Scatter Plot
    sns.scatterplot(
        x=coords[:, 0], 
        y=coords[:, 1], 
        hue=colors, 
        style=colors,
        s=200, 
        palette="deep"
    )
    
    # Add Text Labels
    for i, label in enumerate(labels):
        plt.text(
            coords[i, 0]+0.02, 
            coords[i, 1]+0.02, 
            label, 
            fontsize=11, 
            fontweight='bold',
            alpha=0.9
        )

    plt.title(f"PCA of Steering Vectors (Layer {layer})\nCentered & Normalized", fontsize=16)
    plt.xlabel(f"PC1 ({var_ratio[0]:.1%} Variance)")
    plt.ylabel(f"PC2 ({var_ratio[1]:.1%} Variance)")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = output_dir / "pca_centered.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()