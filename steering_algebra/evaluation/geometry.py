"""
Geometric analysis of steering vectors.
"""

import torch
from torch import Tensor
from typing import Dict, List, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_cosine_similarity(v1: Tensor, v2: Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    v1_flat = v1.flatten().float()
    v2_flat = v2.flatten().float()
    
    return torch.nn.functional.cosine_similarity(
        v1_flat.unsqueeze(0),
        v2_flat.unsqueeze(0)
    ).item()


def compute_similarity_matrix(
    steering_vectors: Dict[str, Tensor]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise cosine similarity matrix for all steering vectors.
    
    Returns:
        Tuple of (similarity_matrix, concept_names)
    """
    concepts = list(steering_vectors.keys())
    n = len(concepts)
    sim_matrix = np.zeros((n, n))
    
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            sim_matrix[i, j] = compute_cosine_similarity(
                steering_vectors[c1],
                steering_vectors[c2]
            )
    
    return sim_matrix, concepts


def categorize_pairs_by_similarity(
    sim_matrix: np.ndarray,
    concepts: List[str],
    orthogonal_threshold: float = 0.05,
    aligned_threshold: float = 0.5
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Categorize concept pairs by their geometric relationship.
    
    Returns:
        Dict with keys: "orthogonal", "aligned", "opposing", "other"
    """
    categories = {
        "orthogonal": [],
        "aligned": [],
        "opposing": [],
        "other": []
    }
    
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            pair = (concepts[i], concepts[j], sim)
            
            if abs(sim) < orthogonal_threshold:
                categories["orthogonal"].append(pair)
            elif sim > aligned_threshold:
                categories["aligned"].append(pair)
            elif sim < -aligned_threshold:
                categories["opposing"].append(pair)
            else:
                categories["other"].append(pair)
    
    return categories


def compute_vector_norms(
    steering_vectors: Dict[str, Tensor]
) -> Dict[str, float]:
    """Compute L2 norms of all steering vectors."""
    return {
        concept: vec.norm().item()
        for concept, vec in steering_vectors.items()
    }


def project_vectors_2d(
    steering_vectors: Dict[str, Tensor],
    method: str = "pca"  # "pca" or "tsne"
) -> Dict[str, np.ndarray]:
    """
    Project steering vectors to 2D for visualization.
    
    Returns:
        Dict mapping concept -> (x, y) coordinates
    """
    concepts = list(steering_vectors.keys())
    vectors = np.array([
        steering_vectors[c].cpu().numpy().flatten()
        for c in concepts
    ])
    
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(concepts)-1))
    
    coords_2d = reducer.fit_transform(vectors)
    
    return {
        concept: coords_2d[i]
        for i, concept in enumerate(concepts)
    }


def analyze_subspace_overlap(
    vectors1: List[Tensor],
    vectors2: List[Tensor],
    n_components: int = 5
) -> float:
    """
    Measure overlap between subspaces spanned by two sets of vectors.
    Uses principal angles between subspaces.
    
    Returns:
        Overlap score in [0, 1] (1 = identical subspaces)
    """
    # Stack vectors into matrices
    V1 = torch.stack([v.flatten() for v in vectors1]).cpu().numpy()
    V2 = torch.stack([v.flatten() for v in vectors2]).cpu().numpy()
    
    # Compute orthonormal bases via SVD
    U1, _, _ = np.linalg.svd(V1.T, full_matrices=False)
    U2, _, _ = np.linalg.svd(V2.T, full_matrices=False)
    
    # Truncate to n_components
    U1 = U1[:, :min(n_components, U1.shape[1])]
    U2 = U2[:, :min(n_components, U2.shape[1])]
    
    # Compute principal angles
    M = U1.T @ U2
    singular_values = np.linalg.svd(M, compute_uv=False)
    
    # Overlap = average of squared cosines of principal angles
    overlap = np.mean(singular_values ** 2)
    
    return float(overlap)


def predict_interference(
    vec_a: Tensor,
    vec_b: Tensor
) -> Dict:
    """
    Predict whether two steering vectors will interfere.
    
    Returns analysis of potential interference patterns.
    """
    cos_sim = compute_cosine_similarity(vec_a, vec_b)
    
    # Decompose vec_b into component parallel and orthogonal to vec_a
    vec_a_norm = vec_a / vec_a.norm()
    parallel_component = (vec_b @ vec_a_norm) * vec_a_norm
    orthogonal_component = vec_b - parallel_component
    
    parallel_magnitude = parallel_component.norm().item()
    orthogonal_magnitude = orthogonal_component.norm().item()
    
    # Predict interference level
    if abs(cos_sim) < 0.2:
        interference_level = "low"
        prediction = "Vectors are approximately orthogonal; expect independent effects"
    elif cos_sim > 0.5:
        interference_level = "high_aligned"
        prediction = "Vectors are aligned; combined effect may be amplified"
    elif cos_sim < -0.5:
        interference_level = "high_opposing"
        prediction = "Vectors are opposing; effects may cancel or create noise"
    else:
        interference_level = "moderate"
        prediction = "Partial overlap; effects may partially interfere"
    
    return {
        "cosine_similarity": cos_sim,
        "parallel_magnitude": parallel_magnitude,
        "orthogonal_magnitude": orthogonal_magnitude,
        "interference_level": interference_level,
        "prediction": prediction
    }