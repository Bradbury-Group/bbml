"""
Analysis utilities for computing layer statistics, finding redundant weights, and correlations.
"""

from typing import List, Callable, Dict, Any, Tuple
import numpy as np
from bbml.analysis.weights.units import WeightUnit
from bbml.analysis.metrics.base import Metric
from bbml.analysis.similarity import compute_similarity_matrix


def compute_layer_statistics(
    units: List[WeightUnit],
    metric: Metric,
    layer_fn: Callable[[int], List[WeightUnit]],
    symmetric: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute mean, max, and min similarity statistics for each layer.
    
    Args:
        units: List of all weight units
        metric: Metric to use for comparison
        layer_fn: Function that takes layer index and returns list of units for that layer
        symmetric: Whether to compute only upper triangular (faster)
    
    Returns:
        List of dicts with keys: layer, mean, max, min
        Sorted by mean similarity (lowest first)
    """
    results = []
    
    # Determine all layers from metadata or by iterating
    layers_set = set()
    for unit in units:
        if unit.layer is not None:
            layers_set.add(unit.layer)
    
    if not layers_set:
        raise ValueError("No layer information found in units")
    
    layers = sorted(layers_set)
    
    for layer in layers:
        layer_units = layer_fn(layer)
        if len(layer_units) == 0:
            continue
        
        # Compute similarity matrix for this layer
        n = len(layer_units)
        sim_matrix = compute_similarity_matrix(
            units=layer_units,
            metric=metric,
            symmetric=symmetric,
            show_progress=False,
        )
        
        # Compute statistics (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = sim_matrix[mask]
        
        mean_sim = float(np.mean(off_diagonal))
        max_sim = float(np.max(off_diagonal))
        min_sim = float(np.min(off_diagonal))
        
        results.append({
            "layer": layer,
            "mean": mean_sim,
            "max": max_sim,
            "min": min_sim,
        })
    
    # Sort by mean similarity (lowest first = most diverse)
    results.sort(key=lambda x: x["mean"])
    return results


def find_redundant_pairs(
    units: List[WeightUnit],
    similarity_matrix: np.ndarray,
    k: int = 10,
) -> List[Tuple[WeightUnit, WeightUnit, float]]:
    """
    Find the k most similar (redundant) weight pairs.
    
    Args:
        units: List of weight units corresponding to similarity matrix rows/cols
        similarity_matrix: NxN symmetric similarity matrix
        k: Number of top pairs to return
    
    Returns:
        List of (unit1, unit2, similarity_score) tuples sorted by similarity descending
    """
    n = len(units)
    masked_matrix = similarity_matrix.copy()
    np.fill_diagonal(masked_matrix, -np.inf)
    
    # Get top-k pairs
    flat_indices = np.argsort(masked_matrix.ravel())[-k:][::-1]
    
    results = []
    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, masked_matrix.shape)
        results.append((units[i], units[j], float(similarity_matrix[i, j])))
    
    return results


def compare_weight_correlation(
    units1: List[WeightUnit],
    units2: List[WeightUnit],
    metric: Metric,
) -> List[Dict[str, Any]]:
    """
    Compare corresponding units from two lists (e.g., Q-heads vs K-heads).
    
    Args:
        units1: First list of units (e.g., Q-heads)
        units2: Second list of units (e.g., K-heads)
        metric: Metric to use for comparison
    
    Returns:
        List of dicts with keys: unit1_key, unit2_key, similarity
    """
    if len(units1) != len(units2):
        raise ValueError(f"Lists must have same length: {len(units1)} vs {len(units2)}")
    
    results = []
    for u1, u2 in zip(units1, units2):
        result = metric.compare(u1.tensor, u2.tensor)
        results.append({
            "unit1_key": u1.key,
            "unit2_key": u2.key,
            "similarity": float(result.score),
        })
    
    return results


def compute_per_layer_mean_correlation(
    units1: List[WeightUnit],
    units2: List[WeightUnit],
    metric: Metric,
) -> List[Dict[str, Any]]:
    """
    Compare corresponding units from two lists and compute mean per layer.
    
    Args:
        units1: First list of units (e.g., Q-heads)
        units2: Second list of units (e.g., K-heads)
        metric: Metric to use for comparison
    
    Returns:
        List of dicts with keys: layer, mean_similarity
    """
    pairs = compare_weight_correlation(units1, units2, metric)
    
    # Group by layer
    layer_sims = {}
    for pair in pairs:
        # Extract layer from key (e.g., "layer0.attn.q.head0" -> 0)
        key = pair["unit1_key"]
        if "layer" in key:
            layer_str = key.split("layer")[1].split(".")[0]
            try:
                layer = int(layer_str)
                if layer not in layer_sims:
                    layer_sims[layer] = []
                layer_sims[layer].append(pair["similarity"])
            except (ValueError, IndexError):
                pass
    
    results = []
    for layer in sorted(layer_sims.keys()):
        mean_sim = float(np.mean(layer_sims[layer]))
        results.append({
            "layer": layer,
            "mean_similarity": mean_sim,
        })
    
    return results
