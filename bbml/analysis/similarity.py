from typing import List
import numpy as np
from tqdm import tqdm
from bbml.analysis.weights.units import WeightUnit
from bbml.analysis.metrics.base import Metric


def _initialize_similarity_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _pair_indices(size: int, symmetric: bool):
    if symmetric:
        return ((i, j) for i in range(size) for j in range(i + 1, size))
    return ((i, j) for i in range(size) for j in range(size))


def _pair_count(size: int, symmetric: bool) -> int:
    if symmetric:
        return (size * (size - 1)) // 2
    return size * size


def compute_similarity_matrix(
    units: List[WeightUnit],
    metric: Metric,
    symmetric: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    n = len(units)
    matrix = _initialize_similarity_matrix(n)

    total_pairs = _pair_count(n, symmetric)
    pbar = tqdm(total=total_pairs, disable=not show_progress, desc="Computing similarities")

    for i, j in _pair_indices(n, symmetric):
        if i == j:
            pbar.update(1)
            continue

        result = metric.compare(units[i].tensor, units[j].tensor)
        similarity = result.score
        matrix[i, j] = similarity

        if symmetric:
            matrix[j, i] = similarity

        pbar.update(1)

    pbar.close()

    return matrix
