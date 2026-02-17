from typing import List
import numpy as np
from tqdm import tqdm
from bbml.analysis.weights.units import WeightUnit
from bbml.analysis.metrics.base import Metric


def compute_similarity_matrix(
    units: List[WeightUnit],
    metric: Metric,
    symmetric: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    n = len(units)
    matrix = np.zeros((n, n), dtype=np.float32)
    
    if symmetric:
        total_pairs = (n * (n - 1)) // 2
        pbar = tqdm(total=total_pairs, disable=not show_progress, desc="Computing similarities")
        
        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                result = metric.compare(units[i].tensor, units[j].tensor)
                sim = result.score
                matrix[i, j] = sim
                matrix[j, i] = sim
                pbar.update(1)
        
        pbar.close()
    else:
        total_pairs = n * n
        pbar = tqdm(total=total_pairs, disable=not show_progress, desc="Computing similarities")
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    result = metric.compare(units[i].tensor, units[j].tensor)
                    sim = result.score
                    matrix[i, j] = sim
                pbar.update(1)
        
        pbar.close()
    
    return matrix
