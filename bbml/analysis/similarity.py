from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from bbml.analysis.weights.units import WeightUnit
from bbml.analysis.metrics.base import Metric


@dataclass
class SimilarityReport:
    """Structured report accompanying a similarity matrix.

    Attributes:
        matrix: The computed similarity matrix.
        labels: Optional labels corresponding to each row/column.
        details: Arbitrary extra information (e.g. metric name, parameters).
    """
    matrix: np.ndarray
    labels: Optional[List[str]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SimilarityMetric(ABC):
    """Computes a pairwise similarity matrix over a list of layers.

    Implementations define how every pair of weight units is compared,
    producing an N×N similarity matrix.
    """

    @abstractmethod
    def compute_matrix(
        self,
        layers: List[WeightUnit],
    ) -> np.ndarray:
        """Compute the pairwise similarity matrix.

        Args:
            layers: The list of (possibly transformed) weight units.

        Returns:
            A 2-D numpy array of shape ``(N, N)`` where ``N = len(layers)``.
        """
        pass

    def compute_report(
        self,
        layers: List[WeightUnit],
    ) -> SimilarityReport:
        """Compute the similarity matrix together with a structured report.

        The default implementation delegates to ``compute_matrix`` and wraps
        the result in a ``SimilarityReport`` with layer keys as labels.
        Subclasses may override this to attach richer details.

        Args:
            layers: The list of weight units.

        Returns:
            A ``SimilarityReport`` containing the matrix and metadata.
        """
        matrix = self.compute_matrix(layers)
        labels = [unit.key for unit in layers]
        return SimilarityReport(matrix=matrix, labels=labels)


# ---------------------------------------------------------------------------
# Legacy helper functions (pre-interface API, kept for backward compatibility)
# ---------------------------------------------------------------------------


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
