from abc import ABC, abstractmethod
from typing import List

import numpy as np

from bbml.analysis.containers import ClusterResult
from bbml.analysis.weights.units import WeightUnit


class LayerClusterer(ABC):
    """Clusters layers based on a precomputed similarity matrix.

    The clusterer groups layers that are redundant or structurally similar.
    It is solely responsible for grouping — it does **not** pick
    representatives.
    """

    @abstractmethod
    def cluster(
        self,
        similarity_matrix: np.ndarray,
        layers: List[WeightUnit],
    ) -> ClusterResult:
        """Assign layers to clusters.

        Args:
            similarity_matrix: An N×N pairwise similarity matrix.
            layers: The original list of weight units (length N), aligned
                with the rows/columns of ``similarity_matrix``.

        Returns:
            A ``ClusterResult`` with cluster assignments and optional
            dendrogram data.
        """
        pass
