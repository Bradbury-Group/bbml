from dataclasses import dataclass, field
from typing import Any, Dict, List

from bbml.analysis.weights.units import WeightUnit


@dataclass
class ClusterResult:
    """Output of a LayerClusterer.

    Attributes:
        assignments: Mapping of cluster ID to the list of member weight units.
        dendrogram: Optional hierarchical linkage or dendrogram data for
            visualisation. The exact structure depends on the clustering method.
    """
    assignments: Dict[int, List[WeightUnit]]
    dendrogram: Any = None


@dataclass
class ClusterInput:
    """Input to a LayerCompressor for a single cluster.

    Attributes:
        representative: The single representative layer chosen by a
            RepresentativePicker.
        members: All weight units belonging to the cluster (including the
            representative, if applicable).
    """
    representative: WeightUnit
    members: List[WeightUnit]


@dataclass
class CompressionResult:
    """Output of the full compression pipeline.

    Attributes:
        compressed_layers: The list of compressed weight units, one per cluster.
        metadata: Arbitrary metadata produced during compression (e.g. per-cluster
            compression ratios, residual norms).
    """
    compressed_layers: List[WeightUnit]
    metadata: Dict[str, Any] = field(default_factory=dict)
