from abc import ABC, abstractmethod

from bbml.analysis.containers import ClusterInput
from bbml.analysis.weights.units import WeightUnit


class LayerCompressor(ABC):
    """Compresses a cluster into a single compact layer.

    The compressor receives the representative chosen by a
    ``RepresentativePicker`` together with all member layers in the cluster
    and produces a single compressed weight unit ready to be reassembled
    into a model.
    """

    @abstractmethod
    def compress(self, cluster_input: ClusterInput) -> WeightUnit:
        """Compress a cluster into one layer.

        Args:
            cluster_input: A ``ClusterInput`` containing the representative
                and the full list of cluster members.

        Returns:
            A single compressed weight unit.
        """
        pass
