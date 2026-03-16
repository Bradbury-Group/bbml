from abc import ABC, abstractmethod
from typing import List

from bbml.analysis.weights.units import WeightUnit


class RepresentativePicker(ABC):
    """Selects or computes a single representative layer for a cluster.

    Called once per cluster after clustering is complete.  The chosen
    representative is later paired with the full member list to form
    a ``ClusterInput`` for the compressor.
    """

    @abstractmethod
    def pick(self, members: List[WeightUnit]) -> WeightUnit:
        """Choose or compute a representative from a cluster's members.

        Args:
            members: All weight units belonging to a single cluster.

        Returns:
            A single weight unit representing the cluster (may be one of the
            members or a newly constructed unit, depending on strategy).
        """
        pass
