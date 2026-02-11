from abc import ABC, abstractmethod
from typing import Any
from bbml.analysis.weights.units import WeightIndex


class WeightExtractor(ABC):
    @abstractmethod
    def load(self, model: Any, device: str = "cpu") -> None:
        pass
    
    @abstractmethod
    def extract_index(
        self,
        include_heads: bool = False,
        include_full: bool = True,
        include_ffn: bool = True,
    ) -> WeightIndex:
        pass
