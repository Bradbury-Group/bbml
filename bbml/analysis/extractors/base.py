from abc import ABC, abstractmethod
from typing import Any, Dict
from bbml.analysis.weights.units import WeightIndex


class WeightExtractor(ABC):
    @abstractmethod
    def load(self, model: Any, device: str = "cpu") -> "WeightExtractor":
        """Load weights from a model. Returns self for method chaining."""
        pass
    
    @abstractmethod
    def extract_index(
        self,
        include_heads: bool = False,
        include_full: bool = True,
        include_ffn: bool = True,
    ) -> WeightIndex:
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration like n_layers, n_head, n_embd, etc."""
        pass


# Alias for compatibility with gptcompress naming
ModelAdapter = WeightExtractor
