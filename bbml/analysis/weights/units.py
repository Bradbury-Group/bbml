from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch


@dataclass
class WeightUnit:
    key: str
    tensor: torch.Tensor
    kind: str
    layer: Optional[int] = None
    head: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # Backward compatibility property
    @property
    def weight(self) -> torch.Tensor:
        return self.tensor
    
    # Backward compatibility property for old 'metadata' name
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.meta
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        self.meta = value


class WeightIndex:
    def __init__(self, units: List[WeightUnit]):
        self._units = units
    
    def __len__(self) -> int:
        return len(self._units)
    
    def __getitem__(self, idx: int) -> WeightUnit:
        return self._units[idx]
    
    def __iter__(self):
        return iter(self._units)
    
    def select(
        self,
        kind: Optional[str] = None,
        layer: Optional[int] = None,
        head: Optional[int] = None,
    ) -> List[WeightUnit]:
        filtered = self._units
        
        if kind is not None:
            filtered = [u for u in filtered if u.kind == kind]
        
        if layer is not None:
            filtered = [u for u in filtered if u.layer == layer]
        
        if head is not None:
            filtered = [u for u in filtered if u.head == head]
        
        return filtered
    
    def add(self, unit: WeightUnit) -> "WeightIndex":
        """Add a unit to the index and return self for chaining."""
        self._units.append(unit)
        return self
    
    def kinds(self) -> List[str]:
        """Get all unique kinds in this index."""
        return sorted(set(u.kind for u in self._units))
    
    def layers(self) -> List[int]:
        """Get all unique layers in this index."""
        layers = [u.layer for u in self._units if u.layer is not None]
        return sorted(set(layers))
    
    def heads(self) -> List[int]:
        """Get all unique heads in this index."""
        heads = [u.head for u in self._units if u.head is not None]
        return sorted(set(heads))
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the index contents."""
        return {
            "total_units": len(self._units),
            "kinds": self.kinds(),
            "layers": self.layers(),
            "heads": self.heads(),
        }
