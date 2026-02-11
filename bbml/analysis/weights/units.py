from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch


@dataclass
class WeightUnit:
    key: str
    weight: torch.Tensor
    kind: str
    layer: Optional[int] = None
    head: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    ) -> "WeightIndex":
        filtered = self._units
        
        if kind is not None:
            filtered = [u for u in filtered if u.kind == kind]
        
        if layer is not None:
            filtered = [u for u in filtered if u.layer == layer]
        
        if head is not None:
            filtered = [u for u in filtered if u.head == head]
        
        return WeightIndex(filtered)
    
    def get_kinds(self) -> List[str]:
        return sorted(set(u.kind for u in self._units))
    
    def get_layers(self) -> List[int]:
        layers = [u.layer for u in self._units if u.layer is not None]
        return sorted(set(layers))
    
    def get_heads(self) -> List[int]:
        heads = [u.head for u in self._units if u.head is not None]
        return sorted(set(heads))
