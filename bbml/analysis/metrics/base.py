from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any
import torch


@dataclass
class MetricResult:
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    def __init__(self, name: str = "metric"):
        self.name = name
    
    @abstractmethod
    def compare(self, weight1: torch.Tensor, weight2: torch.Tensor) -> MetricResult:
        """Compare two weight tensors and return a MetricResult."""
        pass
    
    def compute(self, weight1: torch.Tensor, weight2: torch.Tensor) -> float:
        """Backward compatibility: compute returns the score from compare."""
        result = self.compare(weight1, weight2)
        return result.score
