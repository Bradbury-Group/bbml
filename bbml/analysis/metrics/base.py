from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class MetricResult:
    value: float
    metadata: Dict[str, Any]


class Metric(ABC):
    @abstractmethod
    def compute(self, weight1: torch.Tensor, weight2: torch.Tensor) -> float:
        pass
