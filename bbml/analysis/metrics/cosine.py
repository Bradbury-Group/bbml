import torch
import torch.nn.functional as F
from bbml.analysis.metrics.base import Metric, MetricResult


class CosineMetric(Metric):
    def __init__(self, name: str = "cosine"):
        super().__init__(name=name)
    
    def compare(self, weight1: torch.Tensor, weight2: torch.Tensor) -> MetricResult:
        w1_flat = weight1.flatten()
        w2_flat = weight2.flatten()
        
        max_len = max(w1_flat.size(0), w2_flat.size(0))
        
        if w1_flat.size(0) < max_len:
            w1_flat = F.pad(w1_flat, (0, max_len - w1_flat.size(0)))
        if w2_flat.size(0) < max_len:
            w2_flat = F.pad(w2_flat, (0, max_len - w2_flat.size(0)))
        
        similarity = F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0))
        score = similarity.item()
        
        # Return MetricResult with details
        return MetricResult(
            score=score,
            details={
                "w1_norm": float(torch.norm(w1_flat).item()),
                "w2_norm": float(torch.norm(w2_flat).item()),
                "w1_shape": tuple(weight1.shape),
                "w2_shape": tuple(weight2.shape),
            }
        )
