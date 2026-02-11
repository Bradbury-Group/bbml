import torch
import torch.nn.functional as F
from bbml.analysis.metrics.base import Metric


class CosineMetric(Metric):
    def compute(self, weight1: torch.Tensor, weight2: torch.Tensor) -> float:
        w1_flat = weight1.flatten()
        w2_flat = weight2.flatten()
        
        max_len = max(w1_flat.size(0), w2_flat.size(0))
        
        if w1_flat.size(0) < max_len:
            w1_flat = F.pad(w1_flat, (0, max_len - w1_flat.size(0)))
        if w2_flat.size(0) < max_len:
            w2_flat = F.pad(w2_flat, (0, max_len - w2_flat.size(0)))
        
        similarity = F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0))
        return similarity.item()
