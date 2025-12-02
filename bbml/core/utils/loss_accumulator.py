import warnings
from torch import Tensor
import torch

class LossAccumulator:

    def __init__(
        self,
        terms: dict[str, int|float|dict],
        step: int|None=None,
        split: str|None=None,
        term_groups: dict[str, tuple[str, ...]]|None = None,
    ):
        self.terms = terms
        self.step = step
        self.term_groups = term_groups
        if split is not None:
            self.split = split
            self.prefix = f"{self.split}_"
        else:
            self.split = ""
            self.prefix = ""
        self.unweighted: dict[str, Tensor] = {}
        self.weighted: dict[str, Tensor] = {}

    def resolve_weight(self, name: str) -> float:
        """
        loss weight spec:
          - float | int
          - dict: {"start": int, "end": int, "min": float, "max": float}
        """
        spec = self.terms.get(name, 0.0)
        
        if isinstance(spec, (int, float)):
            return float(spec)
        
        if isinstance(spec, dict):
            try:
                start = int(spec.get("start", 0))
                end = int(spec["end"])  # required
                vmin = float(spec.get("min", 0.0))
                vmax = float(spec["max"])  # required
            except Exception:
                warnings.warn(f"Malformed dict {spec}; treat as disabled")
                return 0.0

            if self.step <= start:
                return vmin
            if self.step >= end:
                return vmax
            span = max(1, end - start)
            t = (self.step - start) / span
            return vmin + (vmax - vmin) * t
        
        warnings.warn(f"Unknown spec type {spec}; treat as disabled")
        return 0.0

    def has_group(self, name: str):
        if name not in self.term_groups:
            return False
        all_group_terms = self.term_groups[name]
        return any([self.resolve_weight(tn) > 0 for tn in all_group_terms])

    def has(self, name: str) -> bool:
        return self.resolve_weight(name) > 0

    def accum(self, name: str, loss_value: Tensor, extra_weight: float|None = None) -> Tensor:
        self.unweighted[name] = loss_value

        w = self.resolve_weight(name) 
        if extra_weight is not None:
            w *= float(extra_weight)

        weighted = loss_value * w
        self.weighted[name] = weighted
        return weighted
    
    @property
    def total(self):
        weighted_losses = list(self.weighted.values())
        return torch.stack(weighted_losses).sum()


    def logs(self) -> dict[str, float]:
        # append prefix and suffix for logs
        logs: dict[str, float] = {}
        for k, v in self.unweighted.items():
            logs[f"{self.prefix}_{k}"] = float(v.detach().item())
        for k, v in self.weighted.items():
            logs[f"{self.prefix}_{k}_weighted"] = float(v.detach().item())
        return logs

