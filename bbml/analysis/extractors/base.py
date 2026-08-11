from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict

import torch
from torch import nn

from bbml.analysis.weights.units import WeightIndex


class WeightExtractor(ABC):
    """Read, write, and replace weights in a transformer model.

    Subclasses encode architecture-specific knowledge (fused QKV layout,
    FFN paths, etc.) once.  Three capabilities:

    1. **Read**: ``extract_index`` clones weights into WeightUnits.
    2. **Write**: ``set_weight`` writes tensors back into live parameters.
    3. **Replace**: ``get_module`` / ``replace_module`` swap nn.Modules
       for compressed variants (ShareLinear, DeltaLinear, etc.).
    """

    # -- Read path -----------------------------------------------------------

    @abstractmethod
    def load(self, model: Any, device: str = "cpu") -> "WeightExtractor":
        """Load weights from a model. Returns self for method chaining."""

    @abstractmethod
    def extract_index(
        self,
        include_heads: bool = False,
        include_full: bool = True,
        include_ffn: bool = True,
    ) -> WeightIndex:
        """Extract a read-only WeightIndex of cloned tensors."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Model configuration (n_layers, n_head, n_embd, etc.)."""

    # -- Write path ----------------------------------------------------------

    def get_weight(
        self, layer: int, kind: str, *, head: int | None = None,
    ) -> torch.Tensor:
        """Read a live parameter slice (not cloned).

        Args:
            layer: Layer index.
            kind: Canonical kind string (``"attn.q"``, ``"ffn.up"``, etc.).
            head: Optional head index for per-head access.

        Returns:
            View into the model parameter tensor.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_weight"
        )

    def set_weight(
        self, layer: int, kind: str, value: torch.Tensor,
        *, head: int | None = None,
    ) -> None:
        """Write a tensor into a live parameter in-place.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            value: Tensor to write (cast to param dtype/device automatically).
            head: Optional head index for per-head writes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement set_weight"
        )

    # -- Module replacement --------------------------------------------------

    def get_module(self, layer: int, kind: str) -> nn.Module:
        """Return the nn.Module that owns the weight for (layer, kind).

        Args:
            layer: Layer index.
            kind: Canonical kind string.

        Returns:
            The live nn.Module (e.g. nn.Linear for ``"ffn.up"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_module"
        )

    def replace_module(
        self, layer: int, kind: str, new_module: nn.Module,
    ) -> None:
        """Swap the nn.Module for (layer, kind) with a compressed variant.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            new_module: Replacement module (e.g. ShareLinear, DeltaLinear).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement replace_module"
        )

    # -- Trial context managers (convenience, concrete) ----------------------

    @contextmanager
    def trial(self, layer: int, kind: str, *, head: int | None = None):
        """Snapshot a weight, yield, restore on exit.

        Usage::

            with extractor.trial(0, "attn.q"):
                extractor.set_weight(0, "attn.q", reconstructed)
                ppl = compute_perplexity(model, ...)
            # original weights restored here
        """
        original = self.get_weight(layer, kind, head=head).clone()
        try:
            yield
        finally:
            self.set_weight(layer, kind, original, head=head)

    @contextmanager
    def trial_param(self, name: str):
        """Snapshot an arbitrary named parameter, restore on exit.

        Args:
            name: Fully qualified parameter name (e.g.
                  ``"transformer.h.0.attn.c_attn.weight"``).
        """
        param = self._get_named_param(name)
        original = param.data.clone()
        try:
            yield
        finally:
            param.data.copy_(original)

    def _get_named_param(self, name: str) -> nn.Parameter:
        """Lookup a parameter by its fully qualified name.

        Subclasses must implement or override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _get_named_param"
        )


# Alias for compatibility with gptcompress naming
ModelAdapter = WeightExtractor
