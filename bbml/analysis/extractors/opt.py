"""Weight extractor for OPT models (Meta/Facebook).

Handles ``OPTForCausalLM``-compatible models whose decoder layers expose
``self_attn.{q_proj, k_proj, v_proj, out_proj}`` and ``{fc1, fc2}``.

Unlike LLaMA, OPT uses:
  - ``model.model.decoder.layers[i]`` (extra ``.decoder`` level)
  - ``out_proj`` (not ``o_proj``) for attention output
  - Simple ReLU MLP with ``fc1`` / ``fc2`` (no SwiGLU gating)
  - Bias on all Linear layers
  - Pre-LN (LayerNorm before attention/FFN, not RMSNorm)

Canonical kind vocabulary (subset — no ``ffn.gate``):
    attn.q, attn.k, attn.v, attn.out   — attention projections
    ffn.up, ffn.down                    — MLP projections (fc1, fc2)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightIndex, WeightUnit

_ATTN_KINDS = frozenset({"attn.q", "attn.k", "attn.v", "attn.out"})
_FFN_KINDS = frozenset({"ffn.up", "ffn.down"})
_ALL_KINDS = _ATTN_KINDS | _FFN_KINDS

# Maps canonical kind → (sub-module attr, projection attr) within a decoder layer
_KIND_PATH: dict[str, tuple[str, str]] = {
    "attn.q":   ("self_attn", "q_proj"),
    "attn.k":   ("self_attn", "k_proj"),
    "attn.v":   ("self_attn", "v_proj"),
    "attn.out": ("self_attn", "out_proj"),
    "ffn.up":   (None,        "fc1"),
    "ffn.down": (None,        "fc2"),
}


def _is_hf_opt(model: Any) -> bool:
    """Accept any OPTForCausalLM-compatible model by structural check."""
    try:
        layers = model.model.decoder.layers
        attn = layers[0].self_attn
        return (
            hasattr(attn, "q_proj")
            and hasattr(attn, "out_proj")
            and hasattr(layers[0], "fc1")
        )
    except (AttributeError, IndexError):
        return False


class OPTWeightExtractor(WeightExtractor):
    """Weight extractor for OPT (separate Q/K/V, simple MLP, bias on all).

    ``load()`` accepts an ``OPTForCausalLM`` (or structurally equivalent)
    HuggingFace model.  MHA only — all heads share the same count.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._n_embd: int = 0
        self._d_head: int = 0
        self._ffn_dim: int = 0
        self._vocab_size: int = 0

    # ---- Read path ---------------------------------------------------------

    def load(self, model: Any, device: str = "cpu") -> "OPTWeightExtractor":
        """Load from a HuggingFace OPTForCausalLM (or compatible).

        Args:
            model: ``OPTForCausalLM`` instance.
            device: Ignored (kept for API compatibility).

        Returns:
            self, for method chaining.
        """
        if not _is_hf_opt(model):
            raise ValueError(
                "Model must be an OPTForCausalLM or structurally compatible "
                "(model.model.decoder.layers[i].self_attn.{q,k,v,out}_proj)"
            )
        self._model = model
        cfg = model.config
        self._n_layers = cfg.num_hidden_layers
        self._n_heads = cfg.num_attention_heads
        self._n_embd = cfg.hidden_size
        self._d_head = self._n_embd // self._n_heads
        self._ffn_dim = cfg.ffn_dim
        self._vocab_size = cfg.vocab_size
        return self

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration dict.

        Returns:
            Dict with n_layers, n_head, n_kv_head, n_embd, intermediate_size,
            vocab_size, head_dim.
        """
        self._check_loaded()
        return {
            "n_layers":          self._n_layers,
            "n_head":            self._n_heads,
            "n_kv_head":         self._n_heads,  # MHA
            "n_embd":            self._n_embd,
            "intermediate_size": self._ffn_dim,
            "vocab_size":        self._vocab_size,
            "head_dim":          self._d_head,
        }

    def extract_index(
        self,
        include_heads: bool = False,
        include_full: bool = True,
        include_ffn: bool = True,
    ) -> WeightIndex:
        """Extract a read-only WeightIndex of cloned weight tensors.

        Args:
            include_heads: Also add per-head slices for attn.{q,k,v}.
            include_full: Include full (unsliced) attention weights.
            include_ffn: Include FFN projections (fc1, fc2).

        Returns:
            WeightIndex with one WeightUnit per (layer, kind) combination.
        """
        self._check_loaded()
        units: list[WeightUnit] = []

        for layer_idx in range(self._n_layers):
            if include_full:
                for kind in ("attn.q", "attn.k", "attn.v", "attn.out"):
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.{kind}",
                        tensor=self.get_weight(layer_idx, kind).clone(),
                        kind=kind,
                        layer=layer_idx,
                    ))

            if include_ffn:
                for kind in ("ffn.up", "ffn.down"):
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.{kind}",
                        tensor=self.get_weight(layer_idx, kind).clone(),
                        kind=kind,
                        layer=layer_idx,
                    ))

            if include_heads:
                for kind in ("attn.q", "attn.k", "attn.v"):
                    for head_idx in range(self._n_heads):
                        units.append(WeightUnit(
                            key=f"layer{layer_idx}.{kind}.head{head_idx}",
                            tensor=self.get_weight(layer_idx, kind, head=head_idx).clone(),
                            kind=kind,
                            layer=layer_idx,
                            head=head_idx,
                        ))

        return WeightIndex(units)

    # ---- Internal helpers --------------------------------------------------

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Must call load() before using the extractor")

    def _layer(self, layer: int):
        """Return the decoder layer at index ``layer``."""
        self._check_loaded()
        return self._model.model.decoder.layers[layer]

    def _resolve(self, layer: int, kind: str) -> nn.Linear:
        """Navigate to the nn.Linear for (layer, kind).

        Args:
            layer: Layer index.
            kind: Canonical kind string.

        Returns:
            The live ``nn.Linear`` module.
        """
        if kind not in _KIND_PATH:
            raise KeyError(f"Unknown kind '{kind}' for OPTWeightExtractor")
        sub, attr = _KIND_PATH[kind]
        block = self._layer(layer)
        if sub is not None:
            return getattr(getattr(block, sub), attr)
        return getattr(block, attr)

    # ---- Write path --------------------------------------------------------

    def get_weight(
        self, layer: int, kind: str, *, head: int | None = None,
    ) -> torch.Tensor:
        """Read a live parameter slice (not cloned).

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            head: Head index for per-head access (attn.{q,k,v} only).

        Returns:
            View into the live parameter.
        """
        w = self._resolve(layer, kind).weight.data
        if head is None:
            return w
        if kind not in ("attn.q", "attn.k", "attn.v"):
            raise ValueError(f"Per-head access not supported for kind='{kind}'")
        if head < 0 or head >= self._n_heads:
            raise IndexError(
                f"head={head} out of range for n_head={self._n_heads}"
            )
        return w[head * self._d_head : (head + 1) * self._d_head, :]

    @torch.no_grad()
    def set_weight(
        self, layer: int, kind: str, value: torch.Tensor,
        *, head: int | None = None,
    ) -> None:
        """Write a tensor into a live parameter in-place.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            value: Tensor to write (auto-cast to param dtype/device).
            head: Head index for per-head writes.
        """
        param = self._resolve(layer, kind).weight
        val = value.to(dtype=param.dtype, device=param.device)
        if head is None:
            param.data.copy_(val)
        else:
            if kind not in ("attn.q", "attn.k", "attn.v"):
                raise ValueError(f"Per-head writes not supported for kind='{kind}'")
            if head < 0 or head >= self._n_heads:
                raise IndexError(
                    f"head={head} out of range for n_head={self._n_heads}"
                )
            param.data[head * self._d_head : (head + 1) * self._d_head, :] = val

    # ---- Module replacement ------------------------------------------------

    def get_module(self, layer: int, kind: str) -> nn.Module:
        """Return the live nn.Linear for (layer, kind).

        Args:
            layer: Layer index.
            kind: Canonical kind string.

        Returns:
            The ``nn.Linear`` module.
        """
        return self._resolve(layer, kind)

    def replace_module(self, layer: int, kind: str, new_module: nn.Module) -> None:
        """Swap the nn.Linear for (layer, kind) with a compressed variant.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            new_module: Replacement module.
        """
        if kind not in _KIND_PATH:
            raise KeyError(f"Unknown kind '{kind}' for OPTWeightExtractor")
        sub, attr = _KIND_PATH[kind]
        block = self._layer(layer)
        if sub is not None:
            setattr(getattr(block, sub), attr, new_module)
        else:
            setattr(block, attr, new_module)

    # ---- Named parameter access --------------------------------------------

    def _get_named_param(self, name: str) -> nn.Parameter:
        """Lookup a parameter by its fully qualified name."""
        for n, p in self._model.named_parameters():
            if n == name:
                return p
        raise KeyError(f"Parameter '{name}' not found in model")
