"""Weight extractor for LLaMA-family models (LLaMA 1/2/3, Mistral, Qwen2, Gemma-2, etc.).

Handles any ``transformers.LlamaForCausalLM``-compatible model — any model
whose decoder layers expose ``self_attn.{q,k,v,o}_proj`` and
``mlp.{gate,up,down}_proj`` as ``nn.Linear`` modules.  Unlike GPT-2, these
architectures use separate projections throughout — no fused QKV or Conv1D
gymnastics required.

Supports Grouped Query Attention (GQA): Q has ``n_heads`` heads, K/V have
``n_kv_heads`` heads (may differ).  Per-head slicing respects this layout.

Supports decoupled ``head_dim``: if the config provides an explicit
``head_dim`` (e.g. Gemma-2 where ``head_dim=256 != hidden_size // n_heads``),
it is used directly.  Otherwise falls back to ``hidden_size // n_heads``.

Canonical kind vocabulary (superset of GPT-2):
    attn.q, attn.k, attn.v, attn.out   — attention projections
    ffn.gate, ffn.up, ffn.down          — SwiGLU FFN projections
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightIndex, WeightUnit

# Maps canonical kind → (sub-module path, attribute name) within a decoder layer
_KIND_PATH: dict[str, tuple[str, str]] = {
    "attn.q":   ("self_attn", "q_proj"),
    "attn.k":   ("self_attn", "k_proj"),
    "attn.v":   ("self_attn", "v_proj"),
    "attn.out": ("self_attn", "o_proj"),
    "ffn.gate": ("mlp",       "gate_proj"),
    "ffn.up":   ("mlp",       "up_proj"),
    "ffn.down": ("mlp",       "down_proj"),
}

_ATTN_KINDS = frozenset({"attn.q", "attn.k", "attn.v", "attn.out"})
_FFN_KINDS  = frozenset({"ffn.gate", "ffn.up", "ffn.down"})
_KV_KINDS   = frozenset({"attn.k", "attn.v"})


def _is_hf_llama(model: Any) -> bool:
    """Accept any LlamaForCausalLM-compatible HF model by structural check."""
    try:
        layers = model.model.layers
        attn = layers[0].self_attn
        return (
            hasattr(attn, "q_proj")
            and hasattr(attn, "k_proj")
            and hasattr(layers[0], "mlp")
        )
    except (AttributeError, IndexError):
        return False


class LlamaWeightExtractor(WeightExtractor):
    """Weight extractor for LLaMA / LLaMA-2 / LLaMA-3 and compatible models.

    ``load()`` accepts a ``LlamaForCausalLM`` (or structurally equivalent)
    HuggingFace model.  No Foundation wrapper needed — the HF model is used
    directly.

    GQA note: ``n_kv_heads`` may be less than ``n_heads``.  K and V weight
    shapes are ``[n_kv_heads * head_dim, hidden_size]``, not
    ``[n_heads * head_dim, hidden_size]``.  Per-head ``head`` indices for
    ``attn.k`` / ``attn.v`` must be in ``[0, n_kv_heads)``.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._cfg: Any = None
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._n_kv_heads: int = 0
        self._head_dim: int = 0

    # ---- Read path ---------------------------------------------------------

    def load(self, model: Any, device: str = "cpu") -> "LlamaWeightExtractor":
        """Load from a HuggingFace LlamaForCausalLM (or compatible model).

        Args:
            model: ``LlamaForCausalLM`` instance (or any model with the same
                   ``model.layers[i].self_attn.{q,k,v,o}_proj`` layout).
            device: Ignored (model is already on its own device); kept for
                    API compatibility with the base class.

        Returns:
            self, for method chaining.
        """
        if not _is_hf_llama(model):
            raise ValueError(
                "Model must be a LlamaForCausalLM or structurally compatible "
                "(model.model.layers[i].self_attn.{q,k,v,o}_proj)"
            )
        self._model = model
        self._cfg = model.config
        self._n_layers = self._cfg.num_hidden_layers
        self._n_heads = self._cfg.num_attention_heads
        self._n_kv_heads = getattr(self._cfg, "num_key_value_heads", self._n_heads)
        self._head_dim = getattr(self._cfg, "head_dim", self._cfg.hidden_size // self._n_heads)
        return self

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration dict.

        Returns:
            Dict with n_layers, n_head, n_kv_head, n_embd, intermediate_size,
            vocab_size, head_dim.
        """
        self._check_loaded()
        return {
            "n_layers":         self._n_layers,
            "n_head":           self._n_heads,
            "n_kv_head":        self._n_kv_heads,
            "n_embd":           self._cfg.hidden_size,
            "intermediate_size": self._cfg.intermediate_size,
            "vocab_size":       self._cfg.vocab_size,
            "head_dim":         self._head_dim,
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
            include_ffn: Include FFN projections (gate, up, down).

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
                for kind in ("ffn.gate", "ffn.up", "ffn.down"):
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.{kind}",
                        tensor=self.get_weight(layer_idx, kind).clone(),
                        kind=kind,
                        layer=layer_idx,
                    ))

            if include_heads:
                # Q: n_heads heads
                for head_idx in range(self._n_heads):
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.attn.q.head{head_idx}",
                        tensor=self.get_weight(layer_idx, "attn.q", head=head_idx).clone(),
                        kind="attn.q",
                        layer=layer_idx,
                        head=head_idx,
                    ))
                # K, V: n_kv_heads heads (may differ from n_heads under GQA)
                for kind in ("attn.k", "attn.v"):
                    for head_idx in range(self._n_kv_heads):
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
        return self._model.model.layers[layer]

    def _resolve(self, layer: int, kind: str) -> nn.Linear:
        """Navigate to the nn.Linear for (layer, kind).

        Args:
            layer: Layer index.
            kind: Canonical kind string.

        Returns:
            The live ``nn.Linear`` module.
        """
        if kind not in _KIND_PATH:
            raise KeyError(f"Unknown kind '{kind}' for LlamaWeightExtractor")
        sub, attr = _KIND_PATH[kind]
        return getattr(getattr(self._layer(layer), sub), attr)

    def _kv_head_count(self, kind: str) -> int:
        """Number of heads for this kind (GQA: K/V have n_kv_heads, Q has n_heads)."""
        return self._n_kv_heads if kind in _KV_KINDS else self._n_heads

    # ---- Write path --------------------------------------------------------

    def get_weight(
        self, layer: int, kind: str, *, head: int | None = None,
    ) -> torch.Tensor:
        """Read a live parameter slice (not cloned).

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            head: Head index for per-head access.  For ``attn.k`` / ``attn.v``
                  valid range is ``[0, n_kv_heads)``.

        Returns:
            View into the live parameter (shape ``[out, in]`` for full,
            ``[head_dim, hidden_size]`` for per-head).
        """
        w = self._resolve(layer, kind).weight.data
        if head is None:
            return w
        n_heads = self._kv_head_count(kind)
        if head < 0 or head >= n_heads:
            raise IndexError(
                f"head={head} out of range for kind='{kind}' (n_heads={n_heads})"
            )
        return w[head * self._head_dim : (head + 1) * self._head_dim, :]

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
            n_heads = self._kv_head_count(kind)
            if head < 0 or head >= n_heads:
                raise IndexError(
                    f"head={head} out of range for kind='{kind}' (n_heads={n_heads})"
                )
            row_start = head * self._head_dim
            param.data[row_start : row_start + self._head_dim, :] = val

    # ---- Module replacement ------------------------------------------------

    def get_module(self, layer: int, kind: str) -> nn.Linear:
        """Return the live nn.Linear for (layer, kind).

        Args:
            layer: Layer index.
            kind: Canonical kind string.

        Returns:
            The ``nn.Linear`` module (standard PyTorch, weight shape ``[out, in]``).
        """
        return self._resolve(layer, kind)

    def replace_module(self, layer: int, kind: str, new_module: nn.Module) -> None:
        """Swap the nn.Linear for (layer, kind) with a compressed variant.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            new_module: Replacement module (e.g. ShareLinear, DeltaLinear).
        """
        if kind not in _KIND_PATH:
            raise KeyError(f"Unknown kind '{kind}' for LlamaWeightExtractor")
        sub, attr = _KIND_PATH[kind]
        setattr(getattr(self._layer(layer), sub), attr, new_module)

    # ---- Named parameter access --------------------------------------------

    def _get_named_param(self, name: str) -> nn.Parameter:
        """Lookup a parameter by its fully qualified name.

        Args:
            name: Parameter name as returned by ``model.named_parameters()``
                  (e.g. ``"model.layers.0.self_attn.q_proj.weight"``).

        Returns:
            The live ``nn.Parameter``.
        """
        for n, p in self._model.named_parameters():
            if n == name:
                return p
        raise KeyError(f"Parameter '{name}' not found in model")
