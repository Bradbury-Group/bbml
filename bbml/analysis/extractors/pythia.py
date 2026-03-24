"""Weight extractor for Pythia / GPT-NeoX models (EleutherAI).

Handles ``GPTNeoXForCausalLM``-compatible models whose decoder layers expose
``attention.query_key_value`` (fused Q+K+V) and ``attention.dense`` (output
projection), plus ``mlp.{dense_h_to_4h, dense_4h_to_h}``.

Unlike LLaMA-family models, GPT-NeoX uses:
  - Fused ``query_key_value`` projection: ``nn.Linear(hidden, 3*hidden)``
  - Simple GELU MLP (no SwiGLU gating): ``dense_h_to_4h`` + ``dense_4h_to_h``
  - LayerNorm (not RMSNorm)
  - Parallel residual connections (attn + MLP run in parallel)

Canonical kind vocabulary (subset — no ``ffn.gate``):
    attn.q, attn.k, attn.v, attn.out   — attention projections
    ffn.up, ffn.down                    — MLP projections
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightIndex, WeightUnit

_PROJ_INDEX = {"q": 0, "k": 1, "v": 2}
_PROJ_NAMES = ("q", "k", "v")


def _is_hf_neox(model: Any) -> bool:
    """Accept any GPTNeoXForCausalLM-compatible model by structural check."""
    try:
        layers = model.gpt_neox.layers
        attn = layers[0].attention
        return (
            hasattr(attn, "query_key_value")
            and hasattr(attn, "dense")
            and hasattr(layers[0], "mlp")
        )
    except (AttributeError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Drop-in replacement for fused query_key_value
# ---------------------------------------------------------------------------

class _DefusedQKV(nn.Module):
    """Three independent projections presenting a fused interface.

    ``forward`` concatenates outputs along dim=-1 so the original
    ``GPTNeoXAttention.forward`` sees no difference.
    """

    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module) -> None:
        super().__init__()
        self.q_proj = q
        self.k_proj = k
        self.v_proj = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)


def _split_fused_qkv(fused: nn.Linear, n_embd: int) -> _DefusedQKV:
    """Split fused query_key_value [3*n_embd, n_embd] into three Linears."""
    has_bias = fused.bias is not None
    projs = []
    for i in range(3):
        lin = nn.Linear(
            n_embd, n_embd, bias=has_bias,
            device=fused.weight.device, dtype=fused.weight.dtype,
        )
        lin.weight.data.copy_(fused.weight.data[i * n_embd : (i + 1) * n_embd])
        if has_bias:
            lin.bias.data.copy_(fused.bias.data[i * n_embd : (i + 1) * n_embd])
        projs.append(lin)
    return _DefusedQKV(*projs)


class PythiaWeightExtractor(WeightExtractor):
    """Weight extractor for Pythia / GPT-NeoX (fused QKV, simple MLP).

    ``load()`` accepts a ``GPTNeoXForCausalLM`` (or structurally equivalent)
    HuggingFace model.  MHA only — all heads share the same count.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._n_embd: int = 0
        self._d_head: int = 0
        self._intermediate: int = 0
        self._vocab_size: int = 0

    # ---- Read path ---------------------------------------------------------

    def load(self, model: Any, device: str = "cpu") -> "PythiaWeightExtractor":
        """Load from a HuggingFace GPTNeoXForCausalLM (or compatible).

        Args:
            model: ``GPTNeoXForCausalLM`` instance.
            device: Ignored (kept for API compatibility).

        Returns:
            self, for method chaining.
        """
        if not _is_hf_neox(model):
            raise ValueError(
                "Model must be a GPTNeoXForCausalLM or structurally compatible "
                "(model.gpt_neox.layers[i].attention.query_key_value)"
            )
        self._model = model
        cfg = model.config
        self._n_layers = cfg.num_hidden_layers
        self._n_heads = cfg.num_attention_heads
        self._n_embd = cfg.hidden_size
        self._d_head = self._n_embd // self._n_heads
        self._intermediate = cfg.intermediate_size
        self._vocab_size = cfg.vocab_size
        return self

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration dict.

        Returns:
            Dict with n_layers, n_head, n_embd, intermediate_size,
            vocab_size, head_dim.
        """
        self._check_loaded()
        return {
            "n_layers":          self._n_layers,
            "n_head":            self._n_heads,
            "n_kv_head":         self._n_heads,  # MHA: same as n_head
            "n_embd":            self._n_embd,
            "intermediate_size": self._intermediate,
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
            include_ffn: Include FFN projections (up, down).

        Returns:
            WeightIndex with one WeightUnit per (layer, kind) combination.
        """
        self._check_loaded()
        units: list[WeightUnit] = []

        for layer_idx in range(self._n_layers):
            if include_full:
                for proj in _PROJ_NAMES:
                    kind = f"attn.{proj}"
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.{kind}",
                        tensor=self.get_weight(layer_idx, kind).clone(),
                        kind=kind,
                        layer=layer_idx,
                    ))
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.attn.out",
                    tensor=self.get_weight(layer_idx, "attn.out").clone(),
                    kind="attn.out",
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
                for proj in _PROJ_NAMES:
                    kind = f"attn.{proj}"
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
        return self._model.gpt_neox.layers[layer]

    def _qkv_module(self, layer: int):
        """Return the query_key_value module (fused or defused)."""
        return self._layer(layer).attention.query_key_value

    def _proj_slice(self, proj: str) -> slice:
        """Row slice into fused weight [3*n_embd, n_embd] for a projection."""
        idx = _PROJ_INDEX[proj]
        n = self._n_embd
        return slice(idx * n, (idx + 1) * n)

    def _head_slice(self, proj: str, head: int) -> slice:
        """Row slice for a single head within fused weight."""
        if head < 0 or head >= self._n_heads:
            raise IndexError(f"head={head} out of range for n_head={self._n_heads}")
        idx = _PROJ_INDEX[proj]
        row_start = idx * self._n_embd + head * self._d_head
        return slice(row_start, row_start + self._d_head)

    # ---- Write path --------------------------------------------------------

    def get_weight(
        self, layer: int, kind: str, *, head: int | None = None,
    ) -> torch.Tensor:
        """Read a live parameter slice (not cloned).

        Args:
            layer: Layer index.
            kind: ``"attn.q"``, ``"attn.k"``, ``"attn.v"``,
                  ``"attn.out"``, ``"ffn.up"``, ``"ffn.down"``.
            head: Head index (only valid for attn.{q,k,v}).

        Returns:
            View into the live parameter.
        """
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = self._qkv_module(layer)

            if isinstance(qkv, _DefusedQKV):
                w = getattr(qkv, f"{proj}_proj").weight.data
                if head is not None:
                    if head < 0 or head >= self._n_heads:
                        raise IndexError(
                            f"head={head} out of range for n_head={self._n_heads}"
                        )
                    return w[head * self._d_head : (head + 1) * self._d_head, :]
                return w

            # Fused layout [3*n_embd, n_embd]
            w = qkv.weight.data
            if head is not None:
                return w[self._head_slice(proj, head), :]
            return w[self._proj_slice(proj), :]

        block = self._layer(layer)
        if kind == "attn.out":
            return block.attention.dense.weight.data
        if kind == "ffn.up":
            return block.mlp.dense_h_to_4h.weight.data
        if kind == "ffn.down":
            return block.mlp.dense_4h_to_h.weight.data

        raise KeyError(f"Unknown kind '{kind}' for PythiaWeightExtractor")

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
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = self._qkv_module(layer)

            if isinstance(qkv, _DefusedQKV):
                w = getattr(qkv, f"{proj}_proj").weight
                val = value.to(dtype=w.dtype, device=w.device)
                if head is not None:
                    if head < 0 or head >= self._n_heads:
                        raise IndexError(
                            f"head={head} out of range for n_head={self._n_heads}"
                        )
                    w.data[head * self._d_head : (head + 1) * self._d_head, :] = val
                else:
                    w.data.copy_(val)
                return

            # Fused layout
            param = qkv.weight
            val = value.to(dtype=param.dtype, device=param.device)
            if head is not None:
                param.data[self._head_slice(proj, head), :] = val
            else:
                param.data[self._proj_slice(proj), :] = val
            return

        block = self._layer(layer)
        if kind == "attn.out":
            w = block.attention.dense.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return
        if kind == "ffn.up":
            w = block.mlp.dense_h_to_4h.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return
        if kind == "ffn.down":
            w = block.mlp.dense_4h_to_h.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return

        raise KeyError(f"Unknown kind '{kind}' for PythiaWeightExtractor")

    # ---- Module replacement ------------------------------------------------

    def get_module(self, layer: int, kind: str) -> nn.Module:
        """Return the nn.Module owning the weight for (layer, kind).

        For attn.{q,k,v}: defuses fused query_key_value on first access.
        """
        block = self._layer(layer)
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = block.attention.query_key_value
            if isinstance(qkv, _DefusedQKV):
                return getattr(qkv, f"{proj}_proj")
            defused = _split_fused_qkv(qkv, self._n_embd)
            block.attention.query_key_value = defused
            return getattr(defused, f"{proj}_proj")
        if kind == "attn.out":
            return block.attention.dense
        if kind == "ffn.up":
            return block.mlp.dense_h_to_4h
        if kind == "ffn.down":
            return block.mlp.dense_4h_to_h
        raise KeyError(f"Unknown kind '{kind}' for PythiaWeightExtractor")

    def replace_module(self, layer: int, kind: str, new_module: nn.Module) -> None:
        """Swap an nn.Module with a compressed variant.

        For attn.{q,k,v}: defuses the fused query_key_value into a
        ``_DefusedQKV`` wrapper (if not already defused) and replaces
        only the target projection.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            new_module: Replacement (e.g. ShareLinear, DeltaLinear).
        """
        block = self._layer(layer)
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = block.attention.query_key_value
            if not isinstance(qkv, _DefusedQKV):
                qkv = _split_fused_qkv(qkv, self._n_embd)
                block.attention.query_key_value = qkv
            setattr(qkv, f"{proj}_proj", new_module)
            return
        if kind == "attn.out":
            block.attention.dense = new_module
            return
        if kind == "ffn.up":
            block.mlp.dense_h_to_4h = new_module
            return
        if kind == "ffn.down":
            block.mlp.dense_4h_to_h = new_module
            return
        raise KeyError(f"Unknown kind '{kind}' for PythiaWeightExtractor")

    # ---- Named parameter access --------------------------------------------

    def _get_named_param(self, name: str) -> nn.Parameter:
        """Lookup a parameter by its fully qualified name."""
        for n, p in self._model.named_parameters():
            if n == name:
                return p
        raise KeyError(f"Parameter '{name}' not found in model")
