"""Weight extractor for Phi-3 models (Microsoft).

Handles ``Phi3ForCausalLM``-compatible models whose decoder layers expose
``self_attn.qkv_proj`` (fused Q+K+V) and ``self_attn.o_proj`` (output),
plus ``mlp.gate_up_proj`` (fused gate+up) and ``mlp.down_proj``.

Unlike LLaMA (separate projections) or GPT-2 (fused QKV but separate MLP),
Phi-3 fuses *both* attention QKV and MLP gate+up:
  - ``qkv_proj``: ``nn.Linear(hidden, n_heads*hd + 2*n_kv_heads*hd)``
  - ``gate_up_proj``: ``nn.Linear(hidden, 2*intermediate)``

For attention, defusion splits Q/K/V into independent modules for
per-projection compression.  For MLP, ``get_weight`` slices into the fused
tensor (read path works), but ``replace_module`` raises for the fused MLP
kinds since FFN basis sharing is not viable.

Canonical kind vocabulary:
    attn.q, attn.k, attn.v, attn.out   — attention projections
    ffn.up, ffn.down                    — MLP projections (ffn.up is gate+up fused)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightIndex, WeightUnit

_PROJ_INDEX = {"q": 0, "k": 1, "v": 2}
_PROJ_NAMES = ("q", "k", "v")


def _is_hf_phi3(model: Any) -> bool:
    """Accept any Phi3ForCausalLM-compatible model by structural check."""
    try:
        layers = model.model.layers
        attn = layers[0].self_attn
        return (
            hasattr(attn, "qkv_proj")
            and hasattr(attn, "o_proj")
            and hasattr(layers[0], "mlp")
            and hasattr(layers[0].mlp, "gate_up_proj")
        )
    except (AttributeError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Drop-in replacement for fused qkv_proj
# ---------------------------------------------------------------------------

class _DefusedQKV(nn.Module):
    """Three independent projections presenting a fused ``qkv_proj`` interface.

    Handles both MHA (Q/K/V same size) and GQA (K/V smaller than Q).
    ``forward`` concatenates [Q, K, V] along dim=-1.
    """

    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module) -> None:
        super().__init__()
        self.q_proj = q
        self.k_proj = k
        self.v_proj = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)


def _split_fused_qkv(
    fused: nn.Linear, n_embd: int,
    n_heads: int, n_kv_heads: int, head_dim: int,
) -> _DefusedQKV:
    """Split fused qkv_proj into three independent Linears.

    Fused weight layout (rows):
      [0 : n_heads*hd]                         → Q
      [n_heads*hd : n_heads*hd + n_kv_heads*hd] → K
      [n_heads*hd + n_kv_heads*hd : ]           → V
    """
    has_bias = fused.bias is not None
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    dims = [q_dim, kv_dim, kv_dim]
    offset = 0
    projs = []
    for dim in dims:
        lin = nn.Linear(
            n_embd, dim, bias=has_bias,
            device=fused.weight.device, dtype=fused.weight.dtype,
        )
        lin.weight.data.copy_(fused.weight.data[offset : offset + dim])
        if has_bias:
            lin.bias.data.copy_(fused.bias.data[offset : offset + dim])
        projs.append(lin)
        offset += dim
    return _DefusedQKV(*projs)


class Phi3WeightExtractor(WeightExtractor):
    """Weight extractor for Phi-3 (fused QKV + fused gate_up MLP).

    ``load()`` accepts a ``Phi3ForCausalLM`` (or structurally equivalent)
    HuggingFace model.  Phi-3-mini is MHA (32/32) but the extractor
    handles GQA variants if they exist.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._n_layers: int = 0
        self._n_heads: int = 0
        self._n_kv_heads: int = 0
        self._n_embd: int = 0
        self._head_dim: int = 0
        self._intermediate: int = 0
        self._vocab_size: int = 0

    # ---- Read path ---------------------------------------------------------

    def load(self, model: Any, device: str = "cpu") -> "Phi3WeightExtractor":
        """Load from a HuggingFace Phi3ForCausalLM (or compatible).

        Args:
            model: ``Phi3ForCausalLM`` instance.
            device: Ignored (kept for API compatibility).

        Returns:
            self, for method chaining.
        """
        if not _is_hf_phi3(model):
            raise ValueError(
                "Model must be a Phi3ForCausalLM or structurally compatible "
                "(model.model.layers[i].self_attn.qkv_proj)"
            )
        self._model = model
        cfg = model.config
        self._n_layers = cfg.num_hidden_layers
        self._n_heads = cfg.num_attention_heads
        self._n_kv_heads = getattr(cfg, "num_key_value_heads", self._n_heads)
        self._n_embd = cfg.hidden_size
        self._head_dim = getattr(cfg, "head_dim", self._n_embd // self._n_heads)
        self._intermediate = cfg.intermediate_size
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
            "n_kv_head":         self._n_kv_heads,
            "n_embd":            self._n_embd,
            "intermediate_size": self._intermediate,
            "vocab_size":        self._vocab_size,
            "head_dim":          self._head_dim,
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
            include_ffn: Include FFN projections.

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
                # Q heads
                for head_idx in range(self._n_heads):
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.attn.q.head{head_idx}",
                        tensor=self.get_weight(layer_idx, "attn.q", head=head_idx).clone(),
                        kind="attn.q",
                        layer=layer_idx,
                        head=head_idx,
                    ))
                # K/V heads (may differ from Q under GQA)
                for proj in ("k", "v"):
                    kind = f"attn.{proj}"
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

    def _qkv_module(self, layer: int):
        """Return the qkv_proj module (fused or defused)."""
        return self._layer(layer).self_attn.qkv_proj

    def _qkv_offsets(self) -> tuple[int, int, int]:
        """Row offsets for Q, K, V within fused qkv_proj weight."""
        q_dim = self._n_heads * self._head_dim
        kv_dim = self._n_kv_heads * self._head_dim
        return 0, q_dim, q_dim + kv_dim

    def _kv_head_count(self, kind: str) -> int:
        """Number of heads for this kind (GQA-aware)."""
        if kind in ("attn.k", "attn.v"):
            return self._n_kv_heads
        return self._n_heads

    # ---- Write path --------------------------------------------------------

    def get_weight(
        self, layer: int, kind: str, *, head: int | None = None,
    ) -> torch.Tensor:
        """Read a live parameter slice (not cloned).

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            head: Head index for per-head access.

        Returns:
            View into the live parameter.
        """
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = self._qkv_module(layer)

            if isinstance(qkv, _DefusedQKV):
                w = getattr(qkv, f"{proj}_proj").weight.data
                if head is not None:
                    n_h = self._kv_head_count(kind)
                    if head < 0 or head >= n_h:
                        raise IndexError(
                            f"head={head} out of range for kind='{kind}' (n_heads={n_h})"
                        )
                    return w[head * self._head_dim : (head + 1) * self._head_dim, :]
                return w

            # Fused layout: [Q_rows | K_rows | V_rows]
            w = qkv.weight.data
            q_off, k_off, v_off = self._qkv_offsets()
            offsets = {"q": q_off, "k": k_off, "v": v_off}
            dims = {
                "q": self._n_heads * self._head_dim,
                "k": self._n_kv_heads * self._head_dim,
                "v": self._n_kv_heads * self._head_dim,
            }
            start = offsets[proj]
            dim = dims[proj]

            if head is not None:
                n_h = self._kv_head_count(kind)
                if head < 0 or head >= n_h:
                    raise IndexError(
                        f"head={head} out of range for kind='{kind}' (n_heads={n_h})"
                    )
                row = start + head * self._head_dim
                return w[row : row + self._head_dim, :]
            return w[start : start + dim, :]

        block = self._layer(layer)
        if kind == "attn.out":
            return block.self_attn.o_proj.weight.data
        if kind == "ffn.up":
            # gate_up_proj fused: [2*intermediate, hidden]
            # First half is gate, second half is up — return the full fused tensor
            return block.mlp.gate_up_proj.weight.data
        if kind == "ffn.down":
            return block.mlp.down_proj.weight.data

        raise KeyError(f"Unknown kind '{kind}' for Phi3WeightExtractor")

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
                    n_h = self._kv_head_count(kind)
                    if head < 0 or head >= n_h:
                        raise IndexError(
                            f"head={head} out of range for kind='{kind}' (n_heads={n_h})"
                        )
                    w.data[head * self._head_dim : (head + 1) * self._head_dim, :] = val
                else:
                    w.data.copy_(val)
                return

            # Fused layout
            param = qkv.weight
            val = value.to(dtype=param.dtype, device=param.device)
            q_off, k_off, v_off = self._qkv_offsets()
            offsets = {"q": q_off, "k": k_off, "v": v_off}
            dims = {
                "q": self._n_heads * self._head_dim,
                "k": self._n_kv_heads * self._head_dim,
                "v": self._n_kv_heads * self._head_dim,
            }
            start = offsets[proj]
            dim = dims[proj]

            if head is not None:
                n_h = self._kv_head_count(kind)
                if head < 0 or head >= n_h:
                    raise IndexError(
                        f"head={head} out of range for kind='{kind}' (n_heads={n_h})"
                    )
                row = start + head * self._head_dim
                param.data[row : row + self._head_dim, :] = val
            else:
                param.data[start : start + dim, :] = val
            return

        block = self._layer(layer)
        if kind == "attn.out":
            w = block.self_attn.o_proj.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return
        if kind == "ffn.up":
            w = block.mlp.gate_up_proj.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return
        if kind == "ffn.down":
            w = block.mlp.down_proj.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return

        raise KeyError(f"Unknown kind '{kind}' for Phi3WeightExtractor")

    # ---- Module replacement ------------------------------------------------

    def get_module(self, layer: int, kind: str) -> nn.Module:
        """Return the nn.Module owning the weight for (layer, kind).

        For attn.{q,k,v}: defuses fused qkv_proj on first access.
        """
        block = self._layer(layer)
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = block.self_attn.qkv_proj
            if isinstance(qkv, _DefusedQKV):
                return getattr(qkv, f"{proj}_proj")
            defused = _split_fused_qkv(
                qkv, self._n_embd,
                self._n_heads, self._n_kv_heads, self._head_dim,
            )
            block.self_attn.qkv_proj = defused
            return getattr(defused, f"{proj}_proj")
        if kind == "attn.out":
            return block.self_attn.o_proj
        if kind == "ffn.up":
            return block.mlp.gate_up_proj
        if kind == "ffn.down":
            return block.mlp.down_proj
        raise KeyError(f"Unknown kind '{kind}' for Phi3WeightExtractor")

    def replace_module(self, layer: int, kind: str, new_module: nn.Module) -> None:
        """Swap an nn.Module with a compressed variant.

        For attn.{q,k,v}: defuses fused qkv_proj and replaces only the
        target projection.  For ffn.up (fused gate_up_proj): raises
        NotImplementedError — MLP defusion is not implemented.

        Args:
            layer: Layer index.
            kind: Canonical kind string.
            new_module: Replacement module.
        """
        block = self._layer(layer)
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            qkv = block.self_attn.qkv_proj
            if not isinstance(qkv, _DefusedQKV):
                qkv = _split_fused_qkv(
                    qkv, self._n_embd,
                    self._n_heads, self._n_kv_heads, self._head_dim,
                )
                block.self_attn.qkv_proj = qkv
            setattr(qkv, f"{proj}_proj", new_module)
            return
        if kind == "attn.out":
            block.self_attn.o_proj = new_module
            return
        if kind == "ffn.up":
            raise NotImplementedError(
                "replace_module for ffn.up (fused gate_up_proj) is not supported. "
                "Phi-3 fuses gate and up projections — defusion not implemented."
            )
        if kind == "ffn.down":
            block.mlp.down_proj = new_module
            return
        raise KeyError(f"Unknown kind '{kind}' for Phi3WeightExtractor")

    # ---- Named parameter access --------------------------------------------

    def _get_named_param(self, name: str) -> nn.Parameter:
        """Lookup a parameter by its fully qualified name."""
        for n, p in self._model.named_parameters():
            if n == name:
                return p
        raise KeyError(f"Parameter '{name}' not found in model")
