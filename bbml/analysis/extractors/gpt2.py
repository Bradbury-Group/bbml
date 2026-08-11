from typing import Any, Dict

import torch
from torch import nn

from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightIndex, WeightUnit
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation

_PROJ_INDEX = {"q": 0, "k": 1, "v": 2}
_PROJ_NAMES = ("q", "k", "v")


def _is_hf_gpt2(model: Any) -> bool:
    """Check if model is a HuggingFace GPT2LMHeadModel."""
    cls_name = type(model).__name__
    return cls_name in ("GPT2LMHeadModel", "GPT2Model")


def _convert_conv1d_to_linear(model: nn.Module) -> None:
    """Convert all Conv1D modules to nn.Linear in-place.

    HuggingFace GPT-2 uses Conv1D with weight shape [in, out].
    nanoGPT/bbml uses nn.Linear with weight shape [out, in].
    This transposes weights and swaps the modules so the extractor
    works uniformly with nn.Linear everywhere.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        return

    for name, module in list(model.named_modules()):
        if isinstance(module, Conv1D):
            # Conv1D stores weight as [in_features, out_features]
            in_f, out_f = module.weight.shape
            linear = nn.Linear(
                in_f, out_f,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            linear.weight.data.copy_(module.weight.data.T)
            if module.bias is not None:
                linear.bias.data.copy_(module.bias.data)

            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], linear)


# ---------------------------------------------------------------------------
# Drop-in replacement for fused c_attn that holds Q, K, V independently
# ---------------------------------------------------------------------------

class _DefusedQKV(nn.Module):
    """Three independent projections presenting a fused ``c_attn`` interface.

    Created automatically by ``replace_module`` when replacing an individual
    Q, K, or V projection.  ``forward`` concatenates outputs along dim=-1
    so ``CausalSelfAttention.forward`` sees no difference.
    """

    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module) -> None:
        super().__init__()
        self.q_proj = q
        self.k_proj = k
        self.v_proj = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)


def _split_fused_linear(fused: nn.Linear, n_embd: int) -> _DefusedQKV:
    """Split a fused ``c_attn`` Linear [3*n_embd, n_embd] into three Linears."""
    has_bias = fused.bias is not None
    projs = []
    for i in range(3):
        lin = nn.Linear(n_embd, n_embd, bias=has_bias, device=fused.weight.device,
                        dtype=fused.weight.dtype)
        # Fused weight is [3*n_embd, n_embd]; rows i*n..(i+1)*n are Q/K/V
        lin.weight.data.copy_(fused.weight.data[i * n_embd : (i + 1) * n_embd])
        if has_bias:
            lin.bias.data.copy_(fused.bias.data[i * n_embd : (i + 1) * n_embd])
        projs.append(lin)
    return _DefusedQKV(*projs)


class GPT2WeightExtractor(WeightExtractor):
    """Weight extractor for nanoGPT-style GPT-2 (nn.Linear, not Conv1D).

    ``c_attn`` is ``nn.Linear(n_embd, 3*n_embd)`` → weight shape
    ``[3*n_embd, n_embd]``.  Rows ``[0:n, :]`` = Q, ``[n:2n, :]`` = K,
    ``[2n:3n, :]`` = V.  Per-head slicing within each projection block
    uses ``d_head``-wide row spans.
    """

    def __init__(self):
        self.foundation = None
        self.config = None

    # ---- Read path ---------------------------------------------------------

    def load(self, model: Any, device: str = "cpu") -> "GPT2WeightExtractor":
        if isinstance(model, GPT2Foundation):
            self.foundation = model
            self.config = model.config
        elif _is_hf_gpt2(model):
            # Wrap HF model: convert Conv1D→Linear, extract config
            _convert_conv1d_to_linear(model)
            # Create a lightweight shell so downstream code sees .model and .config
            shell = object.__new__(GPT2Foundation)
            object.__setattr__(shell, "model", model)
            object.__setattr__(shell, "config", model.config)
            object.__setattr__(shell, "device", device)
            object.__setattr__(shell, "dtype", next(model.parameters()).dtype)
            self.foundation = shell
            self.config = model.config
        else:
            raise ValueError(
                "Model must be a GPT2Foundation or HuggingFace GPT2LMHeadModel"
            )
        return self

    def get_config(self) -> Dict[str, Any]:
        if self.foundation is None or self.config is None:
            raise RuntimeError("Must call load() before get_config()")
        return {
            "n_layers": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_embd": self.config.n_embd,
            "vocab_size": self.config.vocab_size,
        }

    def extract_index(
        self,
        include_heads: bool = False,
        include_full: bool = True,
        include_ffn: bool = True,
    ) -> WeightIndex:
        if self.foundation is None:
            raise RuntimeError("Must call load() before extract_index()")

        units = []
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        n_embd = self.config.n_embd
        d_head = n_embd // n_head

        for layer_idx in range(n_layer):
            block = self.foundation.model.transformer.h[layer_idx]
            c_attn = block.attn.c_attn

            # Handle both fused and defused layouts
            if isinstance(c_attn, _DefusedQKV):
                q_weight = c_attn.q_proj.weight.data
                k_weight = c_attn.k_proj.weight.data
                v_weight = c_attn.v_proj.weight.data
            else:
                # Fused weight [3*n_embd, n_embd]: row blocks are Q, K, V
                w = c_attn.weight
                qkv = w.view(3, n_embd, n_embd)
                q_weight = qkv[0]
                k_weight = qkv[1]
                v_weight = qkv[2]

            if include_full:
                for name, tensor in [
                    ("q", q_weight), ("k", k_weight), ("v", v_weight),
                ]:
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.attn.{name}",
                        tensor=tensor.clone(),
                        kind=f"attn.{name}",
                        layer=layer_idx,
                    ))

            if include_heads:
                for name, w_full in [
                    ("q", q_weight), ("k", k_weight), ("v", v_weight),
                ]:
                    # Each head: d_head contiguous rows within [n_embd, n_embd]
                    for head_idx in range(n_head):
                        row_start = head_idx * d_head
                        units.append(WeightUnit(
                            key=f"layer{layer_idx}.attn.{name}.head{head_idx}",
                            tensor=w_full[row_start : row_start + d_head, :].clone(),
                            kind=f"attn.{name}",
                            layer=layer_idx,
                            head=head_idx,
                        ))

            if include_ffn:
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.ffn.up",
                    tensor=block.mlp.c_fc.weight.clone(),
                    kind="ffn.up",
                    layer=layer_idx,
                ))
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.ffn.down",
                    tensor=block.mlp.c_proj.weight.clone(),
                    kind="ffn.down",
                    layer=layer_idx,
                ))

        return WeightIndex(units)

    # ---- Internal helpers --------------------------------------------------

    def _block(self, layer: int):
        self._check_loaded()
        return self.foundation.model.transformer.h[layer]

    def _check_loaded(self):
        if self.foundation is None:
            raise RuntimeError("Must call load() first")

    def _attn_param(self, layer: int) -> nn.Parameter:
        """Return the fused c_attn weight. Raises if already defused."""
        c_attn = self._block(layer).attn.c_attn
        if isinstance(c_attn, _DefusedQKV):
            raise RuntimeError(
                "c_attn has been defused — use get_weight/set_weight with kind"
            )
        return c_attn.weight

    def _proj_slice(self, proj: str) -> tuple[slice, slice]:
        """Row/col slices into fused weight [3*n_embd, n_embd] for a projection."""
        idx = _PROJ_INDEX[proj.lower()]
        n = self.config.n_embd
        return slice(idx * n, (idx + 1) * n), slice(None)

    def _head_slice(self, proj: str, head: int) -> tuple[slice, slice]:
        """Row/col slices for a single head within fused weight [3*n_embd, n_embd]."""
        n_head = self.config.n_head
        if head < 0 or head >= n_head:
            raise IndexError(f"head={head} out of range for n_head={n_head}")
        idx = _PROJ_INDEX[proj.lower()]
        n = self.config.n_embd
        d = n // n_head
        row_start = idx * n + head * d
        return slice(row_start, row_start + d), slice(None)

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
            c_attn = self._block(layer).attn.c_attn

            if isinstance(c_attn, _DefusedQKV):
                w = getattr(c_attn, f"{proj}_proj").weight.data
                if head is not None:
                    n_head = self.config.n_head
                    if head < 0 or head >= n_head:
                        raise IndexError(f"head={head} out of range for n_head={n_head}")
                    d = self.config.n_embd // n_head
                    return w[head * d : (head + 1) * d, :]
                return w

            # Fused layout [3*n_embd, n_embd]
            param = c_attn.weight
            if head is not None:
                r, c = self._head_slice(proj, head)
            else:
                r, c = self._proj_slice(proj)
            return param.data[r, c]

        block = self._block(layer)
        if kind == "ffn.up":
            return block.mlp.c_fc.weight.data
        if kind == "ffn.down":
            return block.mlp.c_proj.weight.data
        if kind == "attn.out":
            return block.attn.c_proj.weight.data

        raise KeyError(f"Unknown kind '{kind}' for GPT2WeightExtractor")

    @torch.no_grad()
    def set_weight(
        self, layer: int, kind: str, value: torch.Tensor,
        *, head: int | None = None,
    ) -> None:
        """Write a tensor into a live parameter in-place."""
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            c_attn = self._block(layer).attn.c_attn

            if isinstance(c_attn, _DefusedQKV):
                w = getattr(c_attn, f"{proj}_proj").weight
                val = value.to(dtype=w.dtype, device=w.device)
                if head is not None:
                    n_head = self.config.n_head
                    if head < 0 or head >= n_head:
                        raise IndexError(f"head={head} out of range for n_head={n_head}")
                    d = self.config.n_embd // n_head
                    w.data[head * d : (head + 1) * d, :] = val
                else:
                    w.data.copy_(val)
                return

            # Fused layout [3*n_embd, n_embd]
            param = c_attn.weight
            if head is not None:
                r, c = self._head_slice(proj, head)
            else:
                r, c = self._proj_slice(proj)
            param.data[r, c] = value.to(dtype=param.dtype, device=param.device)
            return

        # Non-attn: explicit per-kind writes
        block = self._block(layer)
        if kind == "ffn.up":
            w = block.mlp.c_fc.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return
        if kind == "ffn.down":
            w = block.mlp.c_proj.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return
        if kind == "attn.out":
            w = block.attn.c_proj.weight
            w.data.copy_(value.to(dtype=w.dtype, device=w.device))
            return

        raise KeyError(f"Unknown kind '{kind}' for GPT2WeightExtractor")

    # ---- Module replacement ------------------------------------------------

    def get_module(self, layer: int, kind: str) -> nn.Module:
        """Return the nn.Module owning the weight for (layer, kind).

        For ``"attn.q"`` / ``"attn.k"`` / ``"attn.v"``: returns the individual
        projection module, defusing the fused ``c_attn`` on first access.
        """
        block = self._block(layer)
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            c_attn = block.attn.c_attn
            if isinstance(c_attn, _DefusedQKV):
                return getattr(c_attn, f"{proj}_proj")
            defused = _split_fused_linear(c_attn, self.config.n_embd)
            block.attn.c_attn = defused
            return getattr(defused, f"{proj}_proj")
        if kind == "attn.out":
            return block.attn.c_proj
        if kind == "ffn.up":
            return block.mlp.c_fc
        if kind == "ffn.down":
            return block.mlp.c_proj
        raise KeyError(f"Unknown kind '{kind}'")

    def replace_module(
        self, layer: int, kind: str, new_module: nn.Module,
    ) -> None:
        """Swap an nn.Module with a compressed variant.

        For ``"attn.q"`` / ``"attn.k"`` / ``"attn.v"``: defuses the fused
        ``c_attn`` into a ``_DefusedQKV`` wrapper (if not already defused)
        and replaces only the target projection.

        Args:
            layer: Layer index.
            kind: ``"attn.q"``, ``"attn.k"``, ``"attn.v"``,
                  ``"attn.out"``, ``"ffn.up"``, ``"ffn.down"``.
            new_module: Replacement (e.g. ShareLinear, DeltaLinear).
        """
        block = self._block(layer)
        if kind.startswith("attn.") and kind.split(".")[1] in _PROJ_INDEX:
            proj = kind.split(".")[1]
            c_attn = block.attn.c_attn
            if not isinstance(c_attn, _DefusedQKV):
                c_attn = _split_fused_linear(c_attn, self.config.n_embd)
                block.attn.c_attn = c_attn
            setattr(c_attn, f"{proj}_proj", new_module)
            return

        if kind == "attn.out":
            block.attn.c_proj = new_module
        elif kind == "ffn.up":
            block.mlp.c_fc = new_module
        elif kind == "ffn.down":
            block.mlp.c_proj = new_module
        else:
            raise KeyError(f"Unknown kind '{kind}'")

    # ---- Named parameter access --------------------------------------------

    def _get_named_param(self, name: str) -> nn.Parameter:
        for n, p in self.foundation.model.named_parameters():
            if n == name:
                return p
        raise KeyError(f"Parameter '{name}' not found")
