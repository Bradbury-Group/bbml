"""Tests for Gemma-2-9B via LlamaWeightExtractor (registered as "gemma2").

Gemma-2 is structurally identical to LLaMA for weight extraction, but has
a decoupled head_dim: hidden_size=3584, n_heads=16, head_dim=256
(256 != 3584 // 16 = 224).  These tests verify that the explicit head_dim
from config is respected.

Specs: 42 layers, GQA 16/8, d_model=3584, head_dim=256, intermediate=28672.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bbml.analysis.extractors.llama import LlamaWeightExtractor, _KIND_PATH


def _make_linear(in_f: int, out_f: int) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=False)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    return lin


def _make_fake_gemma2(
    n_layers: int = 2,
    hidden: int = 3584,
    n_heads: int = 16,
    n_kv_heads: int = 8,
    head_dim: int = 256,
    intermediate: int = 28672,
):
    """Build a minimal object matching Gemma2ForCausalLM structure.

    Key difference from LLaMA: head_dim is explicit and != hidden // n_heads.
    Q shape is [n_heads * head_dim, hidden], not [hidden, hidden].
    """
    def make_layer():
        attn = SimpleNamespace(
            q_proj=_make_linear(hidden, n_heads * head_dim),
            k_proj=_make_linear(hidden, n_kv_heads * head_dim),
            v_proj=_make_linear(hidden, n_kv_heads * head_dim),
            o_proj=_make_linear(n_heads * head_dim, hidden),
        )
        mlp = SimpleNamespace(
            gate_proj=_make_linear(hidden, intermediate),
            up_proj=_make_linear(hidden, intermediate),
            down_proj=_make_linear(intermediate, hidden),
        )
        return SimpleNamespace(self_attn=attn, mlp=mlp)

    layers = [make_layer() for _ in range(n_layers)]
    model_inner = SimpleNamespace(layers=layers)
    cfg = SimpleNamespace(
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        hidden_size=hidden,
        head_dim=head_dim,
        intermediate_size=intermediate,
        vocab_size=256128,
    )

    def named_parameters():
        for li, layer in enumerate(layers):
            for kind, (sub, attr) in _KIND_PATH.items():
                mod = getattr(getattr(layer, sub), attr)
                if isinstance(mod, nn.Linear):
                    yield f"model.layers.{li}.{sub}.{attr}.weight", mod.weight

    model = SimpleNamespace(model=model_inner, config=cfg)
    model.named_parameters = named_parameters
    return model


@pytest.fixture()
def extractor_small():
    """Small Gemma-2-shaped model with decoupled head_dim (GQA 4/2, head_dim=16 != 32//4=8)."""
    model = _make_fake_gemma2(
        n_layers=2, hidden=32, n_heads=4, n_kv_heads=2,
        head_dim=16, intermediate=64,
    )
    ext = LlamaWeightExtractor()
    ext.load(model)
    return ext


class TestDecoupledHeadDim:
    def test_head_dim_from_config(self, extractor_small):
        """head_dim should be 16 (from config), not 8 (hidden // n_heads)."""
        cfg = extractor_small.get_config()
        assert cfg["head_dim"] == 16
        assert cfg["n_embd"] == 32
        assert cfg["n_head"] == 4
        # Verify it's NOT hidden // n_heads
        assert cfg["head_dim"] != cfg["n_embd"] // cfg["n_head"]

    def test_per_head_q_shape(self, extractor_small):
        """Per-head Q slice should be [head_dim, hidden] = [16, 32]."""
        w = extractor_small.get_weight(0, "attn.q", head=0)
        assert w.shape == (16, 32)

    def test_per_head_k_shape(self, extractor_small):
        """Per-head K slice should be [head_dim, hidden] = [16, 32]."""
        w = extractor_small.get_weight(0, "attn.k", head=0)
        assert w.shape == (16, 32)

    def test_full_q_shape(self, extractor_small):
        """Full Q weight: [n_heads * head_dim, hidden] = [64, 32]."""
        w = extractor_small.get_weight(0, "attn.q")
        assert w.shape == (4 * 16, 32)

    def test_full_k_shape(self, extractor_small):
        """Full K weight: [n_kv_heads * head_dim, hidden] = [32, 32]."""
        w = extractor_small.get_weight(0, "attn.k")
        assert w.shape == (2 * 16, 32)

    def test_real_gemma2_dims(self):
        """Verify shapes match Gemma-2-9B spec."""
        model = _make_fake_gemma2(n_layers=2)
        ext = LlamaWeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 16
        assert cfg["n_kv_head"] == 8
        assert cfg["n_embd"] == 3584
        assert cfg["head_dim"] == 256
        # Q: [16*256, 3584] = [4096, 3584]
        assert ext.get_weight(0, "attn.q").shape == (4096, 3584)
        # K: [8*256, 3584] = [2048, 3584]
        assert ext.get_weight(0, "attn.k").shape == (2048, 3584)
        # O: [3584, 4096]
        assert ext.get_weight(0, "attn.out").shape == (3584, 4096)


class TestGemma2GQA:
    def test_head_counts_in_index(self, extractor_small):
        idx = extractor_small.extract_index(include_heads=True, include_full=False, include_ffn=False)
        q_units = [u for u in idx if u.kind == "attn.q"]
        k_units = [u for u in idx if u.kind == "attn.k"]
        v_units = [u for u in idx if u.kind == "attn.v"]
        assert len(q_units) == 4 * 2   # 4 Q heads × 2 layers
        assert len(k_units) == 2 * 2   # 2 KV heads × 2 layers
        assert len(v_units) == 2 * 2

    def test_kv_head_out_of_range(self, extractor_small):
        with pytest.raises(IndexError, match="out of range"):
            extractor_small.get_weight(0, "attn.k", head=2)

    def test_set_weight_per_head(self, extractor_small):
        head_val = torch.ones(16, 32)
        extractor_small.set_weight(0, "attn.q", head_val, head=1)
        assert torch.allclose(extractor_small.get_weight(0, "attn.q", head=1), head_val)


class TestGemma2Registry:
    def test_registered_as_gemma2(self):
        from bbml.analysis import get_adapter
        ext = get_adapter("gemma2")
        assert isinstance(ext, LlamaWeightExtractor)
