"""Tests for Qwen2 via LlamaWeightExtractor (registered as "qwen").

Qwen2ForCausalLM is structurally identical to LlamaForCausalLM.
Uses Qwen2.5-7B dimensions: 28 layers, 28 Q heads, 4 KV heads,
d_model=3584, d_head=128, intermediate=18944.
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


def _make_fake_qwen(
    n_layers: int = 2,
    hidden: int = 3584,
    n_heads: int = 28,
    n_kv_heads: int = 4,
    intermediate: int = 18944,
):
    """Build a minimal object matching Qwen2ForCausalLM structure."""
    head_dim = hidden // n_heads

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
        intermediate_size=intermediate,
        vocab_size=151936,
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


# Use small dims for fast tests, real dims for shape validation
@pytest.fixture()
def extractor_small():
    """Small Qwen-shaped model for fast tests (GQA ratio 7:1 like real Qwen2.5-7B)."""
    model = _make_fake_qwen(n_layers=2, hidden=56, n_heads=7, n_kv_heads=1, intermediate=128)
    ext = LlamaWeightExtractor()
    ext.load(model)
    return ext


class TestQwenConfig:
    def test_load_accepts_qwen_structure(self):
        model = _make_fake_qwen(n_layers=2, hidden=56, n_heads=7, n_kv_heads=1, intermediate=128)
        ext = LlamaWeightExtractor()
        assert ext.load(model) is ext

    def test_config_reads_kv_heads(self, extractor_small):
        cfg = extractor_small.get_config()
        assert cfg["n_head"] == 7
        assert cfg["n_kv_head"] == 1
        assert cfg["head_dim"] == 8

    def test_real_qwen_dims(self):
        """Verify shapes match Qwen2.5-7B spec (28 heads, 4 KV heads, d=3584)."""
        model = _make_fake_qwen(n_layers=2, hidden=3584, n_heads=28, n_kv_heads=4, intermediate=18944)
        ext = LlamaWeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 28
        assert cfg["n_kv_head"] == 4
        assert cfg["n_embd"] == 3584
        assert cfg["head_dim"] == 128
        # K/V shape: [n_kv_heads * head_dim, hidden] = [512, 3584]
        k_weight = ext.get_weight(0, "attn.k")
        assert k_weight.shape == (512, 3584)


class TestQwenGQA:
    def test_q_head_count(self, extractor_small):
        idx = extractor_small.extract_index(include_heads=True, include_full=False, include_ffn=False)
        q_units = [u for u in idx if u.kind == "attn.q"]
        assert len(q_units) == 7 * 2  # 7 heads × 2 layers

    def test_kv_head_count(self, extractor_small):
        idx = extractor_small.extract_index(include_heads=True, include_full=False, include_ffn=False)
        k_units = [u for u in idx if u.kind == "attn.k"]
        assert len(k_units) == 1 * 2  # 1 KV head × 2 layers

    def test_kv_head_out_of_range(self, extractor_small):
        with pytest.raises(IndexError, match="out of range"):
            extractor_small.get_weight(0, "attn.k", head=1)

    def test_per_head_shape(self, extractor_small):
        # head_dim = 56 // 7 = 8
        q_head = extractor_small.get_weight(0, "attn.q", head=0)
        assert q_head.shape == (8, 56)
        k_head = extractor_small.get_weight(0, "attn.k", head=0)
        assert k_head.shape == (8, 56)


class TestQwenRegistry:
    def test_registered_as_qwen(self):
        from bbml.analysis import get_adapter
        ext = get_adapter("qwen")
        assert isinstance(ext, LlamaWeightExtractor)

    def test_qwen_and_llama_same_class(self):
        from bbml.analysis import get_adapter
        qwen = get_adapter("qwen")
        llama = get_adapter("llama")
        assert type(qwen) is type(llama)
