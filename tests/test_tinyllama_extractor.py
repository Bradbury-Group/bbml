"""Tests for TinyLlama-1.1B via LlamaWeightExtractor (registered as "tinyllama").

TinyLlama is LlamaForCausalLM with GQA: 22 layers, 32 Q heads, 4 KV heads,
hidden=2048, intermediate=5632, vocab=32000.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bbml.analysis.extractors.llama import _KIND_PATH, LlamaWeightExtractor


def _make_linear(in_f: int, out_f: int) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=False)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    return lin


def _make_fake_tinyllama(
    n_layers: int = 2,
    hidden: int = 64,
    n_heads: int = 8,
    n_kv_heads: int = 1,
    intermediate: int = 128,
):
    """Build a minimal object matching TinyLlama (LlamaForCausalLM, GQA)."""
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
        vocab_size=32000,
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
    """Small TinyLlama-shaped model (GQA ratio 8:1)."""
    model = _make_fake_tinyllama()
    ext = LlamaWeightExtractor()
    ext.load(model)
    return ext


class TestTinyLlamaConfig:
    def test_load_accepts_tinyllama_structure(self):
        model = _make_fake_tinyllama()
        ext = LlamaWeightExtractor()
        assert ext.load(model) is ext

    def test_config_reads_gqa(self, extractor_small):
        cfg = extractor_small.get_config()
        assert cfg["n_head"] == 8
        assert cfg["n_kv_head"] == 1
        assert cfg["head_dim"] == 8

    def test_real_tinyllama_dims(self):
        """Verify shapes match TinyLlama-1.1B spec (32 heads, 4 KV heads)."""
        model = _make_fake_tinyllama(
            n_layers=2, hidden=2048, n_heads=32,
            n_kv_heads=4, intermediate=5632,
        )
        ext = LlamaWeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 32
        assert cfg["n_kv_head"] == 4
        assert cfg["n_embd"] == 2048
        assert cfg["head_dim"] == 64
        # Q: [32*64, 2048] = [2048, 2048]
        q = ext.get_weight(0, "attn.q")
        assert q.shape == (2048, 2048)
        # K: [4*64, 2048] = [256, 2048]
        k = ext.get_weight(0, "attn.k")
        assert k.shape == (256, 2048)


class TestTinyLlamaGQA:
    def test_q_head_count(self, extractor_small):
        idx = extractor_small.extract_index(
            include_heads=True, include_full=False, include_ffn=False,
        )
        q_units = [u for u in idx if u.kind == "attn.q"]
        assert len(q_units) == 8 * 2  # 8 Q heads × 2 layers

    def test_kv_head_count(self, extractor_small):
        idx = extractor_small.extract_index(
            include_heads=True, include_full=False, include_ffn=False,
        )
        k_units = [u for u in idx if u.kind == "attn.k"]
        assert len(k_units) == 1 * 2  # 1 KV head × 2 layers

    def test_kv_head_out_of_range(self, extractor_small):
        with pytest.raises(IndexError, match="out of range"):
            extractor_small.get_weight(0, "attn.k", head=1)

    def test_per_head_shape(self, extractor_small):
        q_head = extractor_small.get_weight(0, "attn.q", head=0)
        assert q_head.shape == (8, 64)
        k_head = extractor_small.get_weight(0, "attn.k", head=0)
        assert k_head.shape == (8, 64)


class TestTinyLlamaRegistry:
    def test_registered_as_tinyllama(self):
        from bbml.analysis import get_adapter
        ext = get_adapter("tinyllama")
        assert isinstance(ext, LlamaWeightExtractor)

    def test_tinyllama_and_llama_same_class(self):
        from bbml.analysis import get_adapter
        tl = get_adapter("tinyllama")
        llama = get_adapter("llama")
        assert type(tl) is type(llama)
