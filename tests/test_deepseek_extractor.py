"""Tests for DeepSeek-7B via LlamaWeightExtractor (registered as "deepseek").

DeepSeek-LLM-7B-base is LlamaForCausalLM-compatible with MHA (32/32).
Uses: 30 layers, 32 heads, hidden=4096, intermediate=11008, vocab=102400.
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


def _make_fake_deepseek(
    n_layers: int = 2,
    hidden: int = 64,
    n_heads: int = 8,
    n_kv_heads: int = 8,
    intermediate: int = 128,
):
    """Build a minimal object matching DeepSeek-LLM-7B-base (LlamaForCausalLM)."""
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
        vocab_size=102400,
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
    """Small DeepSeek-shaped model (MHA, 8 heads)."""
    model = _make_fake_deepseek()
    ext = LlamaWeightExtractor()
    ext.load(model)
    return ext


class TestDeepSeekConfig:
    def test_load_accepts_deepseek_structure(self):
        model = _make_fake_deepseek()
        ext = LlamaWeightExtractor()
        assert ext.load(model) is ext

    def test_config_reads_mha(self, extractor_small):
        cfg = extractor_small.get_config()
        assert cfg["n_head"] == 8
        assert cfg["n_kv_head"] == 8  # MHA: same
        assert cfg["head_dim"] == 8
        assert cfg["n_embd"] == 64

    def test_real_deepseek_dims(self):
        """Verify shapes match DeepSeek-LLM-7B-base spec."""
        model = _make_fake_deepseek(
            n_layers=2, hidden=4096, n_heads=32,
            n_kv_heads=32, intermediate=11008,
        )
        ext = LlamaWeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 32
        assert cfg["n_kv_head"] == 32
        assert cfg["n_embd"] == 4096
        assert cfg["head_dim"] == 128
        # MHA: Q and K have same shape
        q = ext.get_weight(0, "attn.q")
        k = ext.get_weight(0, "attn.k")
        assert q.shape == k.shape == (4096, 4096)


class TestDeepSeekMHA:
    def test_q_and_k_same_head_count(self, extractor_small):
        idx = extractor_small.extract_index(
            include_heads=True, include_full=False, include_ffn=False,
        )
        q_units = [u for u in idx if u.kind == "attn.q"]
        k_units = [u for u in idx if u.kind == "attn.k"]
        assert len(q_units) == len(k_units) == 8 * 2  # 8 heads × 2 layers

    def test_per_head_shape(self, extractor_small):
        q_head = extractor_small.get_weight(0, "attn.q", head=0)
        k_head = extractor_small.get_weight(0, "attn.k", head=0)
        assert q_head.shape == k_head.shape == (8, 64)

    def test_set_weight_round_trip(self, extractor_small):
        original = extractor_small.get_weight(0, "attn.q").clone()
        new_val = torch.zeros_like(original)
        extractor_small.set_weight(0, "attn.q", new_val)
        assert torch.allclose(extractor_small.get_weight(0, "attn.q"), new_val)
        extractor_small.set_weight(0, "attn.q", original)


class TestDeepSeekRegistry:
    def test_registered_as_deepseek(self):
        from bbml.analysis import get_adapter
        ext = get_adapter("deepseek")
        assert isinstance(ext, LlamaWeightExtractor)

    def test_deepseek_and_llama_same_class(self):
        from bbml.analysis import get_adapter
        ds = get_adapter("deepseek")
        llama = get_adapter("llama")
        assert type(ds) is type(llama)
