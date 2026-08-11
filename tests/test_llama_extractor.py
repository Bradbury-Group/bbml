"""Tests for LlamaWeightExtractor.

Uses a minimal synthetic LLaMA-shaped model — no HuggingFace download required.
Covers: load, get_config, extract_index, get/set_weight (full + per-head),
get/replace_module, trial context manager, GQA head bounds, unknown kind errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bbml.analysis.extractors.llama import LlamaWeightExtractor, _KIND_PATH


# ---------------------------------------------------------------------------
# Minimal synthetic LLaMA model (no transformers dependency)
# ---------------------------------------------------------------------------

def _make_linear(in_f: int, out_f: int) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=False)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    return lin


def _make_fake_llama(
    n_layers: int = 2,
    hidden: int = 16,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    intermediate: int = 32,
):
    """Build a minimal object that structurally matches LlamaForCausalLM."""
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
        vocab_size=256,
    )

    # named_parameters stub — walks all nn.Module leaves
    def named_parameters():
        for li, layer in enumerate(layers):
            for kind, (sub, attr) in _KIND_PATH.items():
                mod = getattr(getattr(layer, sub), attr)
                if isinstance(mod, nn.Linear):
                    yield f"model.layers.{li}.{sub}.{attr}.weight", mod.weight

    model = SimpleNamespace(model=model_inner, config=cfg)
    model.named_parameters = named_parameters
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def extractor():
    model = _make_fake_llama()
    ext = LlamaWeightExtractor()
    ext.load(model)
    return ext


@pytest.fixture()
def extractor_gqa():
    """GQA: 4 Q heads, 1 KV head."""
    model = _make_fake_llama(n_kv_heads=1)
    ext = LlamaWeightExtractor()
    ext.load(model)
    return ext


# ---------------------------------------------------------------------------
# load / get_config
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_self(self):
        model = _make_fake_llama()
        ext = LlamaWeightExtractor()
        assert ext.load(model) is ext

    def test_rejects_non_llama(self):
        with pytest.raises(ValueError, match="structurally compatible"):
            LlamaWeightExtractor().load(object())

    def test_get_config(self, extractor):
        cfg = extractor.get_config()
        assert cfg["n_layers"] == 2
        assert cfg["n_head"] == 4
        assert cfg["n_kv_head"] == 2
        assert cfg["n_embd"] == 16
        assert cfg["head_dim"] == 4

    def test_not_loaded_raises(self):
        ext = LlamaWeightExtractor()
        with pytest.raises(RuntimeError, match="load()"):
            ext.get_config()


# ---------------------------------------------------------------------------
# extract_index
# ---------------------------------------------------------------------------

class TestExtractIndex:
    def test_full_kinds(self, extractor):
        idx = extractor.extract_index()
        kinds = set(u.kind for u in idx)
        assert kinds == {"attn.q", "attn.k", "attn.v", "attn.out",
                         "ffn.gate", "ffn.up", "ffn.down"}

    def test_layer_count(self, extractor):
        idx = extractor.extract_index()
        assert set(idx.layers()) == {0, 1}

    def test_no_ffn(self, extractor):
        idx = extractor.extract_index(include_ffn=False)
        assert not any(u.kind.startswith("ffn") for u in idx)

    def test_no_full(self, extractor):
        idx = extractor.extract_index(include_full=False)
        assert not any(u.kind.startswith("attn") for u in idx)

    def test_include_heads(self, extractor):
        idx = extractor.extract_index(include_heads=True, include_full=False, include_ffn=False)
        # Q: 4 heads × 2 layers = 8; K/V: 2 heads × 2 layers × 2 kinds = 8
        assert len(idx) == (4 + 2 + 2) * 2

    def test_tensors_are_clones(self, extractor):
        idx = extractor.extract_index()
        w0 = extractor.get_weight(0, "attn.q")
        unit = next(u for u in idx if u.layer == 0 and u.kind == "attn.q")
        assert not unit.tensor.data_ptr() == w0.data_ptr()


# ---------------------------------------------------------------------------
# get_weight / set_weight
# ---------------------------------------------------------------------------

class TestGetSetWeight:
    @pytest.mark.parametrize("kind", list(_KIND_PATH))
    def test_get_weight_all_kinds(self, extractor, kind):
        w = extractor.get_weight(0, kind)
        assert isinstance(w, torch.Tensor)
        assert w.dim() == 2

    def test_get_weight_unknown_kind(self, extractor):
        with pytest.raises(KeyError):
            extractor.get_weight(0, "attn.banana")

    def test_get_weight_per_head_q(self, extractor):
        w = extractor.get_weight(0, "attn.q", head=0)
        assert w.shape == (4, 16)  # head_dim=4, hidden=16

    def test_get_weight_per_head_k_gqa(self, extractor_gqa):
        # n_kv_heads=1, so only head=0 is valid
        w = extractor_gqa.get_weight(0, "attn.k", head=0)
        assert w.shape == (4, 16)

    def test_get_weight_head_out_of_range(self, extractor_gqa):
        with pytest.raises(IndexError, match="out of range"):
            extractor_gqa.get_weight(0, "attn.k", head=1)

    def test_set_weight_round_trip(self, extractor):
        original = extractor.get_weight(0, "attn.q").clone()
        new_val = torch.zeros_like(original)
        extractor.set_weight(0, "attn.q", new_val)
        assert torch.allclose(extractor.get_weight(0, "attn.q"), new_val)
        extractor.set_weight(0, "attn.q", original)

    def test_set_weight_per_head(self, extractor):
        head_val = torch.ones(4, 16)  # head_dim=4, hidden=16
        extractor.set_weight(0, "attn.q", head_val, head=0)
        assert torch.allclose(extractor.get_weight(0, "attn.q", head=0), head_val)

    def test_set_weight_auto_cast(self, extractor):
        """set_weight should accept FP64 and cast to model dtype (FP32)."""
        w_fp64 = extractor.get_weight(0, "ffn.up").double()
        extractor.set_weight(0, "ffn.up", w_fp64)
        assert extractor.get_weight(0, "ffn.up").dtype == torch.float32


# ---------------------------------------------------------------------------
# get_module / replace_module
# ---------------------------------------------------------------------------

class TestModuleReplacement:
    def test_get_module_returns_linear(self, extractor):
        mod = extractor.get_module(0, "attn.q")
        assert isinstance(mod, nn.Linear)

    @pytest.mark.parametrize("kind", list(_KIND_PATH))
    def test_get_module_all_kinds(self, extractor, kind):
        mod = extractor.get_module(0, kind)
        assert hasattr(mod, "weight")

    def test_replace_module(self, extractor):
        new_lin = nn.Linear(16, 16, bias=False)
        extractor.replace_module(0, "attn.q", new_lin)
        assert extractor.get_module(0, "attn.q") is new_lin

    def test_replace_unknown_kind(self, extractor):
        with pytest.raises(KeyError):
            extractor.replace_module(0, "attn.mystery", nn.Linear(4, 4))


# ---------------------------------------------------------------------------
# trial context manager
# ---------------------------------------------------------------------------

class TestTrial:
    def test_trial_restores_weight(self, extractor):
        original = extractor.get_weight(0, "attn.q").clone()
        with extractor.trial(0, "attn.q"):
            extractor.set_weight(0, "attn.q", torch.zeros_like(original))
            assert torch.allclose(extractor.get_weight(0, "attn.q"), torch.zeros_like(original))
        assert torch.allclose(extractor.get_weight(0, "attn.q"), original)

    def test_trial_restores_on_exception(self, extractor):
        original = extractor.get_weight(0, "ffn.down").clone()
        with pytest.raises(RuntimeError):
            with extractor.trial(0, "ffn.down"):
                extractor.set_weight(0, "ffn.down", torch.ones_like(original))
                raise RuntimeError("test")
        assert torch.allclose(extractor.get_weight(0, "ffn.down"), original)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registered_as_llama():
    from bbml.analysis import get_adapter
    ext = get_adapter("llama")
    assert isinstance(ext, LlamaWeightExtractor)
