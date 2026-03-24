"""Tests for Phi3WeightExtractor (Phi-3 architecture).

Uses a minimal synthetic Phi-3-shaped model — no HuggingFace download required.
Covers: load, get_config, extract_index, get/set_weight (full + per-head),
get/replace_module (including QKV defusion), trial, fused gate_up_proj handling,
registry lookup.

Phi-3-mini-4k dims: 32 layers, 32 heads (MHA), hidden=3072, head_dim=96,
intermediate=8192, vocab=32064.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bbml.analysis.extractors.phi3 import (
    Phi3WeightExtractor,
    _DefusedQKV,
)

# ---------------------------------------------------------------------------
# Minimal synthetic Phi-3 model
# ---------------------------------------------------------------------------

def _make_linear(in_f: int, out_f: int) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=False)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    return lin


def _make_fake_phi3(
    n_layers: int = 2,
    hidden: int = 48,
    n_heads: int = 6,
    n_kv_heads: int = 6,
    intermediate: int = 96,
):
    """Build a minimal object matching Phi3ForCausalLM structure.

    Phi-3 has fused qkv_proj and fused gate_up_proj.
    """
    head_dim = hidden // n_heads
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim

    def make_layer():
        attn = SimpleNamespace(
            qkv_proj=_make_linear(hidden, qkv_dim),
            o_proj=_make_linear(q_dim, hidden),
        )
        mlp = SimpleNamespace(
            gate_up_proj=_make_linear(hidden, 2 * intermediate),
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
        vocab_size=32064,
    )

    def named_parameters():
        for li, layer in enumerate(layers):
            qkv = layer.self_attn.qkv_proj
            if isinstance(qkv, _DefusedQKV):
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    mod = getattr(qkv, proj_name)
                    yield f"model.layers.{li}.self_attn.qkv_proj.{proj_name}.weight", mod.weight
            elif isinstance(qkv, nn.Linear):
                yield f"model.layers.{li}.self_attn.qkv_proj.weight", qkv.weight
            o = layer.self_attn.o_proj
            if isinstance(o, nn.Linear):
                yield f"model.layers.{li}.self_attn.o_proj.weight", o.weight
            gu = layer.mlp.gate_up_proj
            if isinstance(gu, nn.Linear):
                yield f"model.layers.{li}.mlp.gate_up_proj.weight", gu.weight
            down = layer.mlp.down_proj
            if isinstance(down, nn.Linear):
                yield f"model.layers.{li}.mlp.down_proj.weight", down.weight

    model = SimpleNamespace(model=model_inner, config=cfg)
    model.named_parameters = named_parameters
    return model


def _make_fake_phi3_gqa(
    n_layers: int = 2,
    hidden: int = 48,
    n_heads: int = 6,
    n_kv_heads: int = 2,
    intermediate: int = 96,
):
    """Hypothetical Phi-3 GQA variant for future-proofing."""
    return _make_fake_phi3(
        n_layers=n_layers, hidden=hidden, n_heads=n_heads,
        n_kv_heads=n_kv_heads, intermediate=intermediate,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def extractor():
    model = _make_fake_phi3()
    ext = Phi3WeightExtractor()
    ext.load(model)
    return ext


@pytest.fixture()
def extractor_gqa():
    """Hypothetical GQA: 6 Q heads, 2 KV heads."""
    model = _make_fake_phi3_gqa()
    ext = Phi3WeightExtractor()
    ext.load(model)
    return ext


# ---------------------------------------------------------------------------
# load / get_config
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_self(self):
        model = _make_fake_phi3()
        ext = Phi3WeightExtractor()
        assert ext.load(model) is ext

    def test_rejects_non_phi3(self):
        with pytest.raises(ValueError, match="structurally compatible"):
            Phi3WeightExtractor().load(object())

    def test_get_config(self, extractor):
        cfg = extractor.get_config()
        assert cfg["n_layers"] == 2
        assert cfg["n_head"] == 6
        assert cfg["n_kv_head"] == 6  # MHA
        assert cfg["n_embd"] == 48
        assert cfg["head_dim"] == 8
        assert cfg["intermediate_size"] == 96

    def test_not_loaded_raises(self):
        ext = Phi3WeightExtractor()
        with pytest.raises(RuntimeError, match="load()"):
            ext.get_config()

    def test_phi3_mini_dims(self):
        """Verify shapes match Phi-3-mini spec."""
        model = _make_fake_phi3(
            n_layers=2, hidden=3072, n_heads=32, n_kv_heads=32,
            intermediate=8192,
        )
        ext = Phi3WeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 32
        assert cfg["n_kv_head"] == 32
        assert cfg["n_embd"] == 3072
        assert cfg["head_dim"] == 96
        q = ext.get_weight(0, "attn.q")
        assert q.shape == (3072, 3072)
        k = ext.get_weight(0, "attn.k")
        assert k.shape == (3072, 3072)


# ---------------------------------------------------------------------------
# extract_index
# ---------------------------------------------------------------------------

class TestExtractIndex:
    def test_full_kinds(self, extractor):
        idx = extractor.extract_index()
        kinds = set(u.kind for u in idx)
        assert kinds == {"attn.q", "attn.k", "attn.v", "attn.out",
                         "ffn.up", "ffn.down"}

    def test_layer_count(self, extractor):
        idx = extractor.extract_index()
        assert set(idx.layers()) == {0, 1}

    def test_no_ffn(self, extractor):
        idx = extractor.extract_index(include_ffn=False)
        assert not any(u.kind.startswith("ffn") for u in idx)

    def test_include_heads_mha(self, extractor):
        idx = extractor.extract_index(include_heads=True, include_full=False, include_ffn=False)
        # MHA: 6 heads × 3 projections × 2 layers = 36
        assert len(idx) == 6 * 3 * 2

    def test_include_heads_gqa(self, extractor_gqa):
        idx = extractor_gqa.extract_index(include_heads=True, include_full=False, include_ffn=False)
        # Q: 6 heads, K: 2 heads, V: 2 heads → (6+2+2) × 2 layers = 20
        assert len(idx) == (6 + 2 + 2) * 2

    def test_tensors_are_clones(self, extractor):
        idx = extractor.extract_index()
        w0 = extractor.get_weight(0, "attn.q")
        unit = next(u for u in idx if u.layer == 0 and u.kind == "attn.q")
        assert unit.tensor.data_ptr() != w0.data_ptr()

    def test_unit_count_full(self, extractor):
        # 2 layers × (4 attn + 2 ffn) = 12
        idx = extractor.extract_index()
        assert len(idx) == 12


# ---------------------------------------------------------------------------
# get_weight / set_weight
# ---------------------------------------------------------------------------

class TestGetSetWeight:
    @pytest.mark.parametrize("kind", [
        "attn.q", "attn.k", "attn.v", "attn.out", "ffn.up", "ffn.down",
    ])
    def test_get_weight_all_kinds(self, extractor, kind):
        w = extractor.get_weight(0, kind)
        assert isinstance(w, torch.Tensor)
        assert w.dim() == 2

    def test_get_weight_unknown_kind(self, extractor):
        with pytest.raises(KeyError):
            extractor.get_weight(0, "attn.banana")

    def test_fused_qkv_slicing(self, extractor):
        """Q, K, V should be non-overlapping slices of the fused weight."""
        q = extractor.get_weight(0, "attn.q")
        k = extractor.get_weight(0, "attn.k")
        v = extractor.get_weight(0, "attn.v")
        # MHA: all same shape
        assert q.shape == (48, 48)
        assert k.shape == (48, 48)
        assert v.shape == (48, 48)

    def test_fused_qkv_gqa_slicing(self, extractor_gqa):
        """Under GQA, K/V should be smaller than Q."""
        q = extractor_gqa.get_weight(0, "attn.q")
        k = extractor_gqa.get_weight(0, "attn.k")
        v = extractor_gqa.get_weight(0, "attn.v")
        assert q.shape == (48, 48)   # 6 heads × 8 head_dim
        assert k.shape == (16, 48)   # 2 kv_heads × 8 head_dim
        assert v.shape == (16, 48)

    def test_ffn_up_is_fused_gate_up(self, extractor):
        """ffn.up returns the full fused gate_up_proj weight."""
        w = extractor.get_weight(0, "ffn.up")
        assert w.shape == (2 * 96, 48)  # 2*intermediate × hidden

    def test_per_head_shape(self, extractor):
        w = extractor.get_weight(0, "attn.q", head=0)
        assert w.shape == (8, 48)  # head_dim=8, hidden=48

    def test_per_head_kv_gqa(self, extractor_gqa):
        # n_kv_heads=2
        k = extractor_gqa.get_weight(0, "attn.k", head=0)
        assert k.shape == (8, 48)
        with pytest.raises(IndexError, match="out of range"):
            extractor_gqa.get_weight(0, "attn.k", head=2)

    def test_per_head_out_of_range(self, extractor):
        with pytest.raises(IndexError, match="out of range"):
            extractor.get_weight(0, "attn.q", head=6)

    def test_set_weight_round_trip(self, extractor):
        original = extractor.get_weight(0, "attn.q").clone()
        new_val = torch.zeros_like(original)
        extractor.set_weight(0, "attn.q", new_val)
        assert torch.allclose(extractor.get_weight(0, "attn.q"), new_val)
        extractor.set_weight(0, "attn.q", original)
        assert torch.allclose(extractor.get_weight(0, "attn.q"), original)

    def test_set_weight_per_head(self, extractor):
        head_val = torch.ones(8, 48)
        extractor.set_weight(0, "attn.q", head_val, head=2)
        assert torch.allclose(extractor.get_weight(0, "attn.q", head=2), head_val)

    def test_set_weight_ffn(self, extractor):
        original = extractor.get_weight(0, "ffn.down").clone()
        new_val = torch.zeros_like(original)
        extractor.set_weight(0, "ffn.down", new_val)
        assert torch.allclose(extractor.get_weight(0, "ffn.down"), new_val)
        extractor.set_weight(0, "ffn.down", original)

    def test_set_weight_auto_cast(self, extractor):
        w_fp64 = extractor.get_weight(0, "ffn.down").double()
        extractor.set_weight(0, "ffn.down", w_fp64)
        assert extractor.get_weight(0, "ffn.down").dtype == torch.float32


# ---------------------------------------------------------------------------
# get_module / replace_module (defusion)
# ---------------------------------------------------------------------------

class TestModuleReplacement:
    def test_get_module_attn_returns_linear(self, extractor):
        mod = extractor.get_module(0, "attn.q")
        assert isinstance(mod, nn.Linear)

    def test_get_module_defuses_qkv(self, extractor):
        _ = extractor.get_module(0, "attn.q")
        qkv = extractor._model.model.layers[0].self_attn.qkv_proj
        assert isinstance(qkv, _DefusedQKV)

    def test_get_module_ffn_up_returns_fused(self, extractor):
        """ffn.up returns the fused gate_up_proj module."""
        mod = extractor.get_module(0, "ffn.up")
        assert isinstance(mod, nn.Linear)
        assert mod.weight.shape == (2 * 96, 48)

    def test_replace_module_attn(self, extractor):
        new_lin = nn.Linear(48, 48, bias=False)
        extractor.replace_module(0, "attn.k", new_lin)
        assert extractor.get_module(0, "attn.k") is new_lin

    def test_replace_module_ffn_up_raises(self, extractor):
        """Replacing fused gate_up_proj should raise NotImplementedError."""
        new_lin = nn.Linear(48, 192, bias=False)
        with pytest.raises(NotImplementedError, match="fused gate_up_proj"):
            extractor.replace_module(0, "ffn.up", new_lin)

    def test_replace_module_ffn_down(self, extractor):
        new_lin = nn.Linear(96, 48, bias=False)
        extractor.replace_module(0, "ffn.down", new_lin)
        assert extractor.get_module(0, "ffn.down") is new_lin

    def test_replace_unknown_kind(self, extractor):
        with pytest.raises(KeyError):
            extractor.replace_module(0, "attn.mystery", nn.Linear(4, 4))

    def test_defusion_preserves_weights(self, extractor):
        q_before = extractor.get_weight(0, "attn.q").clone()
        k_before = extractor.get_weight(0, "attn.k").clone()
        v_before = extractor.get_weight(0, "attn.v").clone()
        extractor.get_module(0, "attn.q")
        assert torch.allclose(extractor.get_weight(0, "attn.q"), q_before)
        assert torch.allclose(extractor.get_weight(0, "attn.k"), k_before)
        assert torch.allclose(extractor.get_weight(0, "attn.v"), v_before)

    def test_defusion_preserves_weights_gqa(self, extractor_gqa):
        q_before = extractor_gqa.get_weight(0, "attn.q").clone()
        k_before = extractor_gqa.get_weight(0, "attn.k").clone()
        v_before = extractor_gqa.get_weight(0, "attn.v").clone()
        extractor_gqa.get_module(0, "attn.q")
        assert torch.allclose(extractor_gqa.get_weight(0, "attn.q"), q_before)
        assert torch.allclose(extractor_gqa.get_weight(0, "attn.k"), k_before)
        assert torch.allclose(extractor_gqa.get_weight(0, "attn.v"), v_before)


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

def test_registered_as_phi3():
    from bbml.analysis import get_adapter
    ext = get_adapter("phi3")
    assert isinstance(ext, Phi3WeightExtractor)
