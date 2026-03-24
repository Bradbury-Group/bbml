"""Tests for PythiaWeightExtractor (GPT-NeoX architecture).

Uses a minimal synthetic GPT-NeoX-shaped model — no HuggingFace download required.
Covers: load, get_config, extract_index, get/set_weight (full + per-head),
get/replace_module (including defusion), trial context manager, registry lookup.

Pythia-160M dims: 12 layers, 12 heads, hidden=768, intermediate=3072, vocab=50304.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bbml.analysis.extractors.pythia import (
    PythiaWeightExtractor,
    _DefusedQKV,
)

# ---------------------------------------------------------------------------
# Minimal synthetic GPT-NeoX model
# ---------------------------------------------------------------------------

def _make_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=bias)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


def _make_fake_pythia(
    n_layers: int = 2,
    hidden: int = 48,
    n_heads: int = 6,
    intermediate: int = 96,
):
    """Build a minimal object matching GPTNeoXForCausalLM structure."""

    def make_layer():
        attn = SimpleNamespace(
            query_key_value=_make_linear(hidden, 3 * hidden, bias=True),
            dense=_make_linear(hidden, hidden, bias=True),
        )
        mlp = SimpleNamespace(
            dense_h_to_4h=_make_linear(hidden, intermediate, bias=True),
            dense_4h_to_h=_make_linear(intermediate, hidden, bias=True),
        )
        return SimpleNamespace(attention=attn, mlp=mlp)

    raw_layers = []
    for _ in range(n_layers):
        layer = make_layer()
        raw_layers.append(layer)

    # Build the nested namespace structure
    gpt_neox = SimpleNamespace(layers=raw_layers)
    cfg = SimpleNamespace(
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden,
        intermediate_size=intermediate,
        vocab_size=50304,
    )

    def named_parameters():
        for li, layer in enumerate(raw_layers):
            qkv = layer.attention.query_key_value
            if isinstance(qkv, _DefusedQKV):
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    mod = getattr(qkv, proj_name)
                    yield f"gpt_neox.layers.{li}.attention.query_key_value.{proj_name}.weight", mod.weight
            elif isinstance(qkv, nn.Linear):
                yield f"gpt_neox.layers.{li}.attention.query_key_value.weight", qkv.weight
            dense = layer.attention.dense
            if isinstance(dense, nn.Linear):
                yield f"gpt_neox.layers.{li}.attention.dense.weight", dense.weight
            h_to_4h = layer.mlp.dense_h_to_4h
            if isinstance(h_to_4h, nn.Linear):
                yield f"gpt_neox.layers.{li}.mlp.dense_h_to_4h.weight", h_to_4h.weight
            h_from_4h = layer.mlp.dense_4h_to_h
            if isinstance(h_from_4h, nn.Linear):
                yield f"gpt_neox.layers.{li}.mlp.dense_4h_to_h.weight", h_from_4h.weight

    model = SimpleNamespace(gpt_neox=gpt_neox, config=cfg)
    model.named_parameters = named_parameters
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def extractor():
    model = _make_fake_pythia()
    ext = PythiaWeightExtractor()
    ext.load(model)
    return ext


# ---------------------------------------------------------------------------
# load / get_config
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_self(self):
        model = _make_fake_pythia()
        ext = PythiaWeightExtractor()
        assert ext.load(model) is ext

    def test_rejects_non_neox(self):
        with pytest.raises(ValueError, match="structurally compatible"):
            PythiaWeightExtractor().load(object())

    def test_get_config(self, extractor):
        cfg = extractor.get_config()
        assert cfg["n_layers"] == 2
        assert cfg["n_head"] == 6
        assert cfg["n_kv_head"] == 6  # MHA
        assert cfg["n_embd"] == 48
        assert cfg["head_dim"] == 8
        assert cfg["intermediate_size"] == 96

    def test_not_loaded_raises(self):
        ext = PythiaWeightExtractor()
        with pytest.raises(RuntimeError, match="load()"):
            ext.get_config()

    def test_pythia_160m_dims(self):
        """Verify shapes match Pythia-160M spec."""
        model = _make_fake_pythia(
            n_layers=2, hidden=768, n_heads=12, intermediate=3072,
        )
        ext = PythiaWeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 12
        assert cfg["n_embd"] == 768
        assert cfg["head_dim"] == 64
        # QKV fused: get_weight should slice correctly
        q = ext.get_weight(0, "attn.q")
        assert q.shape == (768, 768)
        k = ext.get_weight(0, "attn.k")
        assert k.shape == (768, 768)


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

    def test_no_full(self, extractor):
        idx = extractor.extract_index(include_full=False)
        assert not any(u.kind.startswith("attn") for u in idx)

    def test_include_heads(self, extractor):
        idx = extractor.extract_index(include_heads=True, include_full=False, include_ffn=False)
        # MHA: 6 heads × 3 projections (q,k,v) × 2 layers = 36
        assert len(idx) == 6 * 3 * 2

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
        assert q.shape == (48, 48)
        assert k.shape == (48, 48)
        assert v.shape == (48, 48)
        # They should be different tensors (different rows of fused weight)
        assert not torch.equal(q, k)

    def test_per_head_shape(self, extractor):
        w = extractor.get_weight(0, "attn.q", head=0)
        assert w.shape == (8, 48)  # head_dim=8, hidden=48

    def test_per_head_out_of_range(self, extractor):
        with pytest.raises(IndexError, match="out of range"):
            extractor.get_weight(0, "attn.q", head=6)

    def test_set_weight_round_trip(self, extractor):
        original = extractor.get_weight(0, "attn.q").clone()
        new_val = torch.zeros_like(original)
        extractor.set_weight(0, "attn.q", new_val)
        assert torch.allclose(extractor.get_weight(0, "attn.q"), new_val)
        # Restore
        extractor.set_weight(0, "attn.q", original)
        assert torch.allclose(extractor.get_weight(0, "attn.q"), original)

    def test_set_weight_per_head(self, extractor):
        head_val = torch.ones(8, 48)
        extractor.set_weight(0, "attn.q", head_val, head=2)
        assert torch.allclose(extractor.get_weight(0, "attn.q", head=2), head_val)

    def test_set_weight_ffn(self, extractor):
        original = extractor.get_weight(0, "ffn.up").clone()
        new_val = torch.zeros_like(original)
        extractor.set_weight(0, "ffn.up", new_val)
        assert torch.allclose(extractor.get_weight(0, "ffn.up"), new_val)
        extractor.set_weight(0, "ffn.up", original)

    def test_set_weight_auto_cast(self, extractor):
        w_fp64 = extractor.get_weight(0, "ffn.down").double()
        extractor.set_weight(0, "ffn.down", w_fp64)
        assert extractor.get_weight(0, "ffn.down").dtype == torch.float32


# ---------------------------------------------------------------------------
# get_module / replace_module (defusion)
# ---------------------------------------------------------------------------

class TestModuleReplacement:
    def test_get_module_returns_linear(self, extractor):
        mod = extractor.get_module(0, "attn.q")
        assert isinstance(mod, nn.Linear)

    def test_get_module_defuses(self, extractor):
        """Accessing attn.q should defuse the fused query_key_value."""
        _ = extractor.get_module(0, "attn.q")
        qkv = extractor._model.gpt_neox.layers[0].attention.query_key_value
        assert isinstance(qkv, _DefusedQKV)

    def test_get_module_ffn(self, extractor):
        mod = extractor.get_module(0, "ffn.up")
        assert isinstance(mod, nn.Linear)
        assert mod.weight.shape == (96, 48)

    def test_replace_module_attn(self, extractor):
        new_lin = nn.Linear(48, 48, bias=True)
        extractor.replace_module(0, "attn.k", new_lin)
        assert extractor.get_module(0, "attn.k") is new_lin

    def test_replace_module_ffn(self, extractor):
        new_lin = nn.Linear(48, 96, bias=True)
        extractor.replace_module(0, "ffn.up", new_lin)
        assert extractor.get_module(0, "ffn.up") is new_lin

    def test_replace_unknown_kind(self, extractor):
        with pytest.raises(KeyError):
            extractor.replace_module(0, "attn.mystery", nn.Linear(4, 4))

    def test_defusion_preserves_weights(self, extractor):
        """After defusion, individual Q/K/V should match fused slices."""
        q_before = extractor.get_weight(0, "attn.q").clone()
        k_before = extractor.get_weight(0, "attn.k").clone()
        v_before = extractor.get_weight(0, "attn.v").clone()
        # Trigger defusion
        extractor.get_module(0, "attn.q")
        assert torch.allclose(extractor.get_weight(0, "attn.q"), q_before)
        assert torch.allclose(extractor.get_weight(0, "attn.k"), k_before)
        assert torch.allclose(extractor.get_weight(0, "attn.v"), v_before)


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

def test_registered_as_pythia():
    from bbml.analysis import get_adapter
    ext = get_adapter("pythia")
    assert isinstance(ext, PythiaWeightExtractor)
