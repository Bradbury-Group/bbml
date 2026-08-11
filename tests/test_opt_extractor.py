"""Tests for OPTWeightExtractor (OPT architecture).

Uses a minimal synthetic OPT-shaped model — no HuggingFace download required.
Covers: load, get_config, extract_index, get/set_weight (full + per-head),
get/replace_module, trial context manager, registry lookup.

OPT-125M dims: 12 layers, 12 heads, hidden=768, ffn_dim=3072, vocab=50272.
Key differences from LLaMA: out_proj (not o_proj), fc1/fc2 (not gate/up/down),
bias on all linears, model.model.decoder.layers path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bbml.analysis.extractors.opt import _KIND_PATH, OPTWeightExtractor

# ---------------------------------------------------------------------------
# Minimal synthetic OPT model
# ---------------------------------------------------------------------------

def _make_linear(in_f: int, out_f: int) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=True)
    torch.manual_seed(0)
    nn.init.normal_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def _make_fake_opt(
    n_layers: int = 2,
    hidden: int = 48,
    n_heads: int = 6,
    ffn_dim: int = 96,
):
    """Build a minimal object matching OPTForCausalLM structure."""

    def make_layer():
        attn = SimpleNamespace(
            q_proj=_make_linear(hidden, hidden),
            k_proj=_make_linear(hidden, hidden),
            v_proj=_make_linear(hidden, hidden),
            out_proj=_make_linear(hidden, hidden),
        )
        layer = SimpleNamespace(
            self_attn=attn,
            fc1=_make_linear(hidden, ffn_dim),
            fc2=_make_linear(ffn_dim, hidden),
        )
        return layer

    layers = [make_layer() for _ in range(n_layers)]
    decoder = SimpleNamespace(layers=layers)
    model_inner = SimpleNamespace(decoder=decoder)
    cfg = SimpleNamespace(
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden,
        ffn_dim=ffn_dim,
        vocab_size=50272,
    )

    def named_parameters():
        for li, layer in enumerate(layers):
            for kind, (sub, attr) in _KIND_PATH.items():
                if sub is not None:
                    mod = getattr(getattr(layer, sub), attr)
                else:
                    mod = getattr(layer, attr)
                if isinstance(mod, nn.Linear):
                    yield (
                        f"model.decoder.layers.{li}.{sub + '.' if sub else ''}{attr}.weight",
                        mod.weight,
                    )

    model = SimpleNamespace(model=model_inner, config=cfg)
    model.named_parameters = named_parameters
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def extractor():
    model = _make_fake_opt()
    ext = OPTWeightExtractor()
    ext.load(model)
    return ext


# ---------------------------------------------------------------------------
# load / get_config
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_returns_self(self):
        model = _make_fake_opt()
        ext = OPTWeightExtractor()
        assert ext.load(model) is ext

    def test_rejects_non_opt(self):
        with pytest.raises(ValueError, match="structurally compatible"):
            OPTWeightExtractor().load(object())

    def test_get_config(self, extractor):
        cfg = extractor.get_config()
        assert cfg["n_layers"] == 2
        assert cfg["n_head"] == 6
        assert cfg["n_kv_head"] == 6  # MHA
        assert cfg["n_embd"] == 48
        assert cfg["head_dim"] == 8
        assert cfg["intermediate_size"] == 96

    def test_not_loaded_raises(self):
        ext = OPTWeightExtractor()
        with pytest.raises(RuntimeError, match="load()"):
            ext.get_config()

    def test_opt_125m_dims(self):
        """Verify shapes match OPT-125M spec."""
        model = _make_fake_opt(
            n_layers=2, hidden=768, n_heads=12, ffn_dim=3072,
        )
        ext = OPTWeightExtractor().load(model)
        cfg = ext.get_config()
        assert cfg["n_head"] == 12
        assert cfg["n_embd"] == 768
        assert cfg["head_dim"] == 64
        q = ext.get_weight(0, "attn.q")
        assert q.shape == (768, 768)
        fc1 = ext.get_weight(0, "ffn.up")
        assert fc1.shape == (3072, 768)


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
    @pytest.mark.parametrize("kind", list(_KIND_PATH))
    def test_get_weight_all_kinds(self, extractor, kind):
        w = extractor.get_weight(0, kind)
        assert isinstance(w, torch.Tensor)
        assert w.dim() == 2

    def test_get_weight_unknown_kind(self, extractor):
        with pytest.raises(KeyError):
            extractor.get_weight(0, "attn.banana")

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

    def test_replace_module_attn(self, extractor):
        new_lin = nn.Linear(48, 48, bias=True)
        extractor.replace_module(0, "attn.q", new_lin)
        assert extractor.get_module(0, "attn.q") is new_lin

    def test_replace_module_ffn(self, extractor):
        new_lin = nn.Linear(48, 96, bias=True)
        extractor.replace_module(0, "ffn.up", new_lin)
        assert extractor.get_module(0, "ffn.up") is new_lin

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
            assert torch.allclose(
                extractor.get_weight(0, "attn.q"), torch.zeros_like(original),
            )
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

def test_registered_as_opt():
    from bbml.analysis import get_adapter
    ext = get_adapter("opt")
    assert isinstance(ext, OPTWeightExtractor)
