"""Seams 4, 5, 6: hooks, sharding plan default, metrics validation."""
import pytest
import torch

from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.foundation import Foundation
from bbml.train.simple_trainer import SimpleTrainer

from tests.seam_helpers import TinyFoundation, make_pipe


class TestMetricsValidation:
    def test_zerodim_tensor_coerced_to_float(self):
        fnd = TinyFoundation(metrics={"aux": torch.tensor(1.5), "scalar": 2.0})
        _, metrics = fnd({"x": torch.randn(2, 4)})
        assert metrics["aux"] == pytest.approx(1.5)
        assert isinstance(metrics["aux"], float)
        assert metrics["scalar"] == 2.0

    def test_nonscalar_tensor_raises_naming_key(self):
        fnd = TinyFoundation(metrics={"bad": torch.tensor([1.0, 2.0])})
        with pytest.raises(TypeError, match="bad"):
            fnd({"x": torch.randn(2, 4)})

    def test_scalar_only_loss_return_untouched(self):
        # single_step returning a bare Tensor (no dict) is passed through.
        class ScalarOnly(TinyFoundation):
            def single_step(self, batch):
                return self.head(batch["x"]).pow(2).mean()

        out = ScalarOnly()({"x": torch.randn(2, 4)})
        assert isinstance(out, torch.Tensor) and out.ndim == 0


class TestShardingPlan:
    def test_fsdp_blocks_default_raises(self):
        fnd = TinyFoundation()
        with pytest.raises(NotImplementedError, match="fsdp_blocks"):
            Foundation.fsdp_blocks(fnd)

    def test_parallelise_default_shards_fsdp_blocks(self, monkeypatch):
        import bbml.fsdp.parallelise as P
        captured = {}

        def fake(model, *, blocks, output_module, mesh, policy, **kw):
            captured.update(model=model, blocks=blocks, output_module=output_module,
                            mesh=mesh, policy=policy)

        monkeypatch.setattr(P, "fully_shard_model", fake)
        fnd = TinyFoundation()
        fnd.parallelise(mesh="MESH", policy="POLICY")
        assert captured["model"] is fnd
        assert captured["blocks"] is fnd.blocks
        assert captured["output_module"] is None
        assert captured["mesh"] == "MESH"
        assert captured["policy"] == "POLICY"


class TestHookOrder:
    def test_hooks_called_in_order(self, tmp_path):
        fnd = TinyFoundation()
        cfg = TrainerConfig(
            project="test", output_dir=tmp_path, optimizer="AdamW",
            lr_scheduler="ConstantLR", train_epochs=1,
        )
        pipe = make_pipe(n=4, batch_size=2)  # 2 batches -> 2 optimizer steps
        trainer = SimpleTrainer(fnd, cfg, pipe, None, None)
        trainer.train()

        # on_train_start: exactly once, at step 0, before any optimizer step.
        assert fnd.train_start_calls == [0]
        # on_optimizer_step: once per optimizer step, at the pre-increment step.
        assert fnd.optimizer_step_calls == [0, 1]
