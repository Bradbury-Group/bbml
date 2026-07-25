"""Seam 3: Foundation.save_training_state / load_training_state (single-rank)."""
import pytest
import torch

from tests.seam_helpers import TinyFoundation


def _stepped(fnd):
    opt = torch.optim.AdamW(fnd.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=3)
    loss, _ = fnd.single_step({"x": torch.randn(2, 4)})
    loss.backward()
    opt.step()
    sched.step()
    return opt, sched


class TestTrainingState:
    def test_delta_roundtrip(self, tmp_path):
        fnd = TinyFoundation()
        opt, sched = _stepped(fnd)
        orig = {k: v.detach().clone() for k, v in fnd.state_dict().items()}
        orig_lr = sched.get_last_lr()[0]

        fnd.save_training_state(tmp_path, opt, sched, metadata={"tokens": 7})
        assert (tmp_path / "optimizer.pt").exists()
        assert (tmp_path / "lr_scheduler.pt").exists()

        with torch.no_grad():
            for p in fnd.parameters():
                p.add_(1.0)

        opt2 = torch.optim.AdamW(fnd.parameters(), lr=1e-2)
        sched2 = torch.optim.lr_scheduler.ConstantLR(opt2, factor=0.5, total_iters=3)
        meta = fnd.load_training_state(tmp_path, opt2, sched2)

        assert isinstance(meta, dict)
        for k, v in fnd.state_dict().items():
            assert torch.allclose(v, orig[k]), k
        assert sched2.get_last_lr()[0] == orig_lr
        assert opt2.state_dict()["state"], "optimizer state not restored"

    def test_scheduler_none_skips_lrs_file(self, tmp_path):
        fnd = TinyFoundation()
        opt = torch.optim.SGD(fnd.parameters(), lr=0.1)
        fnd.save_training_state(tmp_path, opt, None)
        assert (tmp_path / "optimizer.pt").exists()
        assert not (tmp_path / "lr_scheduler.pt").exists()

    def test_delta_rejected_multirank(self, tmp_path, monkeypatch):
        # world>1 delta writes only local shards; guard must fail loud.
        monkeypatch.setenv("WORLD_SIZE", "2")
        fnd = TinyFoundation()
        opt = torch.optim.SGD(fnd.parameters(), lr=0.1)
        with pytest.raises(RuntimeError, match="delta"):
            fnd.save_training_state(tmp_path, opt, None)
