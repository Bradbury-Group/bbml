"""Seam 7: warmup_stable_decay in LRSchedulerRegistry."""
import pytest
import torch

import bbml.train.lr_schedulers  # noqa: F401  (triggers registration)
from bbml.core.datamodels.configs import TrainerConfig
from bbml.registries import LRSchedulerRegistry
from bbml.train.simple_trainer import init_cls_from_config


class TestWarmupStableDecay:
    def test_registered(self):
        assert "warmup_stable_decay" in LRSchedulerRegistry.keys()

    def test_config_validator_accepts_it(self):
        cfg = TrainerConfig(project="test", lr_scheduler="warmup_stable_decay")
        assert cfg.lr_scheduler == "warmup_stable_decay"

    def test_builds_working_scheduler_via_config(self):
        # kwargs come from TrainerConfig extras (extra="allow").
        cfg = TrainerConfig(
            project="test", lr_scheduler="warmup_stable_decay",
            num_warmup_steps=2, num_stable_steps=2, num_decay_steps=2,
            num_training_steps=6,
        )
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=1.0)
        cls = LRSchedulerRegistry.get(cfg.lr_scheduler)
        sched = init_cls_from_config(cls, cfg, opt)

        # warmup: lr ramps up over the first 2 steps.
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(6):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert lrs[0] < lrs[2]          # warmed up
        assert lrs[2] == pytest.approx(1.0)  # stable at base lr
        assert lrs[-1] < lrs[2]         # decayed
