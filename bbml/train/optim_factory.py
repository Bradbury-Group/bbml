"""Shared optimizer / LR scheduler factory used by ``FullyShardTrainer``.

Factors the optimizer construction logic that currently lives in
``SimpleTrainer`` and ``AccelerateTrainer``. The existing call sites are NOT
updated in this round to keep the change additive; only the FSDP2 trainer
consumes these helpers. A later wave can switch the older trainers over.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from bbml.core.datamodels.configs import TrainerConfig
from bbml.registries import LRSchedulerRegistry, OptimizerRegistry
from bbml.train.param_groups import build_param_groups
from bbml.train.simple_trainer import init_cls_from_config


def build_param_groups_from_config(model: Module, cfg: TrainerConfig) -> list[dict[str, Any]] | Any:
    """Pick the param-group strategy off the config.

    Order of precedence (matches the existing trainers):
        1. ``cfg.param_group_rules`` -> regex-based grouping via
           :func:`bbml.train.param_groups.build_param_groups`.
        2. ``model.get_train_parameters()`` -> finetuner-aware default.
    """
    if getattr(cfg, "param_group_rules", None):
        base_lr = getattr(cfg, "lr", None)
        if base_lr is None:
            raise ValueError(
                "param_group_rules requires cfg.lr to be set (base learning rate)."
            )
        base_wd = getattr(cfg, "weight_decay", 0.0)
        return build_param_groups(
            model,
            base_lr=base_lr,
            base_wd=base_wd,
            rules=cfg.param_group_rules,
        )
    return model.get_train_parameters()


def build_optimizer_from_config(model: Module, cfg: TrainerConfig) -> Optimizer:
    """Construct the optimizer from ``cfg``.

    Resolution order:
        1. If ``model.optimizer`` returns a non-None ``Optimizer``, use it.
        2. Else dispatch on ``cfg.optimizer`` via the ``OptimizerRegistry``;
           param groups come from :func:`build_param_groups_from_config`.

    FSDP2 caveat: this MUST be called AFTER ``fully_shard`` so the optimizer
    sees DTensor-typed parameters; ``FullyShardTrainer`` bakes this ordering
    into its bootstrap.
    """
    model_optim = getattr(model, "optimizer", None)
    if model_optim is not None:
        return model_optim

    if cfg.optimizer is None:
        raise ValueError(
            "Cannot build optimizer: cfg.optimizer is None and model.optimizer is None."
        )

    optimizer_cls = OptimizerRegistry.get(cfg.optimizer)
    param_groups = build_param_groups_from_config(model, cfg)
    return init_cls_from_config(optimizer_cls, cfg, param_groups)


def build_lr_scheduler_from_config(
    optimizer: Optimizer,
    cfg: TrainerConfig,
    model: Module | None = None,
) -> LRScheduler:
    """Construct the LR scheduler from ``cfg``.

    Resolution order:
        1. If ``model is not None`` and ``model.lr_scheduler`` is non-None,
           use it.
        2. Else dispatch on ``cfg.lr_scheduler`` via ``LRSchedulerRegistry``.
    """
    if model is not None:
        model_sched = getattr(model, "lr_scheduler", None)
        if model_sched is not None:
            return model_sched

    if cfg.lr_scheduler is None:
        raise ValueError(
            "Cannot build LR scheduler: cfg.lr_scheduler is None and model.lr_scheduler is None."
        )
    lr_scheduler_cls = LRSchedulerRegistry.get(cfg.lr_scheduler)
    return init_cls_from_config(lr_scheduler_cls, cfg, optimizer)


__all__: list[str] = [
    "build_param_groups_from_config",
    "build_optimizer_from_config",
    "build_lr_scheduler_from_config",
]
