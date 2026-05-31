"""Stateless logging helpers used by FSDP-aware mixins.

Currently exposes :func:`log_loss_buckets`. This is the canonical source of
truth — DualSigmas + pretrain-{drawing,opd,sliders} will switch to this
implementation in a later wave.
"""
from __future__ import annotations

from typing import Optional

import torch


def log_loss_buckets(
    t: torch.Tensor,
    loss_per_sample: torch.Tensor,
    *,
    train_mask: Optional[torch.Tensor] = None,
    num_buckets: int = 10,
) -> tuple[dict[str, float], dict[str, int]]:
    """Bucket per-sample losses by timestep ``t in [0, 1]`` and return means.

    Bins span ``[0.0, 0.1, ..., 0.9, 1.0]`` by default (``num_buckets=10``).
    Empty bins are dropped from the returned ``means`` dict so the wandb chart
    surface stays clean.

    Args:
        t: timestep per sample, shape ``[B]``, expected in ``[0, 1]``.
        loss_per_sample: per-sample loss, shape ``[B]``.
        train_mask: optional binary mask, shape ``[B]``. ``None`` is treated
            as all-ones (every sample participates).
        num_buckets: number of evenly spaced bins. Default 10.

    Returns:
        ``(means, counts)`` where:
            - ``means[key] = mean loss in that bucket`` (only non-empty buckets).
            - ``counts[key] = number of samples in that bucket``.
            - Keys follow the ``loss_t/{lo:.1f}-{hi:.1f}`` convention.

    This function is stateless and side-effect-free; callers (e.g.
    :class:`bbml.train.metrics_mixin.MetricsMixin`) layer the wandb log on top.
    """
    with torch.no_grad():
        if train_mask is None:
            train_mask = torch.ones_like(t)
        t_cpu = t.detach().float().cpu()
        loss_cpu = loss_per_sample.detach().float().cpu()
        train_mask_cpu = train_mask.detach().float().cpu()
        bucket_idx = (t_cpu * num_buckets).long().clamp(0, num_buckets - 1)

        means: dict[str, float] = {}
        counts: dict[str, int] = {}
        for b in range(num_buckets):
            in_bucket = (bucket_idx == b) & (train_mask_cpu > 0)
            count = int(in_bucket.sum().item())
            lo = b / num_buckets
            hi = (b + 1) / num_buckets
            key = f"loss_t/{lo:.1f}-{hi:.1f}"
            counts[key] = count
            if count > 0:
                means[key] = float(loss_cpu[in_bucket].mean().item())
        return means, counts


__all__: list[str] = ["log_loss_buckets"]
