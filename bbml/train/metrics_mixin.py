"""Master-only scalar / loss-bucket logging for FSDP-aware trainers.

Designed as a mixin so any ``Trainer`` subclass can compose it. Operations
are no-ops on non-master ranks; the underlying wandb backend handles the
``commit=False`` semantics. ``log_loss_buckets`` is a thin wrapper over
:func:`bbml.utils.logging_utils.log_loss_buckets` that adds the wandb log
call (which the stateless helper deliberately omits).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from bbml import logger
from bbml.fsdp.dist import is_master
from bbml.utils.logging_utils import log_loss_buckets as _log_loss_buckets_impl


class MetricsMixin:
    """Composable scalar / bucket logging surface for FSDP trainers.

    All log calls are gated on rank-0 to avoid clobbering the wandb stream
    across ranks. Subclasses must not assume distributed init when calling
    these — :func:`bbml.fsdp.dist.is_master` falls back to ``True`` when no
    process group exists, so single-process accelerate / SimpleTrainer
    composition stays valid.

    ``commit=False`` is the default so several log calls within one global
    step batch into a single wandb upload; the trainer's primary metric log
    (typically ``train_loss``) should be the only ``commit=True`` call.
    """

    def log_scalar(
        self,
        key: str,
        value: float | int | torch.Tensor,
        *,
        commit: bool = False,
        gstep: int | None = None,
    ) -> None:
        """Log a single scalar to wandb on master.

        Args:
            key: wandb metric name.
            value: scalar value; tensors are converted via ``.item()``.
            commit: forwarded to ``logger.log``. Default False so multiple
                scalars in the same gstep batch together.
            gstep: optional explicit step; defaults to the logger's fallback
                step counter.
        """
        if not is_master():
            return
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().cpu().item())
        logger.log({key: value}, step=gstep, commit=commit)

    def log_loss_buckets(
        self,
        t: torch.Tensor,
        loss_per_sample: torch.Tensor,
        *,
        train_mask: Optional[torch.Tensor] = None,
        num_buckets: int = 10,
        gstep: int | None = None,
        also_log_counts: bool = False,
    ) -> dict[str, float]:
        """Compute per-t-bucket loss means and log to wandb on master.

        Compute happens on every rank (it's local CPU work over the local
        batch); wandb log happens only on rank-0.

        Returns the means dict so callers can also stash it into a primary
        metric payload (``train_loss`` log alongside the buckets).
        """
        means, counts = _log_loss_buckets_impl(
            t,
            loss_per_sample,
            train_mask=train_mask,
            num_buckets=num_buckets,
        )
        if not means or not is_master():
            return means
        payload: dict[str, Any] = dict(means)
        if also_log_counts:
            for k, c in counts.items():
                payload[k.replace("loss_t/", "loss_t_count/")] = c
        logger.log(payload, step=gstep, commit=False)
        return means


__all__: list[str] = ["MetricsMixin"]
