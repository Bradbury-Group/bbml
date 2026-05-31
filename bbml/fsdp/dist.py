"""Distributed bootstrap + FSDP-aware utilities.

Ported from ``pretrain-drawing/src/krea_pretrain/dist.py`` (the canonical
reference shared across the pretrain-{drawing,opd,sliders} repos). Only the
pieces needed by ``FullyShardTrainer`` are surfaced here; EMA / safetensors
export are intentionally out of scope for round 1.

Load-bearing items reproduced verbatim:
    - 30-min NCCL timeout in ``init_process_group``.
    - ``device_id=cuda:LOCAL_RANK`` in ``init_process_group``.
    - ``_normalize_export_key`` strips ``_orig_mod`` and
      ``_checkpoint_wrapped_module`` prefixes inserted by ``torch.compile`` and
      ``checkpoint_wrapper`` respectively.
    - ``_build_consolidated_export_state_dict`` calls ``.full_tensor()`` on
      every DTensor — this is a collective and MUST be invoked on every rank.

Deferred to round 2:
    EMA / teacher param-copy (``copy_params``) with LoRA fp32-stored /
    bf16-forward dtype handling — this is the EMA update primitive used by
    pretrain-{drawing,opd,sliders}. Reference:
    ``pretrain-sliders/src/krea_pretrain/dist.py:466-520``
    (``copy_params`` + ``emaupdate_lora`` + ``normalize_model_key``). Lands
    alongside the EMA / teacher trainer hooks in round 2.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor


def get_global_rank() -> int:
    """Return the global rank ``RANK`` set by torchrun. 0 outside a torchrun env."""
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """Return the local (per-node) rank ``LOCAL_RANK`` set by torchrun."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return the world size ``WORLD_SIZE``. 1 outside a torchrun env."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_master() -> bool:
    """True iff this is global rank 0."""
    return get_global_rank() == 0


def is_local_master() -> bool:
    """True iff this is local rank 0 (one per node)."""
    return get_local_rank() == 0


def master_print(*args: Any, **kwargs: Any) -> None:
    """``print`` that only emits on global rank 0."""
    if is_master():
        print(*args, **kwargs)


def setup_dist(seed: int = 42) -> None:
    """Initialize the NCCL process group + per-rank seeds.

    Load-bearing flags:
        - ``backend="nccl"`` is fixed; we never run FSDP over gloo.
        - ``timeout=timedelta(minutes=30)`` accommodates slow ranks during
          in-training sampling collectives (default 10 min has tripped runs).
        - ``device_id=cuda:LOCAL_RANK`` so NCCL knows which device this rank
          owns at init time — required for eager-init paths.

    Always followed by a ``dist.barrier()`` so all ranks have a synchronised
    starting point before any subsequent collective.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(get_local_rank())
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30),
            device_id=torch.device(f"cuda:{get_local_rank()}"),
        )
    dist.barrier()


def cleanup_dist() -> None:
    """Destroy the process group. Safe to call when not initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """True iff we're running under a torchrun world (world_size > 1)."""
    return get_world_size() > 1


@dataclass
class ParallelLayout:
    """Parallel-axis layout for HSDP/TP-friendly seed derivation.

    Mirrors ``krea_pretrain.dist.ParallelLayout`` (the canonical reference in
    pretrain-{drawing,opd,sliders}). ``shard`` is the FSDP shard axis,
    ``replicate`` is the HSDP replicate axis, ``tp`` is tensor parallel.

    Data-parallelism semantics (verified against canonical):

        - ``get_dp_rank() = global_rank // tp``. The TP axis is the only one
          that collapses ranks into a single DP slot; replicate × shard ranks
          all see independent data batches even though they share model
          replicas (HSDP) or shards (FSDP).
        - ``get_dp_size() = world_size // tp``. Same reasoning — HSDP's
          replicate axis is fully data-parallel.
        - Pure FSDP (``tp=1``, ``replicate=1``): ``dp_size == world_size``.
        - HSDP (``tp=1``, ``replicate>1``): ``dp_size == world_size`` still;
          each (replicate, shard) coordinate is a distinct DP slot.
        - TP-only (``tp>1``, ``replicate=1``): ``dp_size = world_size // tp``;
          TP groups read the same batch.

    ``base`` is the seed root; per-rank seed = ``base + dp_rank * 10_000``
    to give distinct noise per data-parallel rank without collisions.
    """

    tp: int = 1
    shard: int = 1
    replicate: int = 1
    base: int = 42

    def get_dp_rank(self) -> int:
        """Data-parallel rank: ``global_rank // tp``.

        Replicate axis does NOT collapse DP rank — HSDP replicas process
        distinct batches. Only TP groups share a DP slot.
        """
        return get_global_rank() // max(1, self.tp)

    def get_dp_size(self) -> int:
        """Data-parallel world size: ``world_size // tp``.

        Independent of ``replicate``; see :meth:`get_dp_rank` for the
        reasoning.
        """
        return get_world_size() // max(1, self.tp)

    def get_seed(self) -> int:
        """Per-rank base seed for data sampling (not the noise seed)."""
        return self.base + self.get_dp_rank()

    def get_noise_seed(self) -> int:
        """Per-rank seed for in-training noise/sampling.

        Spaced by 10_000 so that adjacent ranks don't collide in any reasonable
        per-step counter.
        """
        return self.base + self.get_dp_rank() * 10_000


def checkpoint_wrap(blk: nn.Module) -> nn.Module:
    """Wrap a block in ``torch.utils.checkpoint`` (activation checkpointing).

    MUST be invoked BEFORE ``fully_shard`` on the same block. Reversing the
    order breaks the per-block grad-ckpt boundary because ``fully_shard``
    replaces the forward and the checkpoint hook would never fire.
    """
    return checkpoint_wrapper(blk)


@torch.no_grad()
def _get_total_norm_fsdp(
    params: list[nn.Parameter], foreach: bool = False
) -> torch.Tensor:
    """Compute total grad norm; gather DTensor shards to full norm."""
    norm = torch.nn.utils.get_total_norm(params, foreach=foreach)
    if hasattr(norm, "full_tensor"):
        return norm.full_tensor()
    return norm


@torch.no_grad()
def clip_grad_norm_fsdp(
    parameters: list[nn.Parameter],
    max_norm: float,
    foreach: bool = False,
) -> torch.Tensor:
    """Clip grad norm across FSDP shards via a collective.

    Behaves like ``torch.nn.utils.clip_grad_norm_`` but consults the full
    (cross-rank) norm via DTensor's ``full_tensor()`` so the clipping factor
    is consistent on every rank. Returns the total norm as a regular tensor.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm_fsdp(grads, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=foreach)
    return total_norm


class RunState(Stateful):
    """DCP ``Stateful`` bundling model + optimizer for atomic save/load.

    Use with ``torch.distributed.checkpoint.save/load``; the resulting on-disk
    layout has ``run.model.*`` for the sharded model state and ``run.optim.*``
    for the optimizer state.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class ModelOnlyState(Stateful):
    """DCP ``Stateful`` carrying only the model. Same ``run.model.*`` layout as
    ``RunState`` so the on-disk shards are byte-identical to those written by
    a paired full save at the same step — only the optim shard is absent.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return {"model": get_model_state_dict(self.model)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(self.model, model_state_dict=state_dict["model"])


def _normalize_export_key(name: str) -> str:
    """Strip ``_orig_mod`` and ``_checkpoint_wrapped_module`` wrapper segments.

    ``torch.compile`` inserts ``_orig_mod`` and ``checkpoint_wrapper`` inserts
    ``_checkpoint_wrapped_module`` into state-dict keys. Both prefixes need to
    be removed when matching keys against checkpoints written without the
    wrappers (or when consolidating for safetensors export).
    """
    wrapper_parts = {"_orig_mod", "_checkpoint_wrapped_module"}
    return ".".join(part for part in name.split(".") if part not in wrapper_parts)


def _build_consolidated_export_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Gather DTensor shards to full CPU tensors; strip FSDP/checkpoint wrappers.

    MUST be called from every rank because ``full_tensor()`` is a collective.
    The returned dict is identical on every rank but is intended to be written
    out by rank 0 only.
    """
    consolidated: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        export_name = _normalize_export_key(name)
        if export_name in consolidated and export_name != name:
            raise ValueError(
                f"State-dict key collision after wrapper normalization: {name} -> {export_name}"
            )
        if isinstance(value, DTensor):
            consolidated[export_name] = value.full_tensor().detach().cpu()
        else:
            consolidated[export_name] = value.detach().cpu()
    return consolidated


__all__: list[str] = [
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "is_master",
    "is_local_master",
    "master_print",
    "setup_dist",
    "cleanup_dist",
    "is_distributed",
    "ParallelLayout",
    "checkpoint_wrap",
    "clip_grad_norm_fsdp",
    "RunState",
    "ModelOnlyState",
    "_normalize_export_key",
    "_build_consolidated_export_state_dict",
]
