"""DCP (Distributed Checkpoint) save/load helpers for ``FullyShardTrainer``.

Two flavours:
    - :func:`dcp_save` / :func:`dcp_load` — full model + optimizer + scheduler.
    - :func:`dcp_save_lora` / :func:`dcp_load_lora` — LoRA-only filter applied
      to the model state, full optimizer state. Reuses the
      :func:`lora_only_filter` predicate from policies.

Scheduler state is small Python state, so it's pickled by rank 0 only as
``scheduler.pkl``. Metadata (free-form dict) is also rank-0 pickle.

Layout on disk::

    {ckpt_path}/
      .metadata            (DCP marker for the full model save)
      {n}_.distcp          (DCP shards)
      scheduler.pkl        (rank-0 only)
      metadata.pkl         (rank-0 only)

LoRA layout::

    {ckpt_path}/
      lora/.metadata       (DCP marker for the LoRA filtered save)
      lora/{n}_.distcp
      optim/.metadata
      optim/{n}_.distcp
      scheduler.pkl
      metadata.pkl
"""
from __future__ import annotations

import gc
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from bbml.fsdp.dist import RunState, ModelOnlyState, is_master, master_print
from bbml.fsdp.policies import lora_only_filter


def _safe_barrier() -> None:
    """``dist.barrier`` if a process group is initialised, else no-op."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _gc_sync() -> None:
    """GC + CUDA empty + barrier. Used between heavy DCP write phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _safe_barrier()


def dcp_save(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    *,
    ckpt_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model + optimizer (+ optional scheduler / metadata) via DCP.

    Collective: every rank must call this with the SAME ``ckpt_path``. The
    scheduler is rank-0 pickle (matches the upstream recipe), but the DCP
    shards are written from every rank.
    """
    ckpt_path = Path(ckpt_path)
    start = time.time()
    _safe_barrier()

    if is_master():
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if scheduler is not None:
            with open(ckpt_path / "scheduler.pkl", "wb") as f:
                pickle.dump(scheduler.state_dict(), f)
        with open(ckpt_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata or {}, f)
    _gc_sync()

    state = {"run": RunState(model, optimizer)}
    dcp.save(state, checkpoint_id=str(ckpt_path))
    del state
    _gc_sync()

    master_print(f"[dcp_save] saved {ckpt_path} in {time.time() - start:.2f}s")


def dcp_load(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    *,
    ckpt_path: str | Path,
) -> dict[str, Any]:
    """Load model + optimizer (+ optional scheduler) via DCP.

    Returns the metadata dict (empty if no ``metadata.pkl`` is present).
    """
    ckpt_path = Path(ckpt_path)
    master_print(f"[dcp_load] loading {ckpt_path}")

    state = {"run": RunState(model, optimizer)}
    dcp.load(state, checkpoint_id=str(ckpt_path))
    del state
    _gc_sync()

    if scheduler is not None:
        scheduler_pkl = ckpt_path / "scheduler.pkl"
        if scheduler_pkl.exists():
            with open(scheduler_pkl, "rb") as f:
                scheduler.load_state_dict(pickle.load(f))

    metadata: dict[str, Any] = {}
    meta_path = ckpt_path / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    return metadata


def dcp_save_model_only(
    model: nn.Module,
    *,
    ckpt_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Lightweight DCP save: model weights only (no optimizer / scheduler).

    Intended for frequent intermediate snapshots; not resumable on its own.
    """
    ckpt_path = Path(ckpt_path)
    start = time.time()
    _safe_barrier()

    if is_master():
        ckpt_path.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata or {}, f)
    _gc_sync()

    state = {"run": ModelOnlyState(model)}
    dcp.save(state, checkpoint_id=str(ckpt_path))
    del state
    _gc_sync()

    master_print(f"[dcp_save_model_only] saved {ckpt_path} in {time.time() - start:.2f}s")


def dcp_load_model_only(
    model: nn.Module,
    *,
    ckpt_path: str | Path,
) -> dict[str, Any]:
    """Load model-only weights from a DCP checkpoint written by
    :func:`dcp_save_model_only`. Returns the metadata dict (empty if absent).
    """
    ckpt_path = Path(ckpt_path)
    master_print(f"[dcp_load_model_only] loading {ckpt_path}")

    state = {"run": ModelOnlyState(model)}
    dcp.load(state, checkpoint_id=str(ckpt_path))
    del state
    _gc_sync()

    metadata: dict[str, Any] = {}
    meta_path = ckpt_path / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    return metadata


def dcp_save_lora(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    *,
    ckpt_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save LoRA-only model state + optimizer + scheduler via DCP.

    Model state is filtered to keys matching :func:`lora_only_filter` (i.e.
    keys containing ``lora_``). The optimizer state already only covers
    requires_grad params (the LoRA adapter weights). Layout on disk:
    ``{ckpt_path}/lora/`` for the filtered model, ``{ckpt_path}/optim/`` for
    the optimizer.
    """
    ckpt_path = Path(ckpt_path)
    start = time.time()
    _safe_barrier()

    if is_master():
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if scheduler is not None:
            with open(ckpt_path / "scheduler.pkl", "wb") as f:
                pickle.dump(scheduler.state_dict(), f)
        with open(ckpt_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata or {}, f)
    _gc_sync()

    sd = model.state_dict()
    lora_sd = {k: v for k, v in sd.items() if lora_only_filter(k)}
    dcp.save(lora_sd, storage_writer=FileSystemWriter(ckpt_path / "lora"))
    del sd, lora_sd
    _gc_sync()

    od = optimizer.state_dict()
    dcp.save(od, storage_writer=FileSystemWriter(ckpt_path / "optim"))
    del od
    _gc_sync()

    master_print(f"[dcp_save_lora] saved {ckpt_path} in {time.time() - start:.2f}s")


def dcp_load_lora(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    *,
    ckpt_path: str | Path,
) -> dict[str, Any]:
    """Load LoRA-only model state + optimizer (+ scheduler) from a DCP ckpt.

    Falls back to a legacy full-model layout (``{ckpt_path}/model``) when no
    ``lora/`` subdir is present, mirroring the upstream recipe.
    """
    ckpt_path = Path(ckpt_path)
    master_print(f"[dcp_load_lora] loading {ckpt_path}")

    has_lora = (ckpt_path / "lora").exists()
    has_full = (ckpt_path / "model").exists()
    if has_lora:
        full_sd = model.state_dict()
        lora_sd = {k: v for k, v in full_sd.items() if lora_only_filter(k)}
        dcp.load(lora_sd, storage_reader=FileSystemReader(ckpt_path / "lora"))
        model.load_state_dict(full_sd)
        del full_sd, lora_sd
    elif has_full:
        mstates = model.state_dict()
        dcp.load(mstates, storage_reader=FileSystemReader(ckpt_path / "model"))
        model.load_state_dict(mstates)
        del mstates
    else:
        raise FileNotFoundError(
            f"No 'lora/' or 'model/' subdir found in {ckpt_path}; cannot load LoRA checkpoint."
        )
    _gc_sync()

    optstates = optimizer.state_dict()
    dcp.load(optstates, storage_reader=FileSystemReader(ckpt_path / "optim"))
    optimizer.load_state_dict(optstates)
    del optstates
    _gc_sync()

    if scheduler is not None:
        scheduler_pkl = ckpt_path / "scheduler.pkl"
        if scheduler_pkl.exists():
            with open(scheduler_pkl, "rb") as f:
                scheduler.load_state_dict(pickle.load(f))

    metadata: dict[str, Any] = {}
    meta_path = ckpt_path / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    return metadata


def strip_heavy_ckpt(ckpt_dir: str | Path, is_lora: bool = False) -> None:
    """Strip DCP shards + scheduler so the dir is no longer resumable.

    The DCP ``.metadata`` marker is deleted FIRST so an interrupted strip
    still reads as "rotated" (no marker) rather than "broken" (marker present
    but shards missing). This is the same ordering used upstream.

    Args:
        ckpt_dir: checkpoint directory to strip.
        is_lora: if True, strip the LoRA layout (``lora/`` + ``optim/``); else
            strip the full layout (``{n}_.distcp`` + ``.metadata`` at root).
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return

    if is_lora:
        marker = ckpt_dir / "lora" / ".metadata"
        if marker.exists():
            marker.unlink()
        for sub in ("lora", "optim"):
            d = ckpt_dir / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    else:
        marker = ckpt_dir / ".metadata"
        if marker.exists():
            marker.unlink()
        for p in ckpt_dir.glob("*.distcp"):
            p.unlink()

    scheduler_pkl = ckpt_dir / "scheduler.pkl"
    if scheduler_pkl.exists():
        scheduler_pkl.unlink()


def rotate_keep_last(
    parent_dir: str | Path,
    keep_last: int,
    prefix: str = "step_",
    *,
    strip_heavy: bool = True,
    is_lora: bool = False,
) -> None:
    """Keep only the latest ``keep_last`` step dirs under ``parent_dir``.

    Older step dirs have their DCP shards stripped (so they remain available
    for eval / inference if a consolidated export was also written) when
    ``strip_heavy=True``. When ``strip_heavy=False`` the dirs are deleted
    outright. Master-only operation; safe to call from any rank but a no-op
    when called from rank != 0.
    """
    if not is_master():
        return
    parent = Path(parent_dir)
    if not parent.exists():
        return
    step_dirs = sorted(
        (d for d in parent.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda d: int(d.name.removeprefix(prefix)) if d.name.removeprefix(prefix).isdigit() else -1,
    )
    if len(step_dirs) <= keep_last:
        return
    for old in step_dirs[:-keep_last]:
        if strip_heavy:
            strip_heavy_ckpt(old, is_lora=is_lora)
        else:
            shutil.rmtree(old, ignore_errors=True)


__all__: list[str] = [
    "dcp_save",
    "dcp_load",
    "dcp_save_model_only",
    "dcp_load_model_only",
    "dcp_save_lora",
    "dcp_load_lora",
    "strip_heavy_ckpt",
    "rotate_keep_last",
]
