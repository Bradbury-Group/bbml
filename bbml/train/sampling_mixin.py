"""FSDP-safe in-training sampling primitives.

Ported from the ``pretrain-drawing`` _sample_and_log recipe. The contract:

    - **Every rank participates.** Sampling is a collective; rank-conditional
      ``dist.*`` calls deadlock under shard>1. The provided ``sample_fn``
      runs on every rank with a per-rank seed and a slice of the inputs.
    - **Error-tolerance:** ``sample_fn`` is wrapped in try/except + zero-fill
      so a single rank's exception doesn't hang the ``all_gather_object``.
      Failures are explicitly logged on the failing rank.
    - **Master logs.** Only rank-0 emits ``wandb.Image`` payloads and uses
      ``commit=False`` so the images batch into the current gstep upload.

In single-process / non-distributed contexts (``world_size == 1``), the
collective collapses to a local list; the mixin still works.
"""
from __future__ import annotations

import logging as _stdlib_logging
import math
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist

from bbml import logger
from bbml.fsdp.dist import (
    get_global_rank,
    get_world_size,
    is_distributed,
    is_master,
    master_print,
)


def _safe_all_gather_object(payload: Any) -> list[Any]:
    """``dist.all_gather_object`` if distributed; else wrap payload as a list.

    Keeps the collective shape consistent regardless of world_size so callers
    can iterate the result uniformly.
    """
    if not is_distributed():
        return [payload]
    gathered: list[Any] = [None] * get_world_size()
    dist.all_gather_object(gathered, payload)
    return gathered


class SamplingMixin:
    """In-training preview sampling for FSDP-aware trainers.

    Subclasses should expose ``self.train_config`` (for fallback gstep) and
    typically also inherit ``MetricsMixin`` for scalar logging adjacent to
    image previews.
    """

    def fsdp_safe_sample(
        self,
        sample_fn: Callable[..., Any],
        *,
        num_samples: int,
        gstep: int,
        prompts: Iterable[str] | None = None,
        sample_kwargs: Optional[dict[str, Any]] = None,
        key_prefix: str = "sample/preview",
        base_seed: int = 42,
        zero_fill_shape: tuple[int, ...] | None = None,
    ) -> list[Any] | None:
        """Run ``sample_fn`` on every rank, gather, master-log to wandb.

        Args:
            sample_fn: callable invoked once per rank with kwargs:
                ``per_rank`` (int), ``seed`` (int), ``prompts`` (list[str] |
                None), plus any ``sample_kwargs`` the caller provides.
                MUST return a numpy array of shape ``[per_rank, C, H, W]``
                in ``[0, 1]`` (or any image-like array uniform across ranks).
                The shape MUST be identical on every rank to keep
                ``all_gather_object`` aligned.
            num_samples: total samples to log after gathering (output is
                truncated to this count).
            gstep: global step for wandb log.
            prompts: optional global prompt list; sliced per-rank
                contiguously (pad with empty strings when short).
            sample_kwargs: forwarded to ``sample_fn`` as kwargs.
            key_prefix: wandb metric key for the resulting Image list.
            base_seed: seed root; per-rank seed = ``base_seed + dp_rank *
                10_000`` so adjacent ranks never collide.
            zero_fill_shape: fallback shape used when ``sample_fn`` raises;
                if None, infers from a successful sibling rank via the
                gather (but this is rare and best-effort).

        Returns:
            On master: the list of gathered/truncated outputs.
            On non-master: None.
        """
        sample_kwargs = sample_kwargs or {}
        world = get_world_size()
        per_rank = max(1, math.ceil(num_samples / world))
        rank = get_global_rank()
        seed = base_seed + rank * 10_000

        rank_prompts: list[str] | None = None
        if prompts is not None:
            full = list(prompts)
            start = rank * per_rank
            end = start + per_rank
            rank_prompts = full[start:end]
            if len(rank_prompts) < per_rank:
                rank_prompts = rank_prompts + [""] * (per_rank - len(rank_prompts))

        local_payload: dict[str, Any]
        try:
            local_out = sample_fn(
                per_rank=per_rank,
                seed=seed,
                prompts=rank_prompts,
                **sample_kwargs,
            )
            local_payload = {"imgs": local_out, "prompts": rank_prompts or []}
        except Exception as exc:  # noqa: BLE001 — intentionally broad; collective alignment matters
            _stdlib_logging.exception(
                "[SamplingMixin] sample_fn failed on rank %d: %r — emitting zero-fill so collectives align",
                rank,
                exc,
            )
            if zero_fill_shape is not None:
                local_imgs = np.zeros((per_rank, *zero_fill_shape), dtype=np.float32)
            else:
                # Best-effort fallback: 3-channel 64x64 black image. Better
                # than crashing the run; downstream eval should ignore these.
                local_imgs = np.zeros((per_rank, 3, 64, 64), dtype=np.float32)
            local_payload = {"imgs": local_imgs, "prompts": rank_prompts or [""] * per_rank}

        gathered = _safe_all_gather_object(local_payload)
        if not is_master():
            return None

        all_imgs: list[Any] = []
        all_prompts: list[str] = []
        for piece in gathered:
            if piece is None:
                continue
            imgs = piece["imgs"]
            ps = piece.get("prompts", []) or []
            for j in range(len(imgs)):
                all_imgs.append(imgs[j])
                all_prompts.append(ps[j] if j < len(ps) else "")
        all_imgs = all_imgs[:num_samples]
        all_prompts = all_prompts[:num_samples]

        try:
            from PIL import Image

            try:
                import wandb  # type: ignore
            except ImportError:
                wandb = None  # type: ignore

            wandb_imgs = []
            for arr, caption in zip(all_imgs, all_prompts):
                if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr_hw_c = arr.transpose(1, 2, 0)
                else:
                    arr_hw_c = arr
                if isinstance(arr_hw_c, np.ndarray):
                    arr_hw_c = (arr_hw_c * 255.0).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(arr_hw_c.squeeze())
                else:
                    img = arr_hw_c
                if wandb is not None:
                    wandb_imgs.append(wandb.Image(img, caption=caption))
                else:
                    wandb_imgs.append(img)
            logger.log({key_prefix: wandb_imgs}, step=gstep, commit=False)
        except Exception:  # noqa: BLE001
            # Raise loud — if sampling is enabled and the master-only log fails,
            # the user wants to know. Collective alignment is already complete
            # by this point (this block is past the all_gather_object), so
            # raising on master does not cause a NCCL hang.
            _stdlib_logging.exception("[SamplingMixin] master log of preview failed")
            raise

        return all_imgs

    @staticmethod
    def build_triptychs(
        orig: torch.Tensor,
        x_t: torch.Tensor,
        v_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``[orig | x_t | x_hat]`` triptychs without an extra forward.

        ``x_hat = x_t - t * v_pred`` recovers the (one-step) clean prediction
        from a v-parameterised flow forward — no second model call needed.

        Args:
            orig: ground-truth clean batch, shape ``[B, C, H, W]``.
            x_t: noised batch at timestep ``t``, shape ``[B, C, H, W]``.
            v_pred: model output (velocity), shape ``[B, C, H, W]``.
            t: timestep per sample, shape ``[B]`` in ``[0, 1]``.

        Returns:
            A ``[B, C, H, 3*W]`` tensor with the three panels concatenated
            along the width axis.
        """
        if t.ndim == 1:
            t_b = t.view(-1, 1, 1, 1)
        else:
            t_b = t
        x_hat = x_t - t_b * v_pred
        return torch.cat([orig, x_t, x_hat], dim=-1)


__all__: list[str] = ["SamplingMixin"]
