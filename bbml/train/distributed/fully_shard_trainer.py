"""FSDP2-native ``Trainer`` for bbml.

Bootstraps from torchrun env vars (``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``),
builds a ``DeviceMesh`` from ``ParallelismConfig``, asks the ``Foundation`` to
parallelise itself in-place via :meth:`Foundation.parallelise`, then drives a
standard train loop with FSDP-aware logging / sampling mixins and DCP-based
checkpoints.

Load-bearing ordering invariants:
    1. ``setup_dist`` (NCCL init) BEFORE any CUDA model allocation.
    2. ``foundation.parallelise(mesh, policy=...)`` BEFORE optimizer build —
       FSDP2 requires the optimizer to see DTensor-typed params.
    3. ``DistributedSampler`` wraps the train dataloader's dataset, using the
       data-parallel size from ``ParallelLayout`` (not raw world size, to
       leave room for future TP).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from bbml import logger
from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.datapipe import DataPipe
from bbml.core.foundation import Foundation
from bbml.core.interfaces import Trainable
from bbml.core.trainer import Trainer
from bbml.fsdp.dcp import (
    dcp_load,
    dcp_load_lora,
    dcp_load_model_only,
    dcp_save,
    dcp_save_lora,
    dcp_save_model_only,
    rotate_keep_last,
)
from bbml.fsdp.dist import (
    ParallelLayout,
    cleanup_dist,
    clip_grad_norm_fsdp,
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_distributed,
    is_master,
    master_print,
    setup_dist,
)
from bbml.fsdp.parallelise import init_mesh
from bbml.fsdp.policies import default_mp_policy
from bbml.train.metrics_mixin import MetricsMixin
from bbml.train.optim_factory import (
    build_lr_scheduler_from_config,
    build_optimizer_from_config,
)
from bbml.train.sampling_mixin import SamplingMixin


class FullyShardTrainer(Trainer, MetricsMixin, SamplingMixin):
    """FSDP2-native trainer.

    Composes :class:`MetricsMixin` and :class:`SamplingMixin` for FSDP-safe
    scalar / loss-bucket / image logging. Save/load is DCP-based via
    :mod:`bbml.fsdp.dcp`; ``Foundation.save`` is NOT touched (this keeps the
    legacy delta-checkpointing path for the accelerate trainer fully intact).

    Checkpoint-format restriction:
        ``checkpointing.format="delta"`` (the legacy ``Foundation.save`` path)
        is REJECTED under FSDP2 multi-rank (``world_size > 1``) because
        ``model.state_dict()`` returns DTensor shards and
        :func:`bbml.core.checkpointing.extract_delta_state` /
        :func:`save_delta` would silently serialise only the local shard.
        Use ``format="dcp"`` (full DCP) or ``format="dcp_lora"`` (LoRA-only
        DCP) for FSDP2 multi-rank. Single-process (``world_size == 1``) keeps
        ``"delta"`` available as a convenience. The :meth:`train` bootstrap
        raises ``RuntimeError`` early when the mismatch is detected.

    Required ``train_config`` knobs (over and above ``TrainerConfig`` base):
        - ``parallelism`` (``ParallelismConfig`` dict, optional). Default
          assumes pure FSDP across ``world_size``.
        - ``sampling`` (``SamplingConfig`` dict, optional). Default disabled.
        - ``metrics`` (``MetricsConfig`` dict, optional). Default
          ``loss_buckets=True``, ``num_buckets=10``.
        - ``checkpointing`` (``CheckpointingConfig`` dict, optional). Default
          falls back to ``Foundation.save`` (legal only single-rank).

    Args:
        model: a ``Foundation`` (or ``Trainable``) that has a working
            ``parallelise(mesh, policy=...)`` override.
        train_config: trainer config; extra fields read by this class are
            accessed via ``getattr`` so existing configs keep validating.
        train_datapipe: train data pipe; its underlying dataset is wrapped
            with a ``DistributedSampler`` keyed off ``ParallelLayout``.
        val_datapipe: validation pipe (optional).
        test_datapipe: test pipe (optional).
        seed: process-group seed; default 42.
        owns_process_group: if True (default), this trainer calls
            :func:`setup_dist` in ``train()`` and :func:`cleanup_dist` on
            exit. Set False when the caller bootstraps the PG externally.
    """

    def __init__(
        self,
        model: Trainable | Foundation,
        train_config: TrainerConfig,
        train_datapipe: DataPipe,
        val_datapipe: DataPipe | None = None,
        test_datapipe: DataPipe | None = None,
        *,
        seed: int = 42,
        owns_process_group: bool = True,
    ):
        super().__init__(model, train_config, train_datapipe, val_datapipe, test_datapipe)
        self._seed = seed
        self._owns_pg = owns_process_group
        self.mesh = None
        self.layout: ParallelLayout | None = None
        self.mp_policy: MixedPrecisionPolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.device: torch.device | None = None
        self._test_skip_logged: bool = False

    # ------------------------------------------------------------------ #
    # config helpers
    # ------------------------------------------------------------------ #
    def _parallelism_cfg(self) -> dict[str, Any]:
        cfg = getattr(self.train_config, "parallelism", None)
        if cfg is None:
            return {}
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        return dict(cfg)

    def _sampling_cfg(self) -> dict[str, Any]:
        cfg = getattr(self.train_config, "sampling", None)
        if cfg is None:
            return {"enabled": False}
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        return dict(cfg)

    def _metrics_cfg(self) -> dict[str, Any]:
        cfg = getattr(self.train_config, "metrics", None)
        if cfg is None:
            return {"loss_buckets": True, "num_buckets": 10, "train_preview_every": None, "train_preview_n": 4}
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        return dict(cfg)

    def _ckpt_cfg(self) -> dict[str, Any]:
        cfg = getattr(self.train_config, "checkpointing", None)
        if cfg is None:
            return {"format": "delta", "every": 500, "keep_last": None, "rotate_strip_heavy": False, "model_only_every": None}
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump()
        return dict(cfg)

    # ------------------------------------------------------------------ #
    # bootstrap
    # ------------------------------------------------------------------ #
    def _build_mesh(self) -> tuple[torch.distributed.device_mesh.DeviceMesh, ParallelLayout]:
        """Build the FSDP/HSDP/TP device mesh + layout from config.

        Resolution:
            - ``replicate > 1`` -> HSDP mesh ``(replicate, shard)``.
            - ``tensor_parallel > 1`` -> ``(shard, tp)`` two-axis mesh.
            - default -> pure FSDP ``(world_size,)`` mesh.

        TP > 1 + replicate > 1 simultaneously is not implemented in round 1
        (raises ``NotImplementedError``); the upstream sweeps top out at 16
        GPUs with pure FSDP so this is a deliberate scope gate.
        """
        pcfg = self._parallelism_cfg()
        tp = int(pcfg.get("tensor_parallel", 1))
        replicate = int(pcfg.get("replicate", 1))
        world = get_world_size()
        mesh_dim_names = tuple(pcfg.get("mesh_dim_names", ("shard",)))

        if tp > 1 and replicate > 1:
            raise NotImplementedError(
                "tensor_parallel>1 + replicate>1 not implemented yet "
                "(round 1 scope: pure FSDP and HSDP). "
                "Use accelerate trainer or extend FullyShardTrainer._build_mesh."
            )

        if replicate > 1:
            if world % replicate != 0:
                raise ValueError(
                    f"world_size={world} not divisible by replicate={replicate}"
                )
            shard = world // replicate
            mesh = init_mesh((replicate, shard), ("replicate", "shard"))
            layout = ParallelLayout(tp=1, shard=shard, replicate=replicate, base=self._seed)
            return mesh, layout

        if tp > 1:
            if world % tp != 0:
                raise ValueError(f"world_size={world} not divisible by tp={tp}")
            shard = world // tp
            mesh = init_mesh((shard, tp), ("shard", "tp"))
            layout = ParallelLayout(tp=tp, shard=shard, replicate=1, base=self._seed)
            return mesh, layout

        names = mesh_dim_names if len(mesh_dim_names) == 1 else ("shard",)
        mesh = init_mesh((world,), names)
        layout = ParallelLayout(tp=1, shard=world, replicate=1, base=self._seed)
        return mesh, layout

    def _wrap_train_dataloader(self) -> DataLoader:
        """Wrap the train datapipe's underlying dataset with a ``DistributedSampler``.

        ``DataPipe.get_loader`` is bypassed for the sampler-wired loader; we
        re-use the pipe's ``collate_fn`` / ``batch_size`` / ``num_workers``
        attributes so the same transforms apply. ``drop_last`` comes from
        ``train_config.drop_last_train`` (default True); set False for small
        datasets where the dropped tail meaningfully reduces coverage.
        """
        if self.layout is None:
            raise RuntimeError("layout not initialised; called _wrap_train_dataloader too early")
        dp_size = self.layout.get_dp_size()
        dp_rank = self.layout.get_dp_rank()
        drop_last = bool(getattr(self.train_config, "drop_last_train", True))
        sampler = DistributedSampler(
            self.train_datapipe,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=getattr(self.train_datapipe, "shuffle", True),
            drop_last=drop_last,
            seed=self._seed,
        )
        self._train_sampler = sampler
        return DataLoader(
            self.train_datapipe,
            batch_size=self.train_datapipe.batch_size,
            sampler=sampler,
            collate_fn=self.train_datapipe.collate_fn,
            num_workers=self.train_datapipe.num_workers or 0,
        )

    def _setup_logger(self) -> None:
        if not is_master():
            return
        if self.train_config.logging_backends is not None:
            logger.start(
                self.train_config.logging_backends,
                **self.train_config.model_dump(),
            )

    # ------------------------------------------------------------------ #
    # train loop
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        if self._owns_pg:
            setup_dist(self._seed)
        elif not dist.is_initialized():
            raise RuntimeError(
                "FullyShardTrainer launched with owns_process_group=False but "
                "no process group is initialised. Call setup_dist() first."
            )

        self.device = torch.device(f"cuda:{get_local_rank()}")
        master_print(
            f"[FullyShardTrainer] rank={get_global_rank()} local={get_local_rank()} "
            f"world={get_world_size()} device={self.device}"
        )

        if not isinstance(self.model, Foundation):
            raise TypeError(
                "FullyShardTrainer requires a Foundation subclass (got "
                f"{type(self.model).__name__}). Foundation.parallelise() is the "
                "hook used to wire FSDP2."
            )

        # Reject delta save format under FSDP2 multi-rank.
        # extract_delta_state -> filter_state_dict_to_names -> .to('cpu') yields
        # DTensor LOCAL shards (silent corruption). dcp / dcp_lora gather correctly.
        ckpt_cfg_early = self._ckpt_cfg()
        fmt_early = ckpt_cfg_early.get("format", "delta")
        if fmt_early == "delta" and get_world_size() > 1:
            raise RuntimeError(
                f"checkpointing.format='delta' is not safe under FSDP2 multi-rank "
                f"(world_size={get_world_size()}). Foundation.save() iterates "
                "model.state_dict() whose tensors are DTensor shards; "
                "bbml.core.checkpointing.save_delta calls .to('cpu') on each shard "
                "and writes only the local fragment. "
                "Switch to checkpointing.format='dcp' (full model + optimizer) "
                "or 'dcp_lora' (LoRA-only) for FSDP2 multi-rank training."
            )

        self.model.train()
        self.model.to(device=self.device)

        self.mesh, self.layout = self._build_mesh()
        self.mp_policy = default_mp_policy()

        # Foundation owns the FSDP wrap (so it can reach its block list).
        self.model.parallelise(self.mesh, policy=self.mp_policy)

        # FSDP2 invariant: optimizer AFTER fully_shard.
        self.optimizer = build_optimizer_from_config(self.model, self.train_config)
        self.lr_scheduler = build_lr_scheduler_from_config(self.optimizer, self.train_config, model=self.model)

        self._setup_logger()

        if self.train_config.load_path is not None:
            self.load(self.train_config.load_path)

        train_loader = self._wrap_train_dataloader()
        grad_clip = getattr(self.train_config, "grad_clip_norm", None)
        sampling_cfg = self._sampling_cfg()
        metrics_cfg = self._metrics_cfg()
        ckpt_cfg = self._ckpt_cfg()

        for epoch in range(self.train_config.train_epochs):
            self._train_sampler.set_epoch(epoch)
            if hasattr(self.train_datapipe, "set_epoch"):
                self.train_datapipe.set_epoch(epoch)

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}",
                disable=not is_master(),
            )

            for batch_num, batch in pbar:
                step_start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)

                batch["_bbml"] = {
                    "step": self.train_config.step,
                    "batch_num": batch_num,
                    "epoch": epoch,
                    "split": "train",
                }

                result = self.model(batch)
                if isinstance(result, tuple):
                    loss, extra_metrics = result
                else:
                    loss, extra_metrics = result, {}

                loss.backward()

                if grad_clip is not None:
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    clip_grad_norm_fsdp(params, grad_clip)

                self.optimizer.step()
                self.lr_scheduler.step()

                step_time = time.perf_counter() - step_start

                # Reduce loss for logging
                reduced_loss = loss.detach()
                if is_distributed():
                    dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)

                self._log_train_step(
                    reduced_loss=reduced_loss,
                    extra_metrics=extra_metrics,
                    step_time=step_time,
                    batch_num=batch_num,
                    epoch=epoch,
                    metrics_cfg=metrics_cfg,
                    batch=batch,
                )

                if is_master():
                    pbar.set_postfix({"loss": float(reduced_loss.item())})

                self._do_val_test_save(
                    sampling_cfg=sampling_cfg,
                    ckpt_cfg=ckpt_cfg,
                )

                self.train_config.step += 1
                if (
                    self.train_config.max_training_steps is not None
                    and self.train_config.step >= self.train_config.max_training_steps
                ):
                    master_print(
                        f"[FullyShardTrainer] reached max_training_steps={self.train_config.max_training_steps}"
                    )
                    self._do_val_test_save(sampling_cfg=sampling_cfg, ckpt_cfg=ckpt_cfg, do_all=True)
                    if self._owns_pg:
                        cleanup_dist()
                    return

        self._do_val_test_save(sampling_cfg=sampling_cfg, ckpt_cfg=ckpt_cfg, do_all=True)
        if self._owns_pg:
            cleanup_dist()

    def _log_train_step(
        self,
        *,
        reduced_loss: torch.Tensor,
        extra_metrics: dict[str, Any],
        step_time: float,
        batch_num: int,
        epoch: int,
        metrics_cfg: dict[str, Any],
        batch: dict[str, Any],
    ) -> None:
        """Master-only loss-bucket log + primary train log.

        Wandb commit ordering: bucket metrics are logged FIRST with
        ``commit=False`` so they buffer onto the current ``gstep``; the
        primary ``train_loss`` log then commits with ``commit=True`` and
        flushes everything in one upload. Reversing the order attaches the
        bucket metrics to the NEXT step's commit, which is the bug the
        sister-repo pretrain-drawing pattern avoids.
        """
        if not is_master():
            return

        # 1. Bucket loss FIRST (commit=False) so it batches onto the
        # primary commit below.
        if metrics_cfg.get("loss_buckets", True):
            t = extra_metrics.get("t")
            lps = extra_metrics.get("loss_per_sample")
            train_mask = extra_metrics.get("train_mask")
            if isinstance(t, torch.Tensor) and isinstance(lps, torch.Tensor):
                self.log_loss_buckets(
                    t=t,
                    loss_per_sample=lps,
                    train_mask=train_mask if isinstance(train_mask, torch.Tensor) else None,
                    num_buckets=int(metrics_cfg.get("num_buckets", 10)),
                    gstep=self.train_config.step,
                )

        # 2. Primary log LAST (commit=True) — flushes buffered metrics for this gstep.
        # Per-sample tensors are excluded from the primary scalar log; they're
        # consumed by log_loss_buckets above and would clutter the wandb chart.
        _bucket_input_keys = {"loss_per_sample", "t", "train_mask"}
        learning_rates = {f"lr.{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())}
        log_metrics: dict[str, Any] = {
            "train_loss": float(reduced_loss.item()),
            "step": self.train_config.step,
            "batch_num": batch_num,
            "epoch": epoch,
            "timing/train_step_s": step_time,
            **learning_rates,
            **{k: v for k, v in extra_metrics.items() if k not in _bucket_input_keys},
        }
        logger.log(log_metrics, commit=True)

    # ------------------------------------------------------------------ #
    # val / test / save dispatch
    # ------------------------------------------------------------------ #
    def _do_val_test_save(
        self,
        *,
        sampling_cfg: dict[str, Any],
        ckpt_cfg: dict[str, Any],
        do_all: bool = False,
    ) -> None:
        step = self.train_config.step
        should_validate = (
            step > 0
            and self.train_config.check_step_trigger(step, self.train_config.validation_step_trigger)
        ) or do_all
        if should_validate:
            self.validate()
        if self.train_config.check_step_trigger(step, self.train_config.test_step_trigger) or do_all:
            # Skip when subclass hasn't overridden test() — otherwise the
            # base implementation's NotImplementedError fires during
            # end-of-training cleanup (do_all=True) and aborts before save().
            # Symmetric with the _maybe_sample self-detection pattern.
            if type(self).test is FullyShardTrainer.test:
                if not self._test_skip_logged and is_master():
                    master_print(
                        "[FullyShardTrainer] test() not overridden; skipping test phase. "
                        "Override test() to enable; see bbml.train.distributed.accelerate_trainer for the pattern."
                    )
                    self._test_skip_logged = True
            else:
                self.test()
        save_trigger_fired = (
            self.train_config.check_step_trigger(step, self.train_config.save_step_trigger) or do_all
        )
        if save_trigger_fired:
            self.save(self.train_config.output_dir)

        # In-training sampling on cadence (separate from save cadence).
        if sampling_cfg.get("enabled"):
            every = int(sampling_cfg.get("every", 0))
            if every > 0 and (step % every == 0 or do_all):
                self._maybe_sample(sampling_cfg)

        # Optional model-only DCP snapshot (lightweight, more frequent than full save).
        model_only_every = ckpt_cfg.get("model_only_every")
        if model_only_every and step > 0 and (step % int(model_only_every) == 0):
            self._save_model_only(ckpt_cfg)

    def _maybe_sample(self, sampling_cfg: dict[str, Any]) -> None:
        """Hook for subclasses; runs the in-training preview pass.

        Subclasses with a foundation-specific ``sample_fn`` MUST override
        and call ``self.fsdp_safe_sample(sample_fn, ...)``. We do not run
        sampling generically here because the sampling function is tightly
        coupled to the foundation's input/output models.

        Base behaviour:
            - If ``sampling_cfg["enabled"]`` is False / missing: return (no-op).
            - If ``sampling_cfg["enabled"]`` is True AND this method has not
              been overridden by a subclass: raise ``NotImplementedError``
              with a pointer to ``SamplingMixin``. This prevents the silent
              "configured sampling but got nothing" failure mode.
        """
        if not sampling_cfg.get("enabled"):
            return
        if type(self)._maybe_sample is FullyShardTrainer._maybe_sample:
            raise NotImplementedError(
                "FullyShardTrainer._maybe_sample is foundation-specific; "
                "subclass FullyShardTrainer and override _maybe_sample, then "
                "call self.fsdp_safe_sample(sample_fn, num_samples=..., "
                "gstep=self.train_config.step, prompts=..., ...). "
                "See bbml.train.sampling_mixin.SamplingMixin for the contract."
            )

    # ------------------------------------------------------------------ #
    # save / load
    # ------------------------------------------------------------------ #
    def save(self, save_path: str | Path) -> None:
        """Save checkpoint via the configured DCP format.

        Layout: ``{save_path}/{run_name}/step_{step}/`` (matches the
        accelerate trainer convention).
        """
        save_path = Path(save_path)
        run_name = self.train_config.name or "unnamed"
        step = self.train_config.step
        ckpt_dir = save_path / run_name / f"step_{step}"
        ckpt_cfg = self._ckpt_cfg()
        fmt = ckpt_cfg.get("format", "delta")

        metadata = self._build_save_metadata(step)

        if fmt == "delta":
            # Legacy delta path (Foundation.save). FSDP-aware via the wrap:
            # full_tensor() gather happens inside Foundation.save's state_dict
            # iteration. Master-only write.
            if is_master():
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.model.save(ckpt_dir)
                torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
                torch.save(self.lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        elif fmt == "dcp":
            dcp_save(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                ckpt_path=ckpt_dir,
                metadata=metadata,
            )
        elif fmt == "dcp_lora":
            dcp_save_lora(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                ckpt_path=ckpt_dir,
                metadata=metadata,
            )
        elif fmt == "dcp_model_only":
            raise ValueError(
                "checkpointing.format='dcp_model_only' is load-only. "
                "Model-only snapshots are written via the model_only_every cadence "
                "(internal call to dcp_save_model_only). Pair this format with a "
                "full save format (e.g. 'dcp' or 'dcp_lora') for the main save trigger."
            )
        else:
            raise ValueError(f"unknown checkpointing.format: {fmt!r}")

        keep_last = ckpt_cfg.get("keep_last")
        if keep_last is not None and keep_last > 0:
            rotate_keep_last(
                save_path / run_name,
                keep_last=int(keep_last),
                strip_heavy=bool(ckpt_cfg.get("rotate_strip_heavy", False)),
                is_lora=(fmt == "dcp_lora"),
            )

        master_print(f"[FullyShardTrainer] saved checkpoint to {ckpt_dir}")

    def _save_model_only(self, ckpt_cfg: dict[str, Any]) -> None:
        """Lightweight DCP snapshot — model weights only, no optim / sched."""
        save_path = Path(self.train_config.output_dir)
        run_name = self.train_config.name or "unnamed"
        step = self.train_config.step
        ckpt_dir = save_path / run_name / f"step_{step}_model_only"
        dcp_save_model_only(
            self.model,
            ckpt_path=ckpt_dir,
            metadata=self._build_save_metadata(step),
        )

    def _build_save_metadata(self, step: int) -> dict[str, Any]:
        return {
            "step": step,
            "epoch": None,  # not tracked at save time; populate in subclass if needed
            "run_name": self.train_config.name,
        }

    def load(self, load_path: str | Path) -> None:
        load_path = Path(load_path)
        ckpt_cfg = self._ckpt_cfg()
        fmt = ckpt_cfg.get("format", "delta")

        if fmt == "delta":
            self.model.load(load_path)
            optim_path = load_path / "optimizer.pt"
            if optim_path.exists():
                self.optimizer.load_state_dict(
                    torch.load(optim_path, map_location=self.device, weights_only=True)
                )
            lrs_path = load_path / "lr_scheduler.pt"
            if lrs_path.exists():
                self.lr_scheduler.load_state_dict(torch.load(lrs_path, weights_only=True))
        elif fmt == "dcp":
            dcp_load(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                ckpt_path=load_path,
            )
        elif fmt == "dcp_lora":
            dcp_load_lora(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                ckpt_path=load_path,
            )
        elif fmt == "dcp_model_only":
            dcp_load_model_only(self.model, ckpt_path=load_path)
        else:
            raise ValueError(f"unknown checkpointing.format: {fmt!r}")

        master_print(f"[FullyShardTrainer] loaded checkpoint from {load_path}")

    # ------------------------------------------------------------------ #
    # validate / test
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self) -> torch.Tensor:
        """Validation loop with weighted accumulation across ranks."""
        if self.val_datapipe is None:
            if is_master():
                warnings.warn("Validation DataPipe not provided, skipping")
            return torch.tensor(0.0)
        self.model.eval()
        try:
            dp_size = self.layout.get_dp_size() if self.layout else 1
            dp_rank = self.layout.get_dp_rank() if self.layout else 0
            val_sampler = DistributedSampler(
                self.val_datapipe,
                num_replicas=dp_size,
                rank=dp_rank,
                shuffle=False,
                drop_last=False,
            )
            val_loader = DataLoader(
                self.val_datapipe,
                batch_size=self.val_datapipe.batch_size,
                sampler=val_sampler,
                collate_fn=self.val_datapipe.collate_fn,
                num_workers=self.val_datapipe.num_workers or 0,
            )

            total_loss = torch.tensor(0.0, device=self.device)
            total_metrics: dict[str, torch.Tensor] = {}
            total_samples = torch.tensor(0, device=self.device, dtype=torch.long)

            for batch in tqdm(val_loader, desc="Validation", disable=not is_master()):
                batch["_bbml"] = {"step": self.train_config.step, "split": "validation"}
                batch_size = self._infer_batch_size(batch)
                result = self.model(batch)
                if isinstance(result, tuple):
                    loss, extra_metrics = result
                else:
                    loss, extra_metrics = result, {}
                total_loss += loss.detach() * batch_size
                for k, v in extra_metrics.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        if k not in total_metrics:
                            total_metrics[k] = torch.tensor(0.0, device=self.device)
                        total_metrics[k] += v.detach() * batch_size
                total_samples += batch_size

            if is_distributed():
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
                for k in list(total_metrics):
                    dist.all_reduce(total_metrics[k], op=dist.ReduceOp.SUM)

            if total_samples.item() > 0:
                val_loss = (total_loss / total_samples.float()).cpu()
                avg_metrics = {
                    f"validation_{k}": (v / total_samples.float()).cpu().item()
                    for k, v in total_metrics.items()
                }
            else:
                val_loss = torch.tensor(0.0)
                avg_metrics = {}

            if is_master():
                logger.log({"validation_loss": float(val_loss.item()), **avg_metrics}, commit=False)
            return val_loss
        finally:
            self.model.train()

    @torch.no_grad()
    def test(self) -> Any:
        """Test loop hook — MUST be overridden by subclasses with a test surface.

        The test loop materialises full model outputs via
        ``model.run(...)``, which under FSDP2 requires unsharded params. The
        base implementation cannot ship a generic version because the inputs
        / outputs / decoding step are foundation-specific.

        Base behaviour:
            - If ``self.test_datapipe is None``: return ``None`` (no-op; user
              didn't ask for testing).
            - Otherwise: raise ``NotImplementedError`` with a pointer to the
              accelerate trainer's test pattern and the consolidated-export
              helper. A soft warn would let users silently miss a
              load-bearing eval path.
        """
        if self.test_datapipe is None:
            return None
        raise NotImplementedError(
            "FullyShardTrainer.test() is round-1 scope; override to "
            "materialise unsharded model output. Reference patterns: "
            "(1) bbml.train.distributed.accelerate_trainer.AccelerateTrainer.test "
            "(gather_object pattern over wrapped_model.run()); "
            "(2) bbml.fsdp._build_consolidated_export_state_dict for a "
            "full-tensor gather before invoking the foundation's .run() path."
        )

    def _infer_batch_size(self, batch: dict[str, Any]) -> int:
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return int(v.shape[0])
        return 1


__all__: list[str] = ["FullyShardTrainer"]
