from __future__ import annotations

import inspect
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    set_seed,
)
from pydantic import BaseModel
from bbml.logger.utils import is_image_like, is_image_batch_like
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm

from bbml import logger
from bbml.core.datamodels import TrainerConfig
from bbml.core.datapipe import DataPipe, LoaderContext
from bbml.core.foundation import Foundation
from bbml.core.interfaces import Runnable, Trainable
from bbml.core.trainer import Trainer
from bbml.registries import LRSchedulerRegistry, OptimizerRegistry
from bbml.train.param_groups import build_param_groups
from bbml.train.simple_trainer import init_cls_from_config

SHARDING_STRATEGY_MAP = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
}


class AccelerateTrainer(Trainer):
    """Trainer using HuggingFace Accelerate for distributed training.

    Args:
        model: Foundation or Trainable model.
        train_config: TrainerConfig with training parameters.
        train_datapipe: DataPipe for training data.
        val_datapipe: DataPipe for validation (optional).
        test_datapipe: DataPipe for testing (optional).
        mixed_precision: "no", "fp16", or "bf16".
        gradient_accumulation_steps: Steps before optimizer update. None reads from config.
        ddp_find_unused_parameters: DDP flag for unused parameters.
        ddp_static_graph: DDP static graph optimization.
        fsdp_sharding_strategy: FSDP sharding (None=DDP, "FULL_SHARD", "SHARD_GRAD_OP", etc).
        fsdp_transformer_layer_cls: Transformer layer class for FSDP auto-wrap.
        fsdp_cpu_offload: Offload FSDP params to CPU.
        fsdp_use_orig_params: Use original params in FSDP (required for mixed requires_grad, e.g. LoRA).
    """

    def __init__(
        self,
        model: Trainable | Foundation,
        train_config: TrainerConfig,
        train_datapipe: DataPipe,
        val_datapipe: DataPipe | None = None,
        test_datapipe: DataPipe | None = None,
        mixed_precision: Literal["no", "fp16", "bf16"] = "bf16",
        gradient_accumulation_steps: int | None = None,
        ddp_find_unused_parameters: bool = False,
        ddp_static_graph: bool = False,
        fsdp_sharding_strategy: str | None = None,
        fsdp_transformer_layer_cls: type | None = None,
        fsdp_cpu_offload: bool = False,
        fsdp_use_orig_params: bool = False,
        prepare_dataloader: bool = True,
    ):
        super().__init__(model, train_config, train_datapipe, val_datapipe, test_datapipe)

        # True: accelerator.prepare(loader) shards; pipe gets a dp_size=1 ctx.
        # False: pass the real data-parallel ctx and use the loader as returned.
        self.prepare_dataloader = prepare_dataloader
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
            if gradient_accumulation_steps is not None
            else getattr(train_config, "gradient_accumulation_steps", 1)
        )

        fsdp_plugin = None
        kwargs_handlers = []

        if fsdp_sharding_strategy is not None:
            # FSDP mode
            # Build auto_wrap_policy: must be a callable, not a class
            auto_wrap_policy = None
            if fsdp_transformer_layer_cls is not None:
                auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={fsdp_transformer_layer_cls},
                )

            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=SHARDING_STRATEGY_MAP.get(
                    fsdp_sharding_strategy, ShardingStrategy.SHARD_GRAD_OP
                ),
                cpu_offload=fsdp_cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                use_orig_params=fsdp_use_orig_params,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
        else:
            # DDP mode
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=ddp_find_unused_parameters,
                static_graph=ddp_static_graph,
            )
            kwargs_handlers.append(ddp_kwargs)

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            kwargs_handlers=kwargs_handlers,
            fsdp_plugin=fsdp_plugin,
        )

        # Keep unwrapped model reference for save/load
        self.foundation = model
        self.wrapped_model: torch.nn.Module | None = None

    def _log(self, metrics: dict, commit: bool = True) -> None:
        """Log metrics only on main process."""
        if self.accelerator.is_main_process:
            logger.log(metrics, commit=commit)

    def _print(self, msg: str) -> None:
        """Print only on main process."""
        if self.accelerator.is_main_process:
            print(msg)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer from model or config.

        Uses rule-based param grouping if configured, else finetuner-defined groups.
        Must be called on unwrapped model to respect finetuner routing.
        """
        if self.model.optimizer is not None:
            return self.model.optimizer

        if self.train_config.optimizer is not None:
            optimizer_cls = OptimizerRegistry.get(self.train_config.optimizer)
            if self.train_config.param_group_rules:
                param_groups = build_param_groups(
                    self.model,
                    base_lr=self.train_config.lr,
                    base_wd=getattr(self.train_config, "weight_decay", 0.0),
                    rules=self.train_config.param_group_rules,
                )
            else:
                param_groups = self.model.get_train_parameters()
            return init_cls_from_config(optimizer_cls, self.train_config, param_groups)

        raise ValueError("Optimizer couldn't be initiated from model or config")

    def _init_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        """Initialize LR scheduler from model or config."""
        if self.model.lr_scheduler is not None:
            return self.model.lr_scheduler

        if self.train_config.lr_scheduler is not None:
            lr_scheduler_cls = LRSchedulerRegistry.get(self.train_config.lr_scheduler)
            return init_cls_from_config(lr_scheduler_cls, self.train_config, optimizer)

        raise ValueError("LRScheduler couldn't be initiated from model or config")

    def _build_loader(self, pipe: DataPipe):
        """Build a loader honoring ``prepare_dataloader``.

        True (default): pipe gets a ``dp_size=1`` ctx and accelerate does the
        sharding via ``prepare``. False: pipe self-shards from the real ctx and
        the loader is used as returned. Pipes whose ``get_loader`` predates the
        ctx seam fall back to the legacy call with a ``DeprecationWarning``.
        """
        if self.prepare_dataloader:
            ctx = LoaderContext(dp_size=1)
        else:
            ctx = LoaderContext(
                dp_rank=self.accelerator.process_index,
                dp_size=self.accelerator.num_processes,
                seed=self.train_config.seed,
            )
        if len(inspect.signature(pipe.get_loader).parameters) >= 1:
            loader = pipe.get_loader(ctx)
        else:
            warnings.warn(
                "DataPipe.get_loader without a ctx parameter is deprecated; "
                "add `ctx: LoaderContext | None = None`.",
                DeprecationWarning,
            )
            loader = pipe.get_loader()
        return self.accelerator.prepare(loader) if self.prepare_dataloader else loader

    def train(self) -> None:
        """Main training loop with gradient accumulation support."""
        set_seed(self.train_config.seed, device_specific=True)
        self._print(f"[AccelerateTrainer] Seed={self.train_config.seed}, world_size={self.accelerator.num_processes}")

        if hasattr(self.train_config, "expected_world_size"):
            assert self.accelerator.num_processes == self.train_config.expected_world_size

        if self.accelerator.is_main_process and self.train_config.logging_backends is not None:
            logger.start(
                self.train_config.logging_backends,
                **self.train_config.model_dump(),
            )

        self.model.train()
        self.model.to(device=self.accelerator.device)

        # Initialize optimizer BEFORE prepare - must use model.get_train_parameters()
        # which routes through finetuner for LoRA-safe param selection
        optimizer = self._init_optimizer()
        lr_scheduler = self._init_lr_scheduler(optimizer)

        self.wrapped_model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, lr_scheduler
        )

        if self.train_config.load_path is not None:
            self.load(self.train_config.load_path)

        # Foundation hook: after load, before the first batch.
        self.model.on_train_start(self.train_config.step)

        grad_clip = getattr(self.train_config, "grad_clip_norm", None)

        for epoch in range(self.train_config.train_epochs):
            # Re-prepare dataloader each epoch for epoch-seeded shuffling
            if hasattr(self.train_datapipe, "set_epoch"):
                self._print(f"[AccelerateTrainer] Setting epoch={epoch} on train_datapipe")
                self.train_datapipe.set_epoch(epoch)

            dataloader = self._build_loader(self.train_datapipe)
            total_batches = len(dataloader)

            self._print(f"[AccelerateTrainer] Epoch {epoch + 1}/{self.train_config.train_epochs}")

            pbar = tqdm(
                enumerate(dataloader),
                total=total_batches,
                desc=f"Epoch {epoch + 1}",
                disable=not self.accelerator.is_main_process,
            )

            self.optimizer.zero_grad(set_to_none=True)

            for micro_step, batch in pbar:
                micro_step_start = time.perf_counter()
                with self.accelerator.accumulate(self.wrapped_model):
                    # Namespace step info to avoid collision with data keys
                    batch["_bbml"] = {
                        "step": self.train_config.step,
                        "micro_step": micro_step,
                        "batch_num": micro_step,
                        "epoch": epoch,
                        "split": "train",
                    }

                    # Call wrapped_model(batch) which routes to forward() -> single_step()
                    # This ensures DDP/FSDP wrappers properly intercept for grad sync
                    result = self.wrapped_model(batch)
                    if isinstance(result, tuple):
                        loss, extra_metrics = result
                    else:
                        loss, extra_metrics = result, {}

                    self.accelerator.backward(loss)

                micro_step_time = time.perf_counter() - micro_step_start

                # All side effects gated on sync_gradients
                if self.accelerator.sync_gradients:
                    optim_step_start = time.perf_counter()

                    if grad_clip is not None:
                        self.accelerator.clip_grad_norm_(self.wrapped_model.parameters(), grad_clip)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    # Foundation hook: immediately after optimizer + scheduler step.
                    self.model.on_optimizer_step(self.train_config.step)
                    self.optimizer.zero_grad(set_to_none=True)

                    optim_step_time = time.perf_counter() - optim_step_start

                    reduced_loss = self.accelerator.reduce(loss.detach(), reduction="mean")

                    learning_rates = {f"lr.{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())}
                    log_metrics = {
                        "train_loss": reduced_loss.item(),
                        "step": self.train_config.step,
                        "micro_step": micro_step,
                        "epoch": epoch,
                        "timing/micro_step_s": micro_step_time,
                        "timing/optimizer_step_s": optim_step_time,
                        **learning_rates,
                        **extra_metrics,
                    }
                    self._log(log_metrics, commit=True)

                    if self.accelerator.is_main_process:
                        pbar.set_postfix({"loss": reduced_loss.item()})

                    self.do_val_test_save()
                    self.train_config.step += 1

                    if (self.train_config.max_training_steps is not None
                            and self.train_config.step >= self.train_config.max_training_steps):
                        self._print(f"[AccelerateTrainer] Reached max_training_steps={self.train_config.max_training_steps}")
                        self.do_val_test_save(do_all=True)
                        return

        self.do_val_test_save(do_all=True)

    def _infer_batch_size(self, batch: dict) -> int:
        """Infer batch size from first tensor in batch."""
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return v.shape[0]
        return 1

    @torch.no_grad()
    def validate(self) -> torch.Tensor:
        """Validation loop with weighted batch accumulation for correct averaging."""
        if self.val_datapipe is None:
            if self.accelerator.is_main_process:
                warnings.warn("Validation DataPipe not provided, skipping")
            return torch.tensor(0.0)

        self.wrapped_model.eval()
        val_dataloader = self._build_loader(self.val_datapipe)

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        total_metrics: dict[str, torch.Tensor] = {}
        total_samples = torch.tensor(0, device=self.accelerator.device)

        for batch in tqdm(val_dataloader, desc="Validation", disable=not self.accelerator.is_main_process):
            batch["_bbml"] = {"step": self.train_config.step, "split": "validation"}
            batch_size = self._infer_batch_size(batch)

            result = self.wrapped_model(batch)
            if isinstance(result, tuple):
                loss, extra_metrics = result
            else:
                loss, extra_metrics = result, {}

            total_loss += loss.detach() * batch_size
            for k, v in extra_metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = torch.tensor(0.0, device=self.accelerator.device)
                total_metrics[k] += v * batch_size
            total_samples += batch_size

        # All-reduce across ranks
        total_loss = self.accelerator.reduce(total_loss, reduction="sum")
        total_samples = self.accelerator.reduce(total_samples, reduction="sum")
        for k in total_metrics:
            total_metrics[k] = self.accelerator.reduce(total_metrics[k], reduction="sum")

        if total_samples > 0:
            val_loss = (total_loss / total_samples).cpu()
            avg_metrics = {f"validation_{k}": (v / total_samples).cpu().item() for k, v in total_metrics.items()}
        else:
            val_loss = torch.tensor(0.0)
            avg_metrics = {}

        self._log({"validation_loss": val_loss.item(), **avg_metrics}, commit=False)
        self.wrapped_model.train()
        return val_loss

    @torch.no_grad()
    def test(self):
        """Test loop (runs on all ranks, gathers results to main for logging)."""
        if not isinstance(self.model, Runnable):
            if self.accelerator.is_main_process:
                warnings.warn(f"Model {self.model!r} is not runnable, testing via `run()` is not available.")
            return

        if self.test_datapipe is None:
            if self.accelerator.is_main_process:
                warnings.warn("Testing DataPipe not provided, skipping")
            return

        self.wrapped_model.eval()
        test_dataloader = self._build_loader(self.test_datapipe)

        local_inputs: list[dict] = []
        local_outputs: list[dict] = []
        for i, batch in enumerate(tqdm(test_dataloader, desc="Test", disable=not self.accelerator.is_main_process)):
            unwrapped = self.accelerator.unwrap_model(self.wrapped_model)
            test_input = unwrapped.input_model(**batch)
            output: BaseModel = unwrapped.run(test_input)

            local_inputs.append(test_input.model_dump())
            local_outputs.append(output.model_dump())

        # Gather results from all ranks to main process
        from torch.distributed import gather_object

        world_size = self.accelerator.num_processes
        all_inputs = [None] * world_size if self.accelerator.is_main_process else None
        all_outputs = [None] * world_size if self.accelerator.is_main_process else None
        gather_object(local_inputs, all_inputs, dst=0)
        gather_object(local_outputs, all_outputs, dst=0)

        testing_samples = []
        if self.accelerator.is_main_process:
            # Flatten gathered lists: all_inputs is list[list[dict]]
            flat_inputs = [d for rank_list in all_inputs for d in rank_list]
            flat_outputs = [d for rank_list in all_outputs for d in rank_list]

            input_logs: dict[str, list] = {}
            output_logs: dict[str, list] = {}
            for inp, out in zip(flat_inputs, flat_outputs):
                for k, v in inp.items():
                    input_logs.setdefault(f"input_{k}", []).append(v)
                for k, v in out.items():
                    output_logs.setdefault(f"output_{k}", []).append(v)

            # Use prompts as captions; convert any image-like lists to image dicts
            prompts = input_logs.pop("input_prompt", [])
            all_logs = {**input_logs, **output_logs}
            for key, vals in all_logs.items():
                if not isinstance(vals, list) or not vals:
                    continue
                if any(is_image_like(v) or is_image_batch_like(v) for v in vals):
                    if prompts and len(prompts) == len(vals):
                        all_logs[key] = {f"[{i}] {p}": v for i, (p, v) in enumerate(zip(prompts, vals))}
                    else:
                        all_logs[key] = {f"[{i}]": v for i, v in enumerate(vals)}
            logger.log(all_logs, commit=False)

            testing_samples = [{"input": i, "output": o} for i, o in zip(flat_inputs, flat_outputs)]

        self.wrapped_model.train()
        return testing_samples

    def do_val_test_save(self, do_all: bool = False) -> None:
        """Check triggers and run validation/test/save."""
        self.wrapped_model.eval()

        should_validate = (
            self.train_config.step > 0
            and self.train_config.check_step_trigger(
                self.train_config.step,
                self.train_config.validation_step_trigger,
            )
        ) or do_all

        if should_validate:
            self.validate()

        if self.train_config.check_step_trigger(self.train_config.step, self.train_config.test_step_trigger) or do_all:
            self.test()

        if self.train_config.check_step_trigger(self.train_config.step, self.train_config.save_step_trigger) or do_all:
            self.save(self.train_config.output_dir)

        self.wrapped_model.train()

    def save(self, save_path: str | Path) -> None:
        """
        Save checkpoint (main process only after barrier).
        Gathers state dict from WRAPPED model to handle DDP/FSDP correctly,
        then passes to foundation.save() which extracts trainable delta.

        Save path structure: {save_path}/{run_name}/step_{step}/
        """
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            save_path = Path(save_path)
            run_name = self.train_config.name or "unnamed"
            step = self.train_config.step
            save_path = save_path / run_name / f"step_{step}"
            save_path.mkdir(parents=True, exist_ok=True)

            # Gather state dict from WRAPPED model (handles DDP/FSDP sharding)
            # Must call on wrapped, not unwrapped, for proper distributed gathering
            state_dict = self.accelerator.get_state_dict(self.wrapped_model)

            unwrapped = self.accelerator.unwrap_model(self.wrapped_model)
            unwrapped.save(save_path, state_dict=state_dict)

            torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")
            torch.save(self.lr_scheduler.state_dict(), save_path / "lr_scheduler.pt")

            self._print(f"[AccelerateTrainer] Checkpoint saved to {save_path}")

        self.accelerator.wait_for_everyone()

    def load(self, load_path: str | Path) -> None:
        """Load checkpoint (all ranks load, barrier for sync)."""
        load_path = Path(load_path)
        self.accelerator.wait_for_everyone()

        unwrapped = self.accelerator.unwrap_model(self.wrapped_model)
        unwrapped.load(load_path)

        optim_path = load_path / "optimizer.pt"
        if optim_path.exists():
            self.optimizer.load_state_dict(
                torch.load(optim_path, map_location=self.accelerator.device, weights_only=True)
            )

        lrs_path = load_path / "lr_scheduler.pt"
        if lrs_path.exists():
            self.lr_scheduler.load_state_dict(torch.load(lrs_path, weights_only=True))

        self.accelerator.wait_for_everyone()
        self._print(f"[AccelerateTrainer] Checkpoint loaded from {load_path}")
