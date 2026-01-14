from __future__ import annotations

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
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm

from bbml import logger
from bbml.core.datamodels import TrainerConfig
from bbml.core.datapipe import DataPipe
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
    ):
        super().__init__(model, train_config, train_datapipe, val_datapipe, test_datapipe)

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

        grad_clip = getattr(self.train_config, "grad_clip_norm", None)

        for epoch in range(self.train_config.train_epochs):
            # Re-prepare dataloader each epoch for epoch-seeded shuffling
            if hasattr(self.train_datapipe, "set_epoch"):
                self._print(f"[AccelerateTrainer] Setting epoch={epoch} on train_datapipe")
                self.train_datapipe.set_epoch(epoch)

            dataloader = self.accelerator.prepare(self.train_datapipe.get_loader())
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

                # All side effects gated on sync_gradients
                if self.accelerator.sync_gradients:
                    if grad_clip is not None:
                        self.accelerator.clip_grad_norm_(self.wrapped_model.parameters(), grad_clip)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    reduced_loss = self.accelerator.reduce(loss.detach(), reduction="mean")

                    learning_rates = {f"lr.{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())}
                    log_metrics = {
                        "train_loss": reduced_loss.item(),
                        "step": self.train_config.step,
                        "micro_step": micro_step,
                        "epoch": epoch,
                        **learning_rates,
                        **extra_metrics,
                    }
                    self._log(log_metrics, commit=True)

                    if self.accelerator.is_main_process:
                        pbar.set_postfix({"loss": reduced_loss.item()})

                    self.do_val_test_save()
                    self.train_config.step += 1

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
        val_dataloader = self.accelerator.prepare(self.val_datapipe.get_loader())

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
        """Test loop (runs on all ranks, logs on main)."""
        if not isinstance(self.model, Runnable):
            if self.accelerator.is_main_process:
                warnings.warn(f"Model {self.model!r} is not runnable, testing via `run()` is not available.")
            return

        if self.test_datapipe is None:
            if self.accelerator.is_main_process:
                warnings.warn("Testing DataPipe not provided, skipping")
            return

        self.wrapped_model.eval()
        test_dataloader = self.accelerator.prepare(self.test_datapipe.get_loader())

        testing_samples = []
        for i, batch in enumerate(tqdm(test_dataloader, desc="Test", disable=not self.accelerator.is_main_process)):
            unwrapped = self.accelerator.unwrap_model(self.wrapped_model)
            test_input = unwrapped.input_model(**batch)
            output: BaseModel = unwrapped.run(test_input)

            if self.accelerator.is_main_process:
                logger.log({f"input_{k}_{i}": v for k, v in test_input.model_dump().items()}, commit=False)
                logger.log({f"output_{k}_{i}": v for k, v in output.model_dump().items()}, commit=False)

            testing_samples.append({"input": test_input, "output": output})

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
        """
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            save_path = Path(save_path)
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
