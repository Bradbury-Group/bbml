import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer, ParamsT

from bbml.core import checkpointing
from bbml.core.data_transform import DataTransform
from bbml.core.datamodels.configs import FoundationConfig, TrainerConfig
from bbml.core.interfaces import InT, OutT, Runnable, Serializable, Trainable


class Foundation(Trainable, Runnable, Serializable, nn.Module):
    def __init__(self, config: FoundationConfig, train_config: TrainerConfig | None):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.device = None
        self.dtype = None

    @abstractmethod
    def single_step(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict]:
        ...

    @abstractmethod
    def get_train_parameters(self) -> ParamsT:
        ...

    @property
    @abstractmethod
    def data_transforms(self) -> dict[str, DataTransform]:
        ...

    @property
    def optimizer(self) -> Optimizer | None:
        return None

    @property
    def lr_scheduler(self) -> LRScheduler | None:
        return None

    @property
    @abstractmethod
    def input_model(self) -> type[InT]:
        ...

    @property
    @abstractmethod
    def output_model(self) -> type[OutT]:
        ...

    @abstractmethod
    def run(self, input: InT) -> OutT:
        ...

    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict]:
        """Routes to single_step; validates the returned metrics dict.

        A tensor metric crashes ``WandBBackend`` on rank 0 and hangs siblings
        on the next collective, so 0-dim tensors are coerced to float and any
        higher-rank tensor raises (see :meth:`_validate_metrics`).
        """
        result = self.single_step(batch)
        if isinstance(result, tuple):
            loss, metrics = result
            return loss, self._validate_metrics(metrics)
        return result

    def _validate_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Coerce 0-dim tensor metrics to float; raise on non-scalar tensors."""
        bad: list[str] = []
        validated: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, Tensor):
                if value.ndim == 0:
                    validated[key] = float(value.detach())
                else:
                    bad.append(key)
            else:
                validated[key] = value
        if bad:
            raise TypeError(
                f"single_step metrics must be scalar; non-scalar tensor(s) for "
                f"key(s): {sorted(bad)}. Reduce to a scalar (e.g. .item()) before "
                "returning — tensor metrics crash WandBBackend."
            )
        return validated

    # ------------------------------------------------------------------ #
    # training-loop hooks (no-op defaults)
    # ------------------------------------------------------------------ #
    def on_train_start(self, step: int) -> None:
        """After checkpoint load, before the first batch. Seed generators /
        warm caches here. No-op by default."""
        del step

    def on_optimizer_step(self, step: int) -> None:
        """Immediately after ``optimizer.step()`` + ``lr_scheduler.step()``.
        EMA update lives here. No-op by default."""
        del step

    # ------------------------------------------------------------------ #
    # sharding plan
    # ------------------------------------------------------------------ #
    def fsdp_blocks(self) -> Iterable[nn.Module]:
        """Transformer-block ``nn.ModuleList`` sharded by the default
        :meth:`parallelise`. Override to opt into ``FullyShardTrainer``."""
        raise NotImplementedError(
            "override fsdp_blocks() to return the model's transformer-block "
            "nn.ModuleList (e.g. self.model.transformer.h) for the default "
            "parallelise() / FullyShardTrainer"
        )

    def parallelise(self, mesh, *, policy) -> None:
        """In-place FSDP2 parallelisation.

        Default: :func:`bbml.fsdp.parallelise.fully_shard_model` over
        :meth:`fsdp_blocks` with ``policy`` (no separate output module).
        Override for custom wraps (auxiliary heads, EMA / teacher, a distinct
        output module).

        Args:
            mesh: ``DeviceMesh`` (typically the ``shard`` axis); the
                ``FullyShardTrainer`` builds this from ``ParallelismConfig``.
            policy: ``MixedPrecisionPolicy``, by default from
                :func:`bbml.fsdp.policies.default_mp_policy`.
        """
        from bbml.fsdp.parallelise import fully_shard_model

        fully_shard_model(
            self, blocks=self.fsdp_blocks(), output_module=None, mesh=mesh, policy=policy
        )

    def save(self, save_path: str | Path, *, state_dict: dict[str, Tensor] | None = None):
        """Default delta checkpoint save implementation."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        trainable_names = self._determine_trainable_param_names()
        delta = checkpointing.extract_delta_state(self, trainable_names, state_dict)
        metadata = self._build_checkpoint_metadata(trainable_names)
        checkpointing.save_delta(delta, save_path, metadata)
        self._save_auxiliary_assets(save_path)

    def load(self, load_path: str | Path, *, strict: bool = False):
        load_path = Path(load_path)
        delta, loaded_meta = checkpointing.load_delta(load_path)
        trainable_names = self._determine_trainable_param_names()
        expected_meta = self._expected_checkpoint_metadata(trainable_names)
        checkpointing.validate_checkpoint_meta(loaded_meta, expected_meta)
        missing, _ = checkpointing.apply_delta_state(self, delta, strict=strict)
        if missing and not strict:
            warnings.warn(
                f"Checkpoint missing {len(missing)} parameters (e.g. {missing[:3]}). "
                "Inspect structure changes or set strict=True."
            )
        self._load_auxiliary_assets(load_path)

    def _determine_trainable_param_names(self) -> set[str]:
        param_groups = None
        try:
            param_groups = self.get_train_parameters()
        except NotImplementedError:
            param_groups = None
        return checkpointing.extract_trainable_param_names(self, param_groups)

    def _checkpoint_base_id(self) -> str:
        if hasattr(self.config, "model_name_or_path"):
            return getattr(self.config, "model_name_or_path")
        return self.__class__.__name__

    def _checkpoint_format_version(self) -> int:
        return 1

    def _extra_checkpoint_metadata(self) -> dict[str, Any]:
        """Override to attach foundation-specific metadata."""
        return {}

    def _build_checkpoint_metadata(self, trainable_names: set[str]) -> dict[str, Any]:
        meta = {
            "format_version": self._checkpoint_format_version(),
            "base_id": self._checkpoint_base_id(),
            "structure_fingerprint": checkpointing.compute_structure_fingerprint(trainable_names),
            "trainable_count": len(trainable_names),
        }
        extra = self._extra_checkpoint_metadata()
        if extra:
            meta.update(extra)
        return meta

    def _expected_checkpoint_metadata(self, trainable_names: set[str]) -> dict[str, Any]:
        """Metadata expected for validation at load time."""
        return {
            "format_version": self._checkpoint_format_version(),
            "base_id": self._checkpoint_base_id(),
            "structure_fingerprint": checkpointing.compute_structure_fingerprint(trainable_names),
        }

    def _save_auxiliary_assets(self, save_path: Path) -> None:
        del save_path

    def _load_auxiliary_assets(self, load_path: Path) -> None:
        del load_path

    # ------------------------------------------------------------------ #
    # resumable training state
    # ------------------------------------------------------------------ #
    def _resolve_ckpt_format(self) -> str:
        """Checkpoint format from ``train_config.checkpointing`` (default 'delta')."""
        cfg = getattr(self.train_config, "checkpointing", None) if self.train_config is not None else None
        if cfg is None:
            return "delta"
        fmt = getattr(cfg, "format", None)
        if fmt is not None:
            return fmt
        if isinstance(cfg, dict):
            return cfg.get("format", "delta")
        return "delta"

    def save_training_state(self, path, optimizer, scheduler, ema=None, metadata=None) -> None:
        """Persist model + optimizer (+ scheduler / ema / metadata) as one
        resumable checkpoint.

        Default dispatch on ``CheckpointingConfig.format``:
            - ``"delta"`` (single-rank): delta save + ``optimizer.pt`` /
              ``lr_scheduler.pt`` (legacy ``Foundation.save`` path). Rejected
              under multi-rank FSDP2 (writes only local DTensor shards).
            - ``"dcp"`` / ``"dcp_lora"``: ``bbml.fsdp.dcp`` helpers.

        Override to own a bespoke format; ``format`` config is then only a hint.
        """
        from bbml.fsdp.dist import get_world_size, is_master

        path = Path(path)
        fmt = self._resolve_ckpt_format()
        if fmt == "delta":
            if get_world_size() > 1:
                raise RuntimeError(
                    "checkpointing.format='delta' is unsafe under multi-rank FSDP2 "
                    f"(world_size={get_world_size()}); Foundation.save writes only local "
                    "DTensor shards. Use format='dcp' or 'dcp_lora'."
                )
            if is_master():
                path.mkdir(parents=True, exist_ok=True)
                self.save(path)
                torch.save(optimizer.state_dict(), path / "optimizer.pt")
                if scheduler is not None:
                    torch.save(scheduler.state_dict(), path / "lr_scheduler.pt")
            return

        from bbml.fsdp.dcp import dcp_save, dcp_save_lora

        if fmt == "dcp":
            dcp_save(self, optimizer, scheduler, ckpt_path=path, metadata=metadata)
        elif fmt == "dcp_lora":
            dcp_save_lora(self, optimizer, scheduler, ckpt_path=path, metadata=metadata)
        elif fmt == "dcp_model_only":
            raise ValueError(
                "checkpointing.format='dcp_model_only' is load-only; pair with "
                "'dcp' / 'dcp_lora' for the save trigger."
            )
        else:
            raise ValueError(f"unknown checkpointing.format: {fmt!r}")

    def load_training_state(self, path, optimizer, scheduler, ema=None) -> dict:
        """Restore state written by :meth:`save_training_state`; returns metadata.

        Mirrors :meth:`save_training_state`'s format dispatch.
        """
        from bbml.fsdp.dist import get_world_size

        path = Path(path)
        fmt = self._resolve_ckpt_format()
        if fmt == "delta":
            if get_world_size() > 1:
                raise RuntimeError(
                    "checkpointing.format='delta' is unsafe under multi-rank FSDP2 "
                    f"(world_size={get_world_size()}). Use format='dcp' or 'dcp_lora'."
                )
            self.load(path)
            opt_path = path / "optimizer.pt"
            if optimizer is not None and opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, weights_only=True))
            lrs_path = path / "lr_scheduler.pt"
            if scheduler is not None and lrs_path.exists():
                scheduler.load_state_dict(torch.load(lrs_path, weights_only=True))
            return {}

        from bbml.fsdp.dcp import dcp_load, dcp_load_lora, dcp_load_model_only

        if fmt == "dcp":
            return dcp_load(self, optimizer, scheduler, ckpt_path=path)
        elif fmt == "dcp_lora":
            return dcp_load_lora(self, optimizer, scheduler, ckpt_path=path)
        elif fmt == "dcp_model_only":
            return dcp_load_model_only(self, ckpt_path=path)
        raise ValueError(f"unknown checkpointing.format: {fmt!r}")

    # convience method for setting self.device and self.dtype, according to pytorch's Tensor.to and Module.to methods
    def to(self, *args, **kwargs):
        dtype = self.dtype
        device = self.device
        if args:
            arg = args[0]
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (str, torch.device, int)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                dtype = arg.dtype
                device = arg.device
        if kwargs:
            device = kwargs.get("device", device)
            dtype = kwargs.get("dtype", dtype)
        self.dtype = dtype
        self.device = device
        return super().to(*args, **kwargs)

    def should_log_training(self, step: int) -> bool:
        """
            log train step at same step as running validation, but not during validation
        """
        if self.train_config is None:
            warnings.warn("Train config is None")
            return False
        return (
            self.training
            and self.train_config.check_step_trigger(step, self.train_config.validation_step_trigger)
        )
