import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any

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
        """Routes to single_step for distributed training compatibility."""
        return self.single_step(batch)

    def parallelise(self, mesh, *, policy) -> None:
        """In-place FSDP2 parallelisation hook.

        Override on subclasses that opt into ``FullyShardTrainer``. The
        canonical implementation calls
        :func:`bbml.fsdp.parallelise.fully_shard_model` against the inner
        transformer's block list and output module, then optionally shards
        any auxiliary heads (e.g. EMA, teacher) the foundation owns. Base
        implementation raises ``NotImplementedError`` so trainers that try
        to use ``FullyShardTrainer`` against an un-prepared foundation get a
        clear error rather than a silent no-op.

        Args:
            mesh: ``DeviceMesh`` (typically the ``shard`` axis); the
                ``FullyShardTrainer`` builds this from ``ParallelismConfig``.
            policy: ``MixedPrecisionPolicy``, by default from
                :func:`bbml.fsdp.policies.default_mp_policy`.
        """
        raise NotImplementedError(
            "override parallelise() to enable FullyShardTrainer"
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
