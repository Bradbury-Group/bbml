from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, Sequence

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from bbml.registries import LoggingBackendRegistry, LRSchedulerRegistry, OptimizerRegistry


def in_registry(registry) -> Callable:
    """Construct validator to validate value in registry keys"""
    def validate_fn(value: str | Sequence[str]) -> str | Sequence[str]:
        values_seq = [value] if isinstance(value, str) else value
        invalid_values = [v for v in values_seq if v not in registry.keys()]
        if invalid_values:
            raise ValueError(f"Validation: {invalid_values=} not in {registry!r}")
        return value
    return validate_fn

StepTrigger = int|Sequence[int]|Mapping[Literal["at", "every"],int|Sequence[int]]

class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # meta

    project: str
    name: str | None = None
    output_dir: Path = Path("checkpoints")
    name_suffix: dict[str, Any]|None = None
    logging_backends: Annotated[str | list[str], AfterValidator(in_registry(LoggingBackendRegistry))] | None = None
    wandb_entity: str | None = None

    optimizer: Annotated[str, AfterValidator(in_registry(OptimizerRegistry))] | None = None
    lr_scheduler: Annotated[str, AfterValidator(in_registry(LRSchedulerRegistry))] | None = None
    load_path: str | Path | None = None

    seed: int = 0
    train_epochs: int = 1
    batch_size: int = 1
    step: int = 0
    max_training_steps: int | None = None

    validation_step_trigger: StepTrigger|None = None
    test_step_trigger: StepTrigger|None = None
    save_step_trigger: StepTrigger|None = None

    num_validation_samples: int| None = None
    num_test_samples: int| None = None

    gradient_accumulation_steps: int = 1
    grad_clip_norm: float | None = None

    # When True (default), the train ``DistributedSampler`` drops the trailing
    # partial batch so every rank steps the same number of times per epoch.
    # Set False for small datasets where the dropped tail meaningfully reduces
    # train coverage (e.g. MagicBrush dev). Consumed by :class:`FullyShardTrainer`;
    # other trainers ignore the field today (still legal because
    # ``TrainerConfig.model_config = extra="allow"``).
    drop_last_train: bool = True

    # Per-param-group LR/WD rules. Each rule: {match: regex, lr_mult: float, weight_decay?: float}
    # First matching rule wins. Unmatched params get base lr/wd.
    # TODO: Add this info to example config and remove comment here
    param_group_rules: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def add_suffix_to_names(self):
        if self.name is None:
            self.name = ""
        if self.name_suffix is None:
            return self
        suffix_sum = ""
        for suf_name, suf_val in self.name_suffix.items():
            suffix_sum += "_" + suf_name
            suf_val = str(suf_val)
            suffix_sum += "_" + suf_val
        self.name += suffix_sum
        output_dir_str = str(self.output_dir).removesuffix("/")
        self.output_dir = Path(output_dir_str + suffix_sum)
        return self

    @staticmethod
    def check_step_trigger(step: int, trigger: StepTrigger):
        if isinstance(trigger, Mapping):
            return (
                TrainerConfig.check_step_trigger(step, trigger.get("at"))
                or TrainerConfig.check_step_trigger(step, trigger.get("every"))
            )
        elif isinstance(trigger, int):
            return step % trigger == 0
        elif isinstance(trigger, Sequence):
            return step in trigger
        return False


class FoundationConfig(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    # to be extended


class ParallelismConfig(BaseModel):
    """FSDP2 / HSDP / TP parallel-axis sizes.

    Composed into ``TrainerConfig`` as a nested dict; ``TrainerConfig`` has
    ``extra="allow"`` so users can attach this under e.g.
    ``train_config.parallelism`` without modifying the base class.

    Data-parallel dimension is IMPLICIT — it equals
    ``world_size // tensor_parallel`` (only the TP axis collapses ranks into
    a single DP slot; the HSDP replicate axis sees distinct data batches).
    There is no explicit ``data_parallel`` field because the trainer derives
    it from world size and any conflict would be ambiguous. See
    :class:`bbml.fsdp.dist.ParallelLayout` for the canonical semantics.

    Fields:
        tensor_parallel: TP axis size; default 1 (no TP).
        replicate: HSDP replicate axis size; default 1.
        mesh_dim_names: ordered axis names forwarded to ``init_device_mesh``.
        reshard_after_forward_on_output: forwarded to
            :func:`bbml.fsdp.parallelise.fully_shard_model`; ``False``
            (default) keeps the output module's shards materialised through
            backward.
    """

    model_config = ConfigDict(extra="forbid")

    tensor_parallel: int = 1
    replicate: int = 1
    mesh_dim_names: tuple[str, ...] = ("shard",)
    reshard_after_forward_on_output: bool = False


class SamplingConfig(BaseModel):
    """In-training sampling cadence + parameters."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    every: int = 500
    num_samples: int = 4
    seeds: list[int] = Field(default_factory=lambda: [42])
    cfg_scale: float = 3.5
    from_v_pred: bool = True


class MetricsConfig(BaseModel):
    """Per-step metrics behaviour for FSDP-aware trainers."""

    model_config = ConfigDict(extra="forbid")

    loss_buckets: bool = True
    num_buckets: int = 10
    train_preview_every: int | None = None
    train_preview_n: int = 4


class CheckpointingConfig(BaseModel):
    """Checkpoint format + rotation for FSDP-aware trainers.

    ``format``:
        - ``"delta"`` — falls back to ``Foundation.save`` (legacy behaviour).
          REJECTED at :meth:`FullyShardTrainer.train` bootstrap when
          ``world_size > 1``: ``Foundation.save -> extract_delta_state ->
          save_delta`` iterates ``model.state_dict()`` and calls ``.to('cpu')``
          on every entry; under FSDP2 those entries are DTensor LOCAL shards,
          so the on-disk file contains only the rank-0 fragment of each
          parameter. Use ``"dcp"`` or ``"dcp_lora"`` for multi-rank FSDP2.
          ``"delta"`` remains legal single-rank (Trainer / AccelerateTrainer
          / single-GPU FullyShardTrainer).
        - ``"dcp"`` — full DCP via :func:`bbml.fsdp.dcp.dcp_save`.
        - ``"dcp_lora"`` — LoRA-only DCP via
          :func:`bbml.fsdp.dcp.dcp_save_lora`.
        - ``"dcp_model_only"`` — model-weights-only DCP via
          :func:`bbml.fsdp.dcp.dcp_load_model_only`. Recovery / resume path
          for snapshots written by the ``model_only_every`` cadence (which
          uses ``dcp_save_model_only``). Save path still routes through one
          of the other three formats — ``"dcp_model_only"`` is load-only.
    """

    model_config = ConfigDict(extra="forbid")

    format: Literal["delta", "dcp", "dcp_lora", "dcp_model_only"] = "delta"
    every: int = 500
    keep_last: int | None = None
    rotate_strip_heavy: bool = False
    model_only_every: int | None = None
