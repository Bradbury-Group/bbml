from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, Sequence

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from bbml.registries import LRSchedulerRegistry, LoggingBackendRegistry, OptimizerRegistry


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

    train_epochs: int = 1
    batch_size: int = 1
    step: int = 0

    validation_step_trigger: StepTrigger|None = None
    test_step_trigger: StepTrigger|None = None
    save_step_trigger: StepTrigger|None = None

    num_validation_samples: int| None = None
    num_test_samples: int| None = None

    @model_validator(mode="after")
    def add_suffix_to_names(self):
        if self.name is None:
            self.name = ""
        if self.name_suffix is None:
            return
        suffix_sum = ""
        for suf_name,suf_val in self.name_suffix.items():
            suffix_sum += "_" + suf_name
            suf_val = str(suf_val)
            suffix_sum += "_" + suf_val
        self.name += suffix_sum
        self.output_dir = self.output_dir.removesuffix("/") # in case
        self.output_dir += suffix_sum
    
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
