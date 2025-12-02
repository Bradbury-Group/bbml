from pathlib import Path
from typing import Any
from pydantic import BaseModel, model_validator


class TrainerConfig(BaseModel):
    name: str | None = None
    output_dir: Path = Path("checkpoints")
    name_suffix: dict[str, Any]|None = None

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

class FoundationConfig(BaseModel):
    pass
