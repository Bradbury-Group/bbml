"""
Foundation stub - extend bbml.core.foundation.Foundation.

Foundation = your model + training logic + data transforms + serialization.
It implements three interfaces (see bbml/core/interfaces.py):
  - Trainable: single_step(), get_train_parameters(), data_transforms
  - Runnable: input_model, output_model, run()
  - Serializable: save(), load()

See bbml/foundations/gpt2/ for a production example.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel
from torch import Tensor
from torch.optim.optimizer import ParamsT

from bbml.core.foundation import Foundation
from bbml.core.datamodels.configs import FoundationConfig, TrainerConfig
from bbml.core.data_transform import DataTransform


# === CONFIG ===
# extra="allow" is inherited - arbitrary YAML fields pass through.

class MyFoundationConfig(FoundationConfig):
    """Loaded from configs/*.yaml."""

    hidden_dim: int = 256
    num_layers: int = 4
    # Extend


# === I/O MODELS ===
# Pydantic models for run() input/output. Used for validation and API schemas.

class MyInput(BaseModel):
    """Input schema for run(). Define what inference accepts."""
    pass  # e.g., text: str, image: list[float], etc.


class MyOutput(BaseModel):
    """Output schema for run(). Define what inference returns."""
    pass  # e.g., logits: list[float], label: str, etc.


# === DATA TRANSFORMS ===
# DataTransform converts raw dataset items -> tensors, then batches them.
# Foundation.data_transforms returns dict[str, DataTransform] keyed by dataset field.
# DataPipe calls transform() per-item, then batch_transform() to collate.

class MyDataTransform(DataTransform):
    """Transform for one field of your dataset (e.g., 'features', 'text')."""

    def transform(self, input: dict[str, Any]) -> Tensor:
        # input = one item from dataset.__getitem__()
        # Return tensor for this field
        raise NotImplementedError

    def batch_transform(self, inputs: list[Tensor]) -> Tensor:
        # inputs = list of tensors from transform()
        # Return batched tensor (usually torch.stack)
        raise NotImplementedError


# === FOUNDATION ===

class MyFoundation(Foundation):
    """Your model. Extend this with your architecture.

    __init__ receives:
      - config: MyFoundationConfig (your hyperparameters from YAML)
      - train_config: TrainerConfig (optimizer, scheduler, batch_size, etc.)
    """

    def __init__(
        self,
        config: MyFoundationConfig,
        train_config: TrainerConfig | None = None,
    ):
        super().__init__(config, train_config)
        self.config: MyFoundationConfig

        # self.model = nn.Sequential(
        #     nn.Linear(config.input_dim, config.hidden_dim),
        #     ...
        # )

        raise NotImplementedError()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


    def single_step(self, batch: dict[str, Tensor]) -> Tensor:
        """
        One training step. Return scalar loss.

        Args:
            batch: Dict of tensors keyed by dataset field names.
                   Shape depends on your data_transforms.
                   e.g., {"features": (B, D), "label": (B,)}

        Returns:
            Scalar loss tensor for backward().
        """
        raise NotImplementedError

    def get_train_parameters(self) -> ParamsT:
        """Parameters for optimizer. Return list of param group dicts.

        Simple: [{"params": self.parameters()}]
        """
        raise NotImplementedError

    @property
    def data_transforms(self) -> dict[str, DataTransform]:
        """Map dataset field names -> DataTransform instances.

        Keys must match your dataset's __getitem__() dict keys.
        DataPipe.collate_fn() uses this to transform and batch each field.
        """
        raise NotImplementedError

    # --- Runnable interface ---

    @property
    def input_model(self) -> type[MyInput]:
        """Pydantic model for run() input validation."""
        return MyInput

    @property
    def output_model(self) -> type[MyOutput]:
        """Pydantic model for run() output validation."""
        return MyOutput

    def run(self, input: MyInput) -> MyOutput:
        """
        Inference. For example, multi-step diffusion

        """
        raise NotImplementedError

    def save(self, save_path: str | Path) -> None:
        """Save model checkpoint.

        For safetensors: see bbml/foundations/gpt2/gpt2_foundation.py
        Simple: torch.save(self.state_dict(), path / "model.pt")
        """
        raise NotImplementedError

    def load(self, load_path: str | Path) -> None:
        """Load model checkpoint.

        Match your save() format. Handle device mapping if needed.
        """
        raise NotImplementedError
