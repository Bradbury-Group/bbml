
from pydantic import BaseModel

from bbml import FoundationConfig


class MyFoundationConfig(FoundationConfig):
    """Loaded from configs/*.yaml."""

    hidden_dim: int = 256
    num_layers: int = 4
    # Extend


class MyInput(BaseModel):
    """Input schema for run(). Define what inference accepts."""
    pass  # e.g., text: str, image: list[float], etc.


class MyOutput(BaseModel):
    """Output schema for run(). Define what inference returns."""
    pass  # e.g., logits: list[float], label: str, etc.
