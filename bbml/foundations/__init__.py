"""Foundation model implementations for bbml.

Each foundation provides a complete model implementation with:
- Model architecture
- Training logic (single_step)
- Data transforms
- Inference (run)
- Serialization (save/load)
"""

from . import gpt2
from . import qwen_image

__all__ = ["gpt2", "qwen_image"]
