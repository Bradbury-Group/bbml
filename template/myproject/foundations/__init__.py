"""Foundation implementations.

Stubs to extend:
  - MyFoundation: your model architecture
  - MyFoundationConfig: your hyperparameters
  - MyDataTransform: your data transforms

See example_foundation.py for the template.
See bbml/foundations/gpt2/ for a production example.
"""

from myproject.foundations.example_foundation import (
    MyFoundation,
    MyFoundationConfig,
    MyDataTransform,
    MyInput,
    MyOutput,
)

__all__ = [
    "MyFoundation",
    "MyFoundationConfig",
    "MyDataTransform",
    "MyInput",
    "MyOutput",
]
