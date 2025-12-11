"""
Extend bbml.core.data_transform.DataTransform.

DataTransform converts raw dataset items -> tensors, then batches.
Foundation.data_transforms returns dict[str, DataTransform] keyed by field.

DataPipe.collate_fn() orchestrates this automatically.
See bbml/core/data_transform.py for the abstract base.
See bbml/data/transforms.py for examples (ImageDataTransform, etc.).
"""

from typing import Any

from torch import Tensor

from bbml.core.data_transform import DataTransform


class MyTransform(DataTransform):
    """
    Create one per field that needs transformation.
    Register in Foundation.data_transforms property.
    """

    def __init__(self):
        # Store any config (e.g., tokenizer, normalization stats)
        pass

    def transform(self, input: dict[str, Any]) -> Tensor:
        """
        Args:
            input: Full dict from dataset.__getitem__().
                   Extract your field: input["your_field"]

        Returns:
            Tensor for this field (any shape, will be batched later).
        """
        raise NotImplementedError

    def batch_transform(self, inputs: list[Tensor]) -> Tensor:
        """
        Args:
            inputs: List of tensors from transform() calls.

        Returns:
            Batched tensor. Usually: torch.stack(inputs, dim=0)
            For variable-length: pad then stack.
        """
        raise NotImplementedError
