"""Data transforms for Qwen Image training."""

from typing import Any

import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image

from bbml.core.data_transform import DataTransform

from .models.pipeline_qwenimage_edit_plus import calculate_dimensions


class QwenImageDataTransform(DataTransform):
    """Transform for image fields in Qwen training.

    Resizes images to target pixel area while maintaining aspect ratio,
    converts to tensor in [0, 1] range.

    Args:
        target_area: Target pixel area (width * height).
    """

    def __init__(self, target_area: int = 1024 * 1024):
        self.target_area = target_area

    def transform(self, input: Image.Image | torch.Tensor) -> torch.Tensor:
        """Convert single image to tensor.

        Args:
            input: PIL Image or tensor.

        Returns:
            Tensor in [0, 1] range with shape [C, H, W].
        """
        if isinstance(input, torch.Tensor):
            return input

        w, h = input.size
        target_w, target_h = calculate_dimensions(self.target_area, w / h)
        # Resize and convert to tensor
        img_resized = input.resize((target_w, target_h), Image.LANCZOS)
        tensor = TF.to_tensor(img_resized)  # [C, H, W] in [0, 1]
        return tensor

    def batch_transform(self, input: list[torch.Tensor]) -> torch.Tensor:
        """Stack tensors into batch.

        Args:
            input: List of image tensors.

        Returns:
            Batched tensor [B, C, H, W].
        """
        return torch.stack(input, dim=0)


class QwenReferenceDataTransform(DataTransform):
    """Transform for reference image fields.

    Similar to QwenImageDataTransform but targets condition image size.

    Args:
        target_area: Target pixel area for reference images.
    """

    def __init__(self, target_area: int = 384 * 384):
        self.target_area = target_area

    def transform(self, input: Image.Image | torch.Tensor) -> torch.Tensor:
        """Convert single reference image to tensor."""
        if isinstance(input, torch.Tensor):
            return input

        w, h = input.size
        target_w, target_h = calculate_dimensions(self.target_area, w / h)
        img_resized = input.resize((target_w, target_h), Image.LANCZOS)
        tensor = TF.to_tensor(img_resized)
        return tensor

    def batch_transform(self, input: list[torch.Tensor]) -> torch.Tensor:
        """Stack reference tensors into batch."""
        return torch.stack(input, dim=0)


class QwenTextDataTransform(DataTransform):
    """Pass-through transform for text prompts.

    Text encoding is handled in the foundation's single_step method
    to support dynamic prompt caching.
    """

    def transform(self, input: str) -> str:
        """Pass through text unchanged."""
        return input

    def batch_transform(self, input: list[str]) -> list[str]:
        """Return list of texts (tokenization handled in single_step)."""
        return input
