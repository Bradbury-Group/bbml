"""Qwen Image Foundation for bbml.

This module provides a bbml Foundation implementation for the Qwen Image Edit
diffusion model, a dual-stream DiT architecture for image editing.

Example:
    >>> from bbml.foundations.qwen_image import QwenImageFoundation, QwenConfig
    >>> config = QwenConfig()
    >>> foundation = QwenImageFoundation(config)
"""

from .data_transforms import (
    QwenImageDataTransform,
    QwenReferenceDataTransform,
    QwenTextDataTransform,
)
from .datamodels import (
    QwenConfig,
    QwenInput,
    QwenLossTerms,
    QwenOutput,
    TrainingType,
)
from .finetuner import QWEN_LORA_TARGET_MODULES, QwenLoraFinetuner

# Re-export model components for direct access
from .models import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    AutoencoderKLQwenImage,
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
    calculate_dimensions,
)
from .qwen_foundation import QwenImageFoundation
from .sampling import TimestepDistUtils
from .types import DataRange

__all__ = [
    # Foundation
    "QwenImageFoundation",
    # Config
    "QwenConfig",
    "QwenInput",
    "QwenOutput",
    "QwenLossTerms",
    "TrainingType",
    # Data transforms
    "QwenImageDataTransform",
    "QwenReferenceDataTransform",
    "QwenTextDataTransform",
    # Finetuning
    "QwenLoraFinetuner",
    "QWEN_LORA_TARGET_MODULES",
    # Utilities
    "TimestepDistUtils",
    "DataRange",
    # Model components
    "QwenImageTransformer2DModel",
    "AutoencoderKLQwenImage",
    "QwenImageEditPlusPipeline",
    "QwenDoubleStreamAttnProcessor2_0",
    "calculate_dimensions",
    "CONDITION_IMAGE_SIZE",
    "VAE_IMAGE_SIZE",
]
