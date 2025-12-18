"""Configuration and data models for Qwen Image foundation."""

import enum
from pathlib import Path
from typing import Any, Literal

import torch
from diffusers.image_processor import PipelineImageInput
from pydantic import BaseModel, ConfigDict, Field

from bbml.core.datamodels.configs import FoundationConfig

from .types import DataRange


class QwenInput(BaseModel):
    """Input schema for Qwen Image inference."""

    image: PipelineImageInput | None = None
    prompt: str | list[str] | None = None
    height: int | None = None
    width: int | None = None
    negative_prompt: str | list[str] | None = None
    true_cfg_scale: float = 1.0
    num_inference_steps: int = 50
    generator: torch.Generator | list[torch.Generator] | None = None
    max_sequence_length: int = 512
    vae_image_override: int | None = None
    latent_size_override: int | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class QwenOutput(BaseModel):
    """Output schema for Qwen Image inference."""

    images: tuple[str, ...] | list[Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TrainingType(str, enum.Enum):
    """Training mode variants."""

    IM2IM = "im2im"
    NAIVE = "naive"
    REGRESSION = "regression"

    @property
    def is_style(self) -> bool:
        return self in [TrainingType.NAIVE, TrainingType.IM2IM]


LossTermSpecType = int | float | dict[str, int | float] | None


class QwenLossTerms(BaseModel):
    """Loss term weights and configuration."""

    mse: LossTermSpecType = 1.0
    triplet: LossTermSpecType = 0.0
    negative_mse: LossTermSpecType = 0.0
    distribution_matching: LossTermSpecType = 0.0
    pixel_triplet: LossTermSpecType = 0.0
    pixel_lpips: LossTermSpecType = 0.0
    pixel_mse: LossTermSpecType = 0.0
    pixel_distribution_matching: LossTermSpecType = 0.0
    adversarial: LossTermSpecType = 0.0
    teacher: LossTermSpecType = 0.0

    triplet_margin: float = 0.0
    triplet_min_abs_diff: float = 0.0
    teacher_steps: int = 4

    @property
    def pixel_terms(self) -> tuple[str, ...]:
        return (
            "pixel_lpips",
            "pixel_mse",
            "pixel_triplet",
            "pixel_distribution_matching",
        )


class QwenConfig(FoundationConfig):
    """Configuration for QwenImageFoundation.

    Args:
        load_multi_view_lora: Load multi-view LoRA adapter.
        train_max_sequence_length: Max text sequence length for training.
        train_dist: Timestep distribution for training ("linear" or "logit-normal").
        train_shift: Apply resolution-adaptive timestep shift during training.
        inference_dist: Timestep distribution for inference.
        inference_shift: Apply timestep shift during inference.
        static_mu: Fixed mu for timestep shift (overrides dynamic calculation).
        loss_weight_dist: Distribution for loss weighting across timesteps.
        vae_image_size: Target pixel area for VAE encoding.
        offload_text_encoder: Offload text encoder to CPU during training.
        quantize_text_encoder: Apply int4 quantization to text encoder.
        quantize_transformer: Apply fp8 quantization to transformer.
        vae_tiling: Enable VAE tiling for large images.
        gradient_checkpointing: Enable gradient checkpointing for transformer.
    """

    # Model loading
    load_multi_view_lora: bool = False
    from_pretrained: str = "Qwen/Qwen-Image-Edit-2509"

    # Sequence and resolution
    train_max_sequence_length: int = 512
    vae_image_size: int = 1024 * 1024

    # Timestep scheduling
    train_dist: Literal["linear", "logit-normal"] = "linear"
    train_shift: bool = True
    inference_dist: Literal["linear", "logit-normal"] = "linear"
    inference_shift: bool = True
    static_mu: float | None = None
    loss_weight_dist: Literal["scaled_clipped_gaussian", "logit-normal"] | None = None

    # Optimization
    offload_text_encoder: bool = True
    quantize_text_encoder: bool = False
    quantize_transformer: bool = False
    vae_tiling: bool = False
    gradient_checkpointing: bool = False

    # Loss configuration
    train_loss_terms: QwenLossTerms = Field(default_factory=QwenLossTerms)
    validation_loss_terms: QwenLossTerms = Field(default_factory=QwenLossTerms)

    # Training mode
    training_type: TrainingType | None = None
    train_range: DataRange | None = None
    val_range: DataRange | None = None
    test_range: DataRange | None = None

    # Style training
    style_title: str | None = None
    style_base_dir: str | None = None
    style_csv_path: str | None = None
    style_data_dir: str | None = None
    style_ref_dir: str | None = None
    style_val_with: str = "train"
    naive_static_prompt: str | None = None

    # Regression training
    regression_data_dir: str | Path | None = None
    regression_gen_steps: int = 50
    editing_data_dir: str | Path | None = None
    editing_total_per: int = 1
    regression_base_pipe_steps: int = 8

    # Logging
    log_batch_steps: int | None = None
