"""Example training script for Qwen Image Edit.

Usage:
    python examples/qwen_image/train.py -c configs/qwen_image/base.yaml

This script demonstrates:
- Loading QwenImageFoundation
- Setting up LoRA finetuning
- Training with bbml's SimpleTrainer
"""

from pathlib import Path
from typing import Any

from bbml import DataPipe, SimpleTrainer, TrainerConfig, run_interface
from bbml.foundations.qwen_image import (
    QwenConfig,
    QwenImageFoundation,
    QwenLoraFinetuner,
)


def train_fn(cfg_dict: dict[str, Any]) -> None:
    """Main training function.

    Args:
        cfg_dict: Configuration dictionary from YAML files.
    """
    # Parse configs
    train_config = TrainerConfig(**cfg_dict)
    model_config = QwenConfig(**cfg_dict)

    # Initialize foundation
    foundation = QwenImageFoundation(model_config, train_config)

    # Apply LoRA finetuning (optional - comment out for full finetuning)
    lora_rank = cfg_dict.get("lora_rank", 16)
    lora_alpha = cfg_dict.get("lora_alpha", 16)
    finetuner = QwenLoraFinetuner(
        foundation,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    _ = finetuner  # Applied to foundation via monkey-patching

    # Setup data pipeline
    # NOTE: Replace with your actual dataset
    # train_dataset = YourDataset(...)
    # train_dp = (
    #     DataPipe(batch_size=train_config.batch_size, shuffle=True, num_workers=4)
    #     .add_dataset(train_dataset)
    #     .add_transforms(foundation.data_transforms)
    # )

    # Example placeholder for demonstration
    print("Training configuration loaded:")
    print(f"  Project: {train_config.project}")
    print(f"  Name: {train_config.name}")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Model: {model_config.from_pretrained}")

    # trainer = SimpleTrainer(
    #     foundation,
    #     train_config,
    #     train_datapipe=train_dp,
    #     val_datapipe=None,
    #     test_datapipe=None,
    # )
    # trainer.train()


if __name__ == "__main__":
    run_interface(train_fn)
