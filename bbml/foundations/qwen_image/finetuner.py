"""LoRA finetuner for Qwen Image foundation."""

from typing import Any, Mapping, Sequence

from peft import LoraConfig

from bbml.core.foundation import Foundation
from bbml.finetuners.lora import LoraFinetuner

# Default LoRA target modules for Qwen transformer
QWEN_LORA_TARGET_MODULES = [
    # Image attention
    "to_q",
    "to_k",
    "to_v",
    "to_qkv",
    # Text attention
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_added_qkv",
    # Projections
    "proj",
    "txt_in",
    "img_in",
    "txt_mod.1",
    "img_mod.1",
    "proj_out",
    "to_add_out",
    "to_out.0",
    # FFN
    "net.2",
    "linear",
    "linear_1",
    "linear_2",
]


class QwenLoraFinetuner(LoraFinetuner):
    """LoRA finetuner specialized for Qwen Image transformer.

    Applies LoRA adapters to the dual-stream transformer blocks,
    targeting attention projections, modulation layers, and FFN.

    Args:
        model: QwenImageFoundation instance.
        lora_rank: Rank for LoRA decomposition.
        lora_alpha: Alpha scaling for LoRA.
        target_modules: Override default target modules.
        **kwargs: Additional kwargs passed to LoraConfig.
    """

    def __init__(
        self,
        model: Foundation,
        lora_rank: int = 16,
        lora_alpha: int | None = None,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ):
        # Build module kwargs for transformer
        if lora_alpha is None:
            lora_alpha = lora_rank

        module_kwargs = {
            "transformer": {
                "r": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules or QWEN_LORA_TARGET_MODULES,
                **kwargs,
            }
        }

        super().__init__(
            model,
            module_names=["transformer"],
            module_kwargs=module_kwargs,
        )

        # Ensure transformer is in correct dtype after PEFT wrapping
        if hasattr(model, "dtype"):
            self.model.transformer.to(dtype=model.dtype)

    def apply_defaults(
        self,
        module_names: str | Sequence[str] | None,
        module_kwargs: Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | None,
        module_configs: Mapping[str, LoraConfig] | LoraConfig | None,
    ) -> tuple:
        """Apply Qwen-specific defaults.

        If no configuration provided, defaults to transformer with
        standard Qwen LoRA targets.
        """
        if module_names is None and module_kwargs is None and module_configs is None:
            module_names = ["transformer"]
            module_kwargs = {
                "transformer": {
                    "r": 16,
                    "lora_alpha": 16,
                    "target_modules": QWEN_LORA_TARGET_MODULES,
                }
            }
        return module_names, module_kwargs, module_configs
