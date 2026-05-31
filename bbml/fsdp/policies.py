"""FSDP2 mixed-precision policy + key filters.

Load-bearing items:
    - ``reduce_dtype=torch.float32`` is MANDATORY for grad numerics. Setting it
      to bf16 silently degrades training stability.
    - ``cast_forward_inputs=True`` is required so that activations entering the
      sharded module are downcast before the matmul (and not after).
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


def default_mp_policy(
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.bfloat16,
) -> MixedPrecisionPolicy:
    """Build the canonical FSDP2 mixed-precision policy.

    Defaults follow the pretrain-{drawing,opd,sliders} convention:
    bf16 params, fp32 grad reduction, bf16 outputs. ``reduce_dtype=fp32`` is
    mandatory for grad numerics — do NOT pass bf16 here unless you know exactly
    what you're doing.

    Args:
        param_dtype: dtype for sharded parameter shards. Default bf16.
        reduce_dtype: dtype for gradient reduction. MUST be fp32 in practice.
        output_dtype: dtype for module forward outputs. Default bf16.

    Returns:
        A ``MixedPrecisionPolicy`` ready to splat into ``fully_shard(...)``.
    """
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=True,
        output_dtype=output_dtype,
    )


def lora_only_filter(key: str) -> bool:
    """Predicate used by the DCP LoRA save/load path.

    Returns True iff the state-dict key contains the ``lora_`` infix that PEFT
    inserts on every adapter parameter. Used to filter a full FSDP state dict
    down to just the trainable adapter weights.
    """
    return "lora_" in key


__all__: list[str] = ["default_mp_policy", "lora_only_filter"]
