"""FSDP2 utilities for bbml.

Ported from the ``pretrain-{drawing,opd,sliders}`` canonical recipe. The
submodules are intentionally narrow:

    - :mod:`bbml.fsdp.dist` — process-group bootstrap, rank helpers, DCP
      ``Stateful`` wrappers, FSDP-aware ``clip_grad_norm``.
    - :mod:`bbml.fsdp.parallelise` — ``init_mesh`` + ``fully_shard_model``.
    - :mod:`bbml.fsdp.policies` — canonical ``MixedPrecisionPolicy`` + LoRA
      key filter.
    - :mod:`bbml.fsdp.dcp` — ``torch.distributed.checkpoint`` wrappers.
"""
from __future__ import annotations

from bbml.fsdp.dcp import (
    dcp_load,
    dcp_load_lora,
    dcp_load_model_only,
    dcp_save,
    dcp_save_lora,
    dcp_save_model_only,
    rotate_keep_last,
    strip_heavy_ckpt,
)
from bbml.fsdp.dist import (
    ModelOnlyState,
    ParallelLayout,
    RunState,
    checkpoint_wrap,
    cleanup_dist,
    clip_grad_norm_fsdp,
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_distributed,
    is_local_master,
    is_master,
    master_print,
    setup_dist,
)
# Consolidated-export helpers — exposed for round-2 EMA / merge-LoRA paths that
# need a full-tensor gather + wrapper-stripped state dict. Not consumed by
# FullyShardTrainer in round 1; importers should reach for these via
# ``bbml.fsdp._normalize_export_key`` /
# ``bbml.fsdp._build_consolidated_export_state_dict``.
from bbml.fsdp.dist import (
    _build_consolidated_export_state_dict,
    _normalize_export_key,
)
from bbml.fsdp.parallelise import fully_shard_model, init_mesh
from bbml.fsdp.policies import default_mp_policy, lora_only_filter

__all__: list[str] = [
    "setup_dist",
    "cleanup_dist",
    "ParallelLayout",
    "is_master",
    "is_local_master",
    "is_distributed",
    "master_print",
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "init_mesh",
    "fully_shard_model",
    "default_mp_policy",
    "lora_only_filter",
    "RunState",
    "ModelOnlyState",
    "checkpoint_wrap",
    "clip_grad_norm_fsdp",
    "dcp_save",
    "dcp_load",
    "dcp_save_model_only",
    "dcp_load_model_only",
    "dcp_save_lora",
    "dcp_load_lora",
    "strip_heavy_ckpt",
    "rotate_keep_last",
    # Consolidated-export helpers (round-2 EMA / merge-LoRA).
    "_normalize_export_key",
    "_build_consolidated_export_state_dict",
]
