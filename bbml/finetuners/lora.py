"""
! Delete this comment later ! RB

Proposed change:

Save/load no longer overrides foundation with just lora based weights. We now use the foundation method directly.

This is to simplify ensemble-of-models training, and cases like lora wrapping foundations whilst still training some layers as full weight / heads / mlps. We have modified foundation to use 'delta' checkpointing which just saves all trainable params.

How LoRA now works:
1. get_peft_model() wraps a submodule
2. Original weights -> requires_grad=False (frozen by PEFT) Any full finetune based weights should be intentionally unfrozen after lora added, and lora target modules should not include this layer.
3. Wrapped module replaces original in Foundation's module tree

This is simpler and avoids the bug where non-LoRA trainables were dropped.

FSDP2 helper: :meth:`LoraFinetuner.unwrap_peft_for_fsdp` returns the
underlying transformer (``self.model.base_model.model`` for PEFT-wrapped,
else ``self.model``) so callers using
:func:`bbml.fsdp.parallelise.fully_shard_model` can reach the inner block
list without wrestling with PEFT wrappers. The unwrap keeps adapter params
trainable so the subsequent ``fully_shard`` shards them.
"""
import collections
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.utils import id_tensor_storage
from torch import nn
from torch.nn import Linear

from bbml.core.finetuner import Finetuner
from bbml.core.foundation import Foundation
from bbml.core.utils.debug import ftimed


def _find_linear_modules_names(
    model,
    find_unique=True,
    full_name=False,
):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            if full_name:
                layers.append(name)
            else:
                layer_type = name.split('.')[-1]
                if layer_type.isdigit():  # module in list
                    layer_type = ".".join(name.split('.')[-2:])
                layers.append(layer_type)
    if find_unique:
        unique_layers = set(layers)
        return list(unique_layers)
    else:
        return layers

def _is_nested_mapping(d: Mapping[str, Any]):
    if not isinstance(d, Mapping):
        return False
    return all(isinstance(v,Mapping) for v in d.values())


def _torch_compile_key_adjustments(state_dict):
    """
    Remove all occurrences of '_orig_mod.' in keys.
    """
    updated_state_dict = {}
    for k, v in state_dict.items():
        if "_orig_mod." in k:
            newk = k.replace("_orig_mod.", "")
        else:
            newk = k
        updated_state_dict[newk] = v
    return updated_state_dict


def _remove_duplicate_layers(state_dict):
    """
    in case "base_model.model." is duplicated
    """
    updated_state_dict = {}
    for k, v in state_dict.items():
        newk: str = k
        while newk.startswith("base_model.model.base_model.model."):
            newk = newk.removeprefix("base_model.model.")
        updated_state_dict[newk] = v
    return updated_state_dict


def _remove_tensor_aliasing(state_dict):
    """
    From peft PeftModel.save_pretrained
    Clone any aliased tensors so that each key in the state_dict has a unique
    underlying storage pointer. This mirrors the logic used inside
    `peft.peft_model.PeftModel#state_dict` to ensure that safetensors can
    serialise the model without raising the "tensor aliasing" error.
    """
    ptrs = collections.defaultdict(list)

    for name, tensor in state_dict.items():
        # Non-tensor objects (e.g. strings in bitsandbytes state dicts) need
        # to be handled gracefully, so we fall back to the Python object id.
        if isinstance(tensor, torch.Tensor):
            ptrs[id_tensor_storage(tensor)].append(name)
        else:
            ptrs[id(tensor)].append(name)

    # Find all storage pointers that are shared by more than one tensor name.
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

    for _, names in shared_ptrs.items():
        # Keep the first tensor intact and clone the rest so that they point
        # to unique memory locations.
        for shared_tensor_name in names[1:]:
            state_dict[shared_tensor_name] = state_dict[shared_tensor_name].clone()

    return state_dict


class LoraFinetuner(Finetuner):

    def __init__(
        self,
        model: Foundation,
        module_names: str | Sequence[str] | None = None,
        module_kwargs: Mapping[str, Mapping[str, Any]]|Mapping[str, Any]| None = None,
        module_configs: Mapping[str, LoraConfig]|LoraConfig|None = None,
        **kwargs,
    ):
        super().__init__(model)

        if _is_nested_mapping(module_kwargs):
            for k in module_kwargs:
                module_kwargs[k].update(kwargs)
        elif isinstance(module_kwargs, Mapping):
            module_kwargs.update(kwargs)
        elif module_kwargs is None:
            module_kwargs = kwargs

        module_names, module_kwargs, module_configs = self.apply_defaults(module_names, module_kwargs, module_configs)

        if module_names is None and module_kwargs is None and module_configs is None:
            raise ValueError("Attempted initializing LoraFinetuner with no module targets or configs")

        if not _is_nested_mapping(module_kwargs) and isinstance(module_configs, LoraConfig):
            warnings.warn("Using both singular module_kwargs and module_configs, module_kwargs will be ignored")

        if module_configs is None:
            module_configs = {}
        if module_kwargs is None:
            module_kwargs = {}
        if module_names is None:
            module_names = []

        # we need to normalize to dict[str, configs]
        config_dict = {}

        if isinstance(module_names, str):
            module_names = [module_names]
        for name in module_names:
            # check if we do singular kwargs or configs
            if isinstance(module_configs, LoraConfig):
                config_dict[name] = module_configs
            elif not _is_nested_mapping(module_kwargs):
                config_dict[name] = LoraConfig(**module_kwargs)
            else:
                target_modules = _find_linear_modules_names(getattr(self.model, name))
                lora_config = LoraConfig(
                    target_modules=target_modules,
                )
                config_dict[name] = lora_config
        if _is_nested_mapping(module_kwargs):
            for name, m_kwargs in module_kwargs.items():
                config_dict[name] = LoraConfig(**m_kwargs)
        if isinstance(module_configs, Mapping):
            config_dict.update(module_configs)

        # final check: named modules exist
        if not all(hasattr(self.model, name) for name in config_dict.keys()):
            missing = [name for name in config_dict.keys() if not hasattr(self.model, name)]
            raise ValueError(f"Passed in module names not present in model({model.__class__}): {missing=}")

        # load peft lora configs
        self.modules = {}
        for name, config in config_dict.items():
            lora_module = get_peft_model(getattr(self.model, name), config)
            self.modules[name] = lora_module
            setattr(self.model, name, lora_module)

    def apply_defaults(
        self,
        module_names: str|Sequence[str]|None,
        module_kwargs: Mapping[str,Mapping[str,Any]]|Mapping[str,Any]|None,
        module_configs: Mapping[str,LoraConfig]|LoraConfig|None,
    ):
        """
            Helper function for model-specific finetuners
        """
        return module_names, module_kwargs, module_configs


    @ftimed
    def load(self, load_path: str | Path, **kwargs):
        """
        Load checkpoint via Foundation's trainable params loader (delta)
        which includes the LoRA params (part of named params)
        """
        return self.original_load(load_path, **kwargs)

    def save(self, save_path: str | Path, **kwargs):
        return self.original_save(save_path, **kwargs)

    def get_train_parameters(self):
        """
        Intentionally global (not just self.modules) to ensure
        non-LoRA trainables like rm_head are never dropped from optimizer
        or checkpoints.
        """
        return [{"params": [p for p in self.model.parameters() if p.requires_grad]}]

    def export_state_dict(self) -> dict[str, dict]:
        """
        Export LoRA adapter weights only (PEFT format).
        Prefer Foundation.save() for checkpointing.
        """
        state_dicts = {}
        for name, module in self.modules.items():
            state_dict = get_peft_model_state_dict(module)
            state_dict = _torch_compile_key_adjustments(state_dict)
            state_dict = _remove_duplicate_layers(state_dict)
            state_dicts[name] = state_dict
        return state_dicts

    def unwrap_peft_for_fsdp(self) -> nn.Module:
        """Return the underlying transformer module for FSDP2 sharding.

        PEFT wraps the foundation submodule as
        ``PeftModel -> base_model (LoraModel) -> model (the inner transformer)``.
        :func:`bbml.fsdp.parallelise.fully_shard_model` expects a module that
        exposes the block list (e.g. ``transformer_blocks``); this helper
        walks past the PEFT wrappers so callers don't have to.

        Behaviour:
            - If ``self.model`` has a ``base_model.model`` (PEFT wrapped):
              return that inner transformer.
            - Otherwise (no PEFT wrap, or ``self.model`` is the inner
              transformer already): return ``self.model``.

        Trainable status is preserved — the unwrap is purely structural.
        The caller MUST shard BEFORE freezing any branches; freezing a
        parameter before ``fully_shard`` causes FSDP2 to skip sharding it
        and leaves a full tensor instead of a DTensor.
        """
        cur: nn.Module = self.model
        base = getattr(cur, "base_model", None)
        if base is not None:
            inner = getattr(base, "model", None)
            if inner is not None:
                return inner
            return base
        return cur
