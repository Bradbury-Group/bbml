import itertools
import os.path
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid
import collections
import warnings  # builtin
from peft.tuners.lora import LoraLayer
import torch  # external library

from peft import load_peft_weights, set_peft_model_state_dict, get_peft_model_state_dict, LoraConfig, get_peft_model
from peft.utils import id_tensor_storage
from safetensors.torch import load_file, save_file
from torch.nn import Linear

from bbml.core.foundation import Foundation
from bbml.core.utils.debug import ftimed
from bbml.core.finetuner import Finetuner


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
    def load(self, load_path: str|Path):
        """
            load_path should be a directory. loaded by correspnding module names: name.safetensors
        """
        load_path = Path(load_path)
        print(f"Loading Lora from {load_path}")
        device = getattr(self.model, "device", None)
        for name, module in self.modules.items():
            peft_weights = load_file(str(load_path/f"{name}.safetensors"), device=device)            
            keys_warning = set_peft_model_state_dict(module, peft_weights)
            print(f"{keys_warning=}")


    def save(self, save_path: str|Path):
        """
            save_path should be a directory. saved to correspnding module names: name.safetensors
        """
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        for name, module in self.modules.items():
            state_dict = get_peft_model_state_dict(module)
            state_dict = _torch_compile_key_adjustments(state_dict)
            state_dict = _remove_duplicate_layers(state_dict)
            # Remove tensor aliasing as in peft / huggingface
            state_dict = _remove_tensor_aliasing(state_dict)
            save_file(state_dict, str(save_path/f"{name}.safetensors"))


    def get_train_parameters(self):
        all_params = list(itertools.chain.from_iterable(map(lambda m:m.parameters(), self.modules.values())))
        all_trainable_params = [p for p in all_params if p.requires_grad]
        return [{"params": all_trainable_params},]


