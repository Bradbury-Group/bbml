from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
import warnings

import yaml


ListStrategy = Literal["replace", "concat", "elementwise"] | Callable[[list, list], list]


def merge_lists(
    base_list: list,
    update_list: list,
    strategy: ListStrategy
) -> list:
    if callable(strategy):
        return strategy(base_list, update_list)
    
    if strategy == "replace":
        return list(update_list)
    
    if strategy == "concat":
        return list(base_list) + list(update_list)

    if strategy == "elementwise":
        # Merge element-by-element, where dict elements are deep-merged, and non-dict is overriden at index
        result = list(base_list)
        for i, item in enumerate(update_list):
            if i < len(result) and isinstance(result[i], dict) and isinstance(item, dict):
                result[i] = deep_update(dict(result[i]), item, list_strategy=strategy)
            elif i < len(result):
                result[i] = item
            else:
                result.append(item)
        return result

    raise ValueError(f"Unknown list strategy: {strategy!r}")


def deep_update(
    base: dict,
    updates: dict,
    *,
    list_strategy: ListStrategy = "replace"
) -> dict:
    """
        Recursive update dict
    """
    for k, v in updates.items():
        current = base.get(k)

        if isinstance(v, dict) and isinstance(current, dict):
            base[k] = deep_update(dict(current), v, list_strategy=list_strategy)

        elif isinstance(v, list) and isinstance(current, list):
            base[k] = merge_lists(current, v, list_strategy)

        else:
            base[k] = v

    return base



def config_compose(
    update_configs: Sequence[Path| str | Mapping],
    list_strategy: ListStrategy = "replace",
) -> dict:

    if not isinstance(update_configs, Sequence):
        raise ValueError(f"Expected update_configs to be a sequence, but got {type(update_configs)}")

    merged_config = {}

    for update_conf in update_configs:
        if isinstance(update_conf, Mapping):
            update_cfg_dict = update_conf
        else:
            with open(Path(update_conf), "r") as uf:
                update_cfg_dict = yaml.safe_load(uf)
            if not isinstance(update_cfg_dict, Mapping):
                warnings.warn(f"Loaded file {update_conf} is not a mapping {type(update_cfg_dict)}")
                continue
        if isinstance(update_cfg_dict, dict):
            merged_config = deep_update(merged_config, update_cfg_dict, list_strategy=list_strategy)
    
    return merged_config


