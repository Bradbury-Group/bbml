import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
from torch.nn import Module, Parameter


def get_trainable_param_names(model: torch.nn.Module) -> set[str]:
    return {name for name, p in model.named_parameters() if p.requires_grad}


def filter_state_dict_to_names(
    state_dict: dict[str, torch.Tensor],
    names: set[str]
) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k in names}


def compute_structure_fingerprint(names: set[str]) -> str:
    """Hash of sorted names for drift detection"""
    sorted_names = sorted(names)
    content = "\n".join(sorted_names)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _iter_params_from_group(group: Any) -> Iterable[Parameter]:
    if group is None:
        return []
    if isinstance(group, Parameter):
        yield group
        return
    if isinstance(group, Mapping):
        params = group.get("params")
        if params is None:
            return
        yield from _iter_params_from_group(params)
        return
    if isinstance(group, (Sequence, set)):
        for item in group:
            yield from _iter_params_from_group(item)
        return
    # Fallback: treat as iterable
    if isinstance(group, Iterable):
        for item in group:
            yield from _iter_params_from_group(item)


def extract_trainable_param_names(
    model: Module,
    param_groups: Sequence[Any] | None = None,
) -> set[str]:
    """Derive trainable parameter names, optionally from optimizer groups.

    Args:
        model: Module providing named_parameters().
        param_groups: Optional optimizer param groups returned by get_train_parameters().

    Returns:
        Set of parameter names considered trainable.
    """
    name_by_id = {id(p): name for name, p in model.named_parameters()}
    names: set[str] = set()

    if param_groups:
        for group in param_groups:
            for param in _iter_params_from_group(group):
                name = name_by_id.get(id(param))
                if name:
                    names.add(name)

    if not names:
        # Fall back to requires_grad inspection.
        names = get_trainable_param_names(model)

    return names


def extract_delta_state(
    model: Module,
    trainable_names: set[str],
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if state_dict is None:
        state_dict = model.state_dict()
    assert_trainable_coverage(trainable_names, set(state_dict.keys()))
    return filter_state_dict_to_names(dict(state_dict), trainable_names)


def assert_trainable_coverage(
    trainable: set[str],
    state_dict_keys: set[str],
    min_frac: float = 0.95,
) -> None:
    present = trainable & state_dict_keys
    frac = len(present) / max(1, len(trainable))
    if frac < min_frac:
        missing = trainable - state_dict_keys
        missing_sample = list(missing)[:5]
        raise ValueError(
            f"Only {frac:.1%} of trainable params found in state_dict. "
            f"Missing examples: {missing_sample}. "
            f"Check if you need accelerator.get_state_dict(wrapped_model) vs unwrapped."
        )


def save_delta(
    delta: dict[str, torch.Tensor],
    path: Path | str,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    delta = {k: v.detach().to("cpu").contiguous() for k, v in delta.items()}
    safetensors.torch.save_file(delta, path / "delta.safetensors")
    with open(path / "meta.json", "w") as f: json.dump(metadata, f, indent=2)


def load_delta(path: Path | str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(path)
    delta = safetensors.torch.load_file(path / "delta.safetensors")
    with open(path / "meta.json") as f: metadata = json.load(f)
    return delta, metadata


def apply_delta_state(
    model: torch.nn.Module,
    delta: dict[str, torch.Tensor],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    result = model.load_state_dict(delta, strict=strict)
    missing, unexpected = result.missing_keys, result.unexpected_keys
    if unexpected: raise ValueError(f"Unexpected keys in delta: {unexpected[:5]}...")
    return missing, unexpected


def validate_checkpoint_meta(
    loaded_meta: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    errors = []
    if "format_version" in expected:
        loaded_ver = loaded_meta.get("format_version")
        expected_ver = expected.get("format_version")
        if loaded_ver != expected_ver:
            errors.append(
                f"Format version mismatch: checkpoint={loaded_ver}, expected={expected_ver}"
            )

    if loaded_meta.get("base_id") != expected.get("base_id"):
        errors.append(
            f"Base model mismatch: checkpoint={loaded_meta.get('base_id')}, "
            f"expected={expected.get('base_id')}"
        )

    if loaded_meta.get("structure_fingerprint") != expected.get("structure_fingerprint"):
        errors.append(
            f"Structure fingerprint mismatch (trainable params differ): "
            f"checkpoint={loaded_meta.get('structure_fingerprint')}, "
            f"expected={expected.get('structure_fingerprint')}"
        )

    if errors: raise ValueError("Checkpoint validation failed:\n" + "\n".join(errors))
