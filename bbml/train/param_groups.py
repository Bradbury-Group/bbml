from __future__ import annotations

import re
import warnings
from typing import Any

import torch.nn as nn


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    base_wd: float,
    rules: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build optimizer param groups from named_parameters using regex rules.

    Args:
        model: Module to extract trainable parameters from.
        base_lr: Default learning rate for unmatched params.
        base_wd: Default weight decay for unmatched params.
        rules: List of rule dicts. Each rule:
            - match: regex pattern to match parameter names
            - name: (optional) group name for logging
            - lr_mult: (optional) multiplier on base_lr (default 1.0)
            - lr: (optional) absolute LR, overrides lr_mult
            - weight_decay: (optional) override base_wd

    Returns:
        List of param group dicts ready for optimizer.

    Example:
        rules = [
            {"match": "lora_", "lr_mult": 10.0, "weight_decay": 0.0},
            {"match": "^head\\.", "lr_mult": 1.0},
            {"match": "^mixer\\.", "lr_mult": 2.0},
        ]
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if not rules:
        # No rules: single group with all trainable params
        return [{"params": [p for _, p in named], "lr": base_lr, "weight_decay": base_wd}]

    compiled: list[tuple[dict, re.Pattern]] = []
    for r in rules:
        pat = re.compile(r["match"])
        compiled.append((r, pat))

    # Init groups: index 0 is default (unmatched), rest follow rules order
    groups: list[dict[str, Any]] = [
        {"name": "default", "params": [], "lr": base_lr, "weight_decay": base_wd}
    ]
    for r in rules:
        lr = r.get("lr", base_lr * r.get("lr_mult", 1.0))
        groups.append({
            "name": r.get("name", r["match"]),
            "params": [],
            "lr": lr,
            "weight_decay": r.get("weight_decay", base_wd),
        })

    # first matching rule wins
    for name, p in named:
        matched = False
        for i, (_, pat) in enumerate(compiled):
            if pat.search(name):
                groups[i + 1]["params"].append(p)
                matched = True
                break
        if not matched:
            groups[0]["params"].append(p)

    # Warn on unused rules
    for i, (r, _) in enumerate(compiled):
        if not groups[i + 1]["params"]:
            warnings.warn(f"param_group_rule '{r.get('name', r['match'])}' matched no parameters")

    # Drop empty groups, remove 'name' key
    result = []
    for g in groups:
        if g["params"]:
            name = g.pop("name", None)
            result.append(g)

    return result
