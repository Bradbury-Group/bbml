"""FSDP2 mesh + ``fully_shard`` wrapping helpers.

Thin layer over ``torch.distributed.device_mesh.init_device_mesh`` and
``torch.distributed.fsdp.fully_shard``. The wrap recipe mirrors the
``pretrain-{drawing,opd,sliders}`` convention: shard each transformer block
individually, then shard the outer module that owns embeddings / norm /
time-mlps. Set ``reshard_after_forward_on_output=False`` on the last block
so its sharded weights stay materialised through the backward pass — this is
the load-bearing tweak that prevents an extra all-gather during the
output-projection backward.
"""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from bbml.fsdp.dist import checkpoint_wrap


def init_mesh(
    dims: Sequence[int],
    names: Sequence[str],
    device_type: str = "cuda",
) -> DeviceMesh:
    """Build a ``DeviceMesh`` over ``device_type`` with the supplied axes.

    Args:
        dims: per-axis sizes, e.g. ``(world_size,)`` for pure FSDP, or
            ``(replicate, shard)`` for HSDP.
        names: axis names matching ``dims``. The trainer queries these by
            name when wiring the mesh into ``fully_shard``.
        device_type: passed through to ``init_device_mesh``; defaults to cuda.

    Returns:
        A ``DeviceMesh`` ready to splat into ``fully_shard(mesh=...)``.
    """
    return init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))


def fully_shard_model(
    model: nn.Module,
    *,
    blocks: nn.ModuleList,
    output_module: nn.Module | None,
    mesh: DeviceMesh,
    policy: MixedPrecisionPolicy,
    reshard_after_forward_on_output: bool = False,
    activation_checkpoint_blocks: bool = True,
) -> None:
    """In-place FSDP2 shard of ``model`` with per-block + outer wrapping.

    Wrap order is load-bearing:

        1. Each block in ``blocks`` is replaced by
           ``checkpoint_wrap(blk)`` (when ``activation_checkpoint_blocks``)
           THEN ``fully_shard(blk, ...)``. Doing checkpoint BEFORE shard is
           required — reversing breaks the per-block grad-ckpt boundary.
        2. ``output_module`` (typically the final projection) is sharded with
           ``reshard_after_forward=reshard_after_forward_on_output``. Default
           is ``False`` so the output shard stays materialised through the
           backward pass, saving one all-gather.
        3. The outer ``model`` is sharded last to capture embeddings, norm,
           and any time/guidance MLPs that live above the block list.

    Caveats:
        - ``blocks`` MUST be an ``nn.ModuleList`` (not a plain list / tuple /
          other ``Sequence``). The function swaps each entry in-place via
          ``blocks[i] = checkpoint_wrap(blk)`` so the parent module sees the
          wrapped child; ``nn.ModuleList`` is the only container that
          re-registers the child correctly. Passing a plain list silently
          breaks the module tree and a tuple raises ``TypeError`` deep in
          the loop.
        - Any parameter set to ``requires_grad_(False)`` BEFORE
          ``fully_shard`` is skipped by FSDP2 and left as a full
          (un-sharded) tensor. To shard frozen branches anyway, freeze AFTER
          calling this function. Conversely, the trainable underlying module
          returned by ``LoraFinetuner.unwrap_peft_for_fsdp`` keeps adapter
          params trainable so this wrap call shards them.
        - Disabling ``activation_checkpoint_blocks`` without an alternative
          internal grad-ckpt mechanism (e.g. diffusers
          ``enable_gradient_checkpointing``) will OOM at the per-block
          activation store.

    Args:
        model: top-level module to shard in-place.
        blocks: transformer block list (e.g. ``model.transformer_blocks``);
            MUST be ``nn.ModuleList``. See caveats above.
        output_module: final pre-output projection module, or None to skip.
        mesh: FSDP shard mesh (typically the ``shard`` axis of the layout).
        policy: ``MixedPrecisionPolicy`` from :func:`default_mp_policy`.
        reshard_after_forward_on_output: if False (default), keep the output
            module's params materialised through backward to skip an extra
            all-gather. If True, fall back to the default reshard behaviour.
        activation_checkpoint_blocks: wrap every block in a checkpoint wrapper
            before sharding. Default True. Disable only when the inner
            transformer already provides its own grad-ckpt path.
    """
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(
            "fully_shard_model requires `blocks` to be an nn.ModuleList for "
            "in-place child swap; got "
            f"{type(blocks).__name__}. Wrap your block list with nn.ModuleList "
            "or pass model.transformer_blocks (already a ModuleList in diffusers)."
        )
    policy_kwargs = {"mp_policy": policy, "mesh": mesh}

    for i, blk in enumerate(blocks):
        target = blk
        if activation_checkpoint_blocks:
            target = checkpoint_wrap(blk)
            blocks[i] = target
        fully_shard(target, **policy_kwargs)

    if output_module is not None:
        fully_shard(
            output_module,
            **policy_kwargs,
            reshard_after_forward=reshard_after_forward_on_output,
        )

    fully_shard(model, **policy_kwargs)


__all__: list[str] = ["init_mesh", "fully_shard_model"]
