"""Seam 5 (reference): gpt2 foundation exposes fsdp_blocks so it runs under
FullyShardTrainer via the default parallelise()."""
import torch.nn as nn

from bbml.foundations.gpt2.datamodels import GPTConfig
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation


def test_fsdp_blocks_are_transformer_h():
    cfg = GPTConfig(from_hf=None, block_size=8, vocab_size=16, n_layer=2, n_head=2, n_embd=8)
    fnd = GPT2Foundation(cfg, None)
    blocks = fnd.fsdp_blocks()
    assert isinstance(blocks, nn.ModuleList)
    assert blocks is fnd.model.transformer.h
    assert len(blocks) == 2
