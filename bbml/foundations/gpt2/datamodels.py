

from typing import Literal

from pydantic import BaseModel, model_validator
from bbml.core.datamodels.configs import FoundationConfig


class GPTConfig(FoundationConfig):  # adapted
    from_hf: Literal['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']|None = "gpt2"

    weight_decay:float = 1e-1 # perhaps put in train config
    
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPTInput(BaseModel):
    text: str|None = None
    ids: list[int]|None = None
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int|None = None

class GPTOutput(BaseModel):
    text: str|None
    ids: list[int]|None
