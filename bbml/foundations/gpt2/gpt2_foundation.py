from pathlib import Path
from typing import Any
import warnings
import random

from safetensors.torch import load_file, save_model
import torch
from torch import Tensor
import tiktoken
from torch.nn import functional as F
from einops import rearrange

from bbml.core.datamodels.configs import FoundationConfig, TrainerConfig
from bbml.core.foundation import Foundation
from bbml.data.transforms import DataTransform
from bbml.foundations.gpt2.datamodels import GPTConfig, GPTInput, GPTOutput
from bbml.foundations.gpt2.model import GPT
from bbml import logger


class GPT2TextDataTransform(DataTransform):
    def __init__(self, block_size:int):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

    def transform(self, inp: str) -> Tensor:
        start_ind = random.randint(0, len(inp)-self.block_size)
        cropped_inp = inp[start_ind:start_ind+self.block_size]
        return torch.tensor(self.tokenizer.encode_ordinary(cropped_inp))
    
    def batch_transform(self, inp: list[Tensor]) -> Tensor:
        min_len = min(len(i) for i in inp)
        cropped_inp = [i[:min_len] for i in inp]
        return torch.stack(cropped_inp)
    

class GPT2Foundation(Foundation):

    def __init__(self, config: GPTConfig, train_config: TrainerConfig | None):
        super().__init__(config, train_config)
        if config.from_hf is None:
            self.model = GPT(config)
        else:
            self.model = self.from_hf(config.from_hf)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def single_step(self, batch: dict[str, Any]) -> Tensor:
        toks = batch["text"]  # [B, T]
        batch_size, length = toks.shape
        in_toks = toks[:, :length-1]
        if in_toks.size(1) > self.config.block_size:
            raise ValueError(f"Input length {in_toks.size(1)} > {self.config.block_size=}")
        out_toks = toks[:, 1:]

        logits = self.model(in_toks)  # [B, T-1, V]
        loss = F.cross_entropy(
            input=rearrange(logits, "B T C -> (B T) C"),
            target=rearrange(out_toks, "B T -> (B T)"),
            reduction="mean",
        )
        logger.log({"perplexity": torch.exp(loss).item()})

        return loss
        
    
    def get_train_parameters(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        return optim_groups
    
    @property
    def data_transforms(self) -> dict[str, DataTransform]:
        return {"text": GPT2TextDataTransform()}


    @property
    def input_model(self):
        return GPTInput

    @property
    def output_model(self):
        return GPTOutput
    
    def run(self, input: GPTInput) -> GPTOutput:
        if input.text is None and input.ids is None:
            # unconditional generation
            idx = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        elif input.ids is not None:
            idx = input.ids
        else:
            idx = self.tokenizer.encode(input.text, allowed_special={"<|endoftext|>"})

        for _ in range(input.max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / input.temperature
            # optionally crop the logits to only the top k options
            if input.top_k is not None:
                v, _ = torch.topk(logits, min(input.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        text = self.tokenizer.decode(idx)
        return GPTOutput(text=text, ids=idx)
        

    def load(self, load_path):
        if not isinstance(load_path, Path): 
            load_path = Path(load_path)
        if not load_path.is_dir():
            raise ValueError(f"Expected {load_path=} to be a directory")
        model_state_dict = load_file(load_path / f"model.safetensors")
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False, assign=True)
        if missing: 
            warnings.warn(f"model missing {missing}")
        if unexpected: 
            warnings.warn(f"model unexpected {unexpected}")

    def save(self, save_path):
        if not isinstance(save_path, Path): 
            save_path = Path(save_path)
        if not save_path.is_dir():
            raise ValueError(f"Expected {save_path=} to be a directory")
        save_path.mkdir(parents=True, exist_ok=True)
        save_model(self.model, save_path / f"model.safetensors")
        print(f"Saved model to {save_path}")
    

    def from_hf(self, model_type):
        if model_type not in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'")
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        
        # modify config to represent correct values
        self.config = self.config.model_copy(update=config_args)
        
        # create a from-scratch initialized minGPT model
        model = GPT(self.config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

