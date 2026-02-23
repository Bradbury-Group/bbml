from typing import Any, Dict
import torch
from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightUnit, WeightIndex
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation


class GPT2WeightExtractor(WeightExtractor):
    def __init__(self):
        self.foundation = None
        self.config = None
    
    def load(self, model: Any, device: str = "cpu") -> "GPT2WeightExtractor":
        if isinstance(model, GPT2Foundation):
            self.foundation = model
            self.config = model.config
        else:
            raise ValueError("Model must be a GPT2Foundation instance")
        return self
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        if self.foundation is None or self.config is None:
            raise RuntimeError("Must call load() before get_config()")
        
        return {
            "n_layers": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_embd": self.config.n_embd,
            "vocab_size": self.config.vocab_size,
        }
    
    def extract_index(
        self,
        include_heads: bool = False,
        include_full: bool = True,
        include_ffn: bool = True,
    ) -> WeightIndex:
        if self.foundation is None:
            raise RuntimeError("Must call load() before extract_index()")
        
        units = []
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        n_embd = self.config.n_embd
        d_head = n_embd // n_head
        
        for layer_idx in range(n_layer):
            block = self.foundation.model.transformer.h[layer_idx]
            
            c_attn_weight = block.attn.c_attn.weight
            qkv = c_attn_weight.view(n_embd, 3, n_embd)
            q_weight = qkv[:, 0, :]
            k_weight = qkv[:, 1, :]
            v_weight = qkv[:, 2, :]
            
            if include_full:
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.attn.q.full",
                    tensor=q_weight.clone(),
                    kind="attn.q.full",
                    layer=layer_idx,
                ))
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.attn.k.full",
                    tensor=k_weight.clone(),
                    kind="attn.k.full",
                    layer=layer_idx,
                ))
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.attn.v.full",
                    tensor=v_weight.clone(),
                    kind="attn.v.full",
                    layer=layer_idx,
                ))
            
            if include_heads:
                q_heads = q_weight.view(n_embd, n_head, d_head)
                k_heads = k_weight.view(n_embd, n_head, d_head)
                v_heads = v_weight.view(n_embd, n_head, d_head)
                
                for head_idx in range(n_head):
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.attn.q.head{head_idx}",
                        tensor=q_heads[:, head_idx, :].clone(),
                        kind="attn.q.head",
                        layer=layer_idx,
                        head=head_idx,
                    ))
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.attn.k.head{head_idx}",
                        tensor=k_heads[:, head_idx, :].clone(),
                        kind="attn.k.head",
                        layer=layer_idx,
                        head=head_idx,
                    ))
                    units.append(WeightUnit(
                        key=f"layer{layer_idx}.attn.v.head{head_idx}",
                        tensor=v_heads[:, head_idx, :].clone(),
                        kind="attn.v.head",
                        layer=layer_idx,
                        head=head_idx,
                    ))
            
            if include_ffn:
                c_fc_weight = block.mlp.c_fc.weight
                c_proj_weight = block.mlp.c_proj.weight
                
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.ffn.up",
                    tensor=c_fc_weight.clone(),
                    kind="ffn.up",
                    layer=layer_idx,
                ))
                units.append(WeightUnit(
                    key=f"layer{layer_idx}.ffn.down",
                    tensor=c_proj_weight.clone(),
                    kind="ffn.down",
                    layer=layer_idx,
                ))
        
        return WeightIndex(units)
