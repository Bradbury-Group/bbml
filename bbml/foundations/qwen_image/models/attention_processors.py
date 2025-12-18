import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention

from .transformer_qwenimage import apply_rotary_emb_qwen

try:
    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp16_triton,
    )
except ImportError:
    sageattn = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    warnings.warn("Sageattention not imported")

try:
    from kernels import get_kernel
    _k = get_kernel("kernels-community/vllm-flash-attn3")
    _flash_attn_func = _k.flash_attn_func
except Exception as e:
    _flash_attn_func = None
    _kernels_err = e


def _get_projections(attn: Attention, hidden_states, encoder_hidden_states):
    img_q = attn.to_q(hidden_states)
    img_k = attn.to_k(hidden_states)
    img_v = attn.to_v(hidden_states)

    txt_q = attn.add_q_proj(encoder_hidden_states)
    txt_k = attn.add_k_proj(encoder_hidden_states)
    txt_v = attn.add_v_proj(encoder_hidden_states)

    return img_q, img_k, img_v, txt_q, txt_k, txt_v


def _get_fused_projections(attn: Attention, hidden_states, encoder_hidden_states):
    img_q, img_k, img_v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    txt_q, txt_k, txt_v = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return img_q, img_k, img_v, txt_q, txt_k, txt_v

def attn_get_qkv_projections(attn: Attention, hidden_states, encoder_hidden_states):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)

def _ensure_fa3_available():
    if _flash_attn_func is None:
        raise ImportError(
            "FlashAttention-3 via Hugging Face `kernels` is required. "
            "Tried `get_kernel('kernels-community/vllm-flash-attn3')` and failed with:\n"
            f"{_kernels_err}"
        )

@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    outputs, lse = _flash_attn_func(q, k, v, causal=causal)
    return outputs

@flash_attn_func.register_fake
def _fa(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    meta_q = torch.empty_like(q).contiguous()
    return meta_q #, q.new_empty((q.size(0), q.size(2), q.size(1)), dtype=torch.float32)


@torch.library.custom_op("sage::sageattn", mutates_args=())
def sageattn_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    outputs = sageattn(q, k, v)
    return outputs

@sageattn_wrapper.register_fake
def _sageattn_wrapper_fake(q, k, v):
    meta_q = torch.empty_like(q).contiguous()
    return meta_q

@torch.library.custom_op("sage::sageattn_qk_int8_pv_fp16_cuda", mutates_args=())
def sageattn_qk_int8_pv_fp16_cuda_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    outputs = sageattn_qk_int8_pv_fp16_cuda(q, k, v)
    return outputs

@sageattn_qk_int8_pv_fp16_cuda_wrapper.register_fake
def _sageattn_qk_int8_pv_fp16_cuda_wrapper_fake(q, k, v):
    meta_q = torch.empty_like(q).contiguous()
    return meta_q


@torch.library.custom_op("sage::sageattn_qk_int8_pv_fp16_triton", mutates_args=())
def sageattn_qk_int8_pv_fp16_triton_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    outputs = sageattn_qk_int8_pv_fp16_triton(q, k, v)
    return outputs

@sageattn_qk_int8_pv_fp16_triton_wrapper.register_fake
def _sageattn_qk_int8_pv_fp16_triton_wrapper_fake(q, k, v):
    meta_q = torch.empty_like(q).contiguous()
    return meta_q

@torch.library.custom_op("sage::sageattn_qk_int8_pv_fp8_cuda", mutates_args=())
def sageattn_qk_int8_pv_fp8_cuda_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    outputs = sageattn_qk_int8_pv_fp8_cuda(q, k, v)
    return outputs

@sageattn_qk_int8_pv_fp8_cuda_wrapper.register_fake
def _sageattn_qk_int8_pv_fp8_cuda_wrapper_fake(q, k, v):
    meta_q = torch.empty_like(q).contiguous()
    return meta_q

@torch.library.custom_op("sage::sageattn_qk_int8_pv_fp8_cuda_sm90", mutates_args=())
def sageattn_qk_int8_pv_fp8_cuda_sm90_wrapper(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    outputs = sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v)
    return outputs

@sageattn_qk_int8_pv_fp8_cuda_sm90_wrapper.register_fake
def _sageattn_qk_int8_pv_fp8_cuda_sm90_wrapper_fake(q, k, v):
    meta_q = torch.empty_like(q).contiguous()
    return meta_q



class QwenDoubleStreamAttnProcessorFA3:
    """
    FA3-based attention processor for Qwen double-stream architecture.
    Computes joint attention over concatenated [text, image] streams using vLLM FlashAttention-3
    accessed via Hugging Face `kernels`.

    Notes / limitations:
    - General attention masks are not supported here (FA3 path). `is_causal=False` and no arbitrary mask.
    - Optional windowed attention / sink tokens / softcap can be plumbed through if you use those features.
    - Expects an available `apply_rotary_emb_qwen` in scope (same as your non-FA3 processor).
    """

    _attention_backend = "fa3"  # for parity with your other processors, not used internally

    def __init__(self):
        _ensure_fa3_available()

    @torch.no_grad()
    def __call__(
        self,
        attn,  # Attention module with to_q/to_k/to_v/add_*_proj, norms, to_out, to_add_out, and .heads
        hidden_states: torch.FloatTensor,                 # (B, S_img, D_model)  image stream
        encoder_hidden_states: torch.FloatTensor = None,  # (B, S_txt, D_model)  text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,  # unused in FA3 path
        attention_mask: Optional[torch.FloatTensor] = None,    # unused in FA3 path
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (img_freqs, txt_freqs)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessorFA3 requires encoder_hidden_states (text stream).")
        if attention_mask is not None:
            # FA3 kernel path here does not consume arbitrary masks; fail fast to avoid silent correctness issues.
            raise NotImplementedError("attention_mask is not supported in this FA3 implementation.")

        img_q, img_k, img_v, txt_q, txt_k, txt_v = attn_get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        
        
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]


        # ---- Reshape to (B, S, H, D_h) ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K normalization (per your module contract) ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen variant) ----
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # expects tensors shaped (B, S, H, D_h)
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ---- Joint attention over [text, image] along sequence axis ----
        # Shapes: (B, S_total, H, D_h)
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)

        # FlashAttention-3 path expects (B, S, H, D_h) and returns (out, softmax_lse)
        out = flash_attn_func(q, k, v, causal=False)  # out: (B, S_total, H, D_h)

        # ---- Back to (B, S, D_model) ----
        out = out.flatten(2, 3)#.to(q.dtype)

        # Split back to text / image segments
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output projections ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # dropout if present

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None #"_native_flash"

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        img_query, img_key, img_value, txt_query, txt_key, txt_value = attn_get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        # joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output



class QwenDoubleStreamAttnProcessorSageAttn2:
    def __init__(self, sageattn_func):
        self.sageattn_func = sageattn_func

    @torch.no_grad()
    def __call__(
        self,
        attn,  # Attention module with to_q/to_k/to_v/add_*_proj, norms, to_out, to_add_out, and .heads
        hidden_states: torch.FloatTensor,                 # (B, S_img, D_model)  image stream
        encoder_hidden_states: torch.FloatTensor = None,  # (B, S_txt, D_model)  text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,  # unused in FA3 path
        attention_mask: Optional[torch.FloatTensor] = None,    # unused in FA3 path
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (img_freqs, txt_freqs)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessorFA3 requires encoder_hidden_states (text stream).")
        if attention_mask is not None:
            # FA3 kernel path here does not consume arbitrary masks; fail fast to avoid silent correctness issues.
            raise NotImplementedError("attention_mask is not supported in this FA3 implementation.")

        img_q, img_k, img_v, txt_q, txt_k, txt_v = attn_get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        
        
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]


        # ---- Reshape to (B, S, H, D_h) ----
        H = attn.heads
        img_q = img_q.unflatten(-1, (H, -1))
        img_k = img_k.unflatten(-1, (H, -1))
        img_v = img_v.unflatten(-1, (H, -1))

        txt_q = txt_q.unflatten(-1, (H, -1))
        txt_k = txt_k.unflatten(-1, (H, -1))
        txt_v = txt_v.unflatten(-1, (H, -1))

        # ---- Q/K normalization (per your module contract) ----
        if getattr(attn, "norm_q", None) is not None:
            img_q = attn.norm_q(img_q)
        if getattr(attn, "norm_k", None) is not None:
            img_k = attn.norm_k(img_k)
        if getattr(attn, "norm_added_q", None) is not None:
            txt_q = attn.norm_added_q(txt_q)
        if getattr(attn, "norm_added_k", None) is not None:
            txt_k = attn.norm_added_k(txt_k)

        # ---- RoPE (Qwen variant) ----
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # expects tensors shaped (B, S, H, D_h)
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # ---- Joint attention over [text, image] along sequence axis ----
        # Shapes: (B, S_total, H, D_h)
        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)
        

        # sage attention
        q = q.transpose(1, 2) # (B, H, S, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = self.sageattn_func(q, k, v)  # out: (B, H, S, D_h)
        out = out.transpose(1, 2) # to (B, S, H, D_h)

        # ---- Back to (B, S, D_model) ----
        out = out.flatten(2, 3)#.to(q.dtype)

        # Split back to text / image segments
        txt_attn_out = out[:, :S_txt, :]
        img_attn_out = out[:, S_txt:, :]

        # ---- Output projections ----
        img_attn_out = attn.to_out[0](img_attn_out)
        if len(attn.to_out) > 1:
            img_attn_out = attn.to_out[1](img_attn_out)  # dropout if present

        txt_attn_out = attn.to_add_out(txt_attn_out)

        return img_attn_out, txt_attn_out
