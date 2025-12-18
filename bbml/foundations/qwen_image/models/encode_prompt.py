import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

tokenizer_max_length = 1024
prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
prompt_template_encode_start_idx = 34
default_sample_size = 128

def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result


def encode_prompt(
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    tokenizer: Qwen2Tokenizer,
    prompt: str | list[str],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_sequence_length: int|None = 1024,
):


    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]
    txt_tokens = tokenizer(
        txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    prompt_embeds_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if max_sequence_length is not None:
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

    return prompt_embeds, prompt_embeds_mask
