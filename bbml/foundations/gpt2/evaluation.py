from typing import Any

from bbml.evaluation.base_eval_lm import BaseFoundationLM


class GPT2FoundationLM(BaseFoundationLM):
    @property
    def eot_token_id(self) -> int:
        if not hasattr(self, "_eot_id"):
            self._eot_id = self.foundation.tokenizer.encode(
                "<|endoftext|>", allowed_special={"<|endoftext|>"}
            )[0]
        return self._eot_id

    @property
    def max_length(self) -> int:
        return self.foundation.config.block_size

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs: Any,
    ) -> list[int]:
        return self.foundation.tokenizer.encode_ordinary(string,)