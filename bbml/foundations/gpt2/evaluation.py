from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM

from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.foundation import Foundation
from bbml.foundations.gpt2.datamodels import GPTInput


class BaseFoundationLM(TemplateLM):
    """
    Thin adapter that wraps a bbml ``Foundation`` (e.g. ``GPT2Foundation``)
    so it can be used as an ``LM`` within lm-evaluation-harness.

    This class assumes a *causal* decoder-only language model with:
    - a ``tokenizer`` attribute providing GPT-2 compatible tokenization
    - a ``model`` attribute that accepts token IDs of shape ``[B, T]`` and
      returns logits of shape ``[B, T, V]``
    - a config with ``block_size`` giving the max context length.
    """

    def __init__(
        self,
        foundation: Foundation,
        *,
        trainer_config: TrainerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        foundation:
            A concrete ``Foundation`` instance such as ``GPT2Foundation``.
        trainer_config:
            Optional trainer configuration. Included so this adapter can be
            constructed in more generic code paths if needed.
        """
        super().__init__()
        self.foundation = foundation

        # Convenience aliases
        self._model = foundation.model
        self._tokenizer = foundation.tokenizer
        self._device = torch.device(getattr(foundation, "device", "cpu"))
        self._block_size = getattr(foundation.config, "block_size", 1024)

    # ------------------------------------------------------------------
    # TemplateLM required tokenizer API
    # ------------------------------------------------------------------
    @property
    def eot_token_id(self) -> int:
        # GPT‑2 uses ``<|endoftext|>`` as the special end‑of‑text token.
        # Cache the value so we do the lookup only once.
        if not hasattr(self, "_eot_id"):
            self._eot_id = self._tokenizer.encode(
                "<|endoftext|>", allowed_special={"<|endoftext|>"}
            )[0]
        return self._eot_id

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_length(self) -> int:
        return self._block_size

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,  # kept for TemplateLM compatibility
        **_: Any,
    ) -> list[int]:
        """
        Tokenize a string using the underlying bbml tokenizer.

        ``add_special_tokens`` is accepted for API compatibility but ignored,
        since GPT‑2 style tokenizers typically do not add BOS/EOS by default.
        """

        # Allow the GPT‑2 EOT token if it appears in the text.
        return self._tokenizer.encode(
            string,
            allowed_special={"<|endoftext|>"},
        )

    # ------------------------------------------------------------------
    # Core loglikelihood implementation
    # ------------------------------------------------------------------
    def _compute_loglikelihood(
        self,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        """
        Compute log p(continuation | context) for a single example.
        """
        if not continuation_enc:
            return 0.0, True

        # Respect model context window by left‑truncating context only.
        total_len = len(context_enc) + len(continuation_enc)
        if total_len > self.max_length:
            overflow = total_len - self.max_length
            context_enc = context_enc[overflow:]

        if len(continuation_enc) > self.max_length:
            msg = (
                f"Continuation length {len(continuation_enc)} exceeds model "
                f"context window {self.max_length}."
            )
            raise ValueError(msg)

        tokens = context_enc + continuation_enc
        ctx_len = len(context_enc)

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=self.device)[
            None, :
        ]  # [1, T-1]
        target_ids = torch.tensor(tokens[1:], dtype=torch.long, device=self.device)[
            None, :
        ]  # [1, T-1]

        with torch.no_grad():
            logits: Tensor = self._model(input_ids)  # [1, T-1, V]
            log_probs = F.log_softmax(logits, dim=-1)  # [1, T-1, V]

        # Gather log‑probs for the actual next tokens.
        token_logprobs = log_probs.gather(  # [1, T-1, 1] -> [1, T-1]
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Continuation tokens start at position ``ctx_len`` in the combined
        # token sequence, which corresponds to index ``ctx_len - 1`` in targets.
        cont_start = max(ctx_len - 1, 0)
        cont_logprobs = token_logprobs[:, cont_start:]

        total_logprob = float(cont_logprobs.sum().item())

        # Greedy correctness: check argmax against targets on continuation slice.
        greedy_tokens = logits.argmax(dim=-1)[:, cont_start:]
        target_slice = target_ids[:, cont_start:]
        is_greedy = bool((greedy_tokens == target_slice).all().item())

        return total_logprob, is_greedy

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        **_: Any,
    ) -> list[tuple[float, bool]]:
        """
        Concrete implementation expected by ``TemplateLM.loglikelihood``.

        Each request is a triple:
            ((context_str, continuation_str) | None, context_enc, continuation_enc)
        """
        results: list[tuple[float, bool]] = []
        for _, context_enc, continuation_enc in requests:
            logprob, is_greedy = self._compute_loglikelihood(
                context_enc, continuation_enc
            )
            results.append((logprob, is_greedy))
        return results

    # ------------------------------------------------------------------
    # Rolling loglikelihood (perplexity‑style)
    # ------------------------------------------------------------------
    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        """
        Full‑document loglikelihood using a rolling window over long texts.
        """
        from tqdm import tqdm  # imported lazily to keep core deps minimal

        loglikelihoods: list[float] = []

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running rolling loglikelihood",
        ):
            token_list = self.tok_encode(string)
            rolling_windows = map(
                utils.make_disjoint_window,
                utils.get_rolling_token_windows(
                    token_list=token_list,
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            )

            total_lp = 0.0
            for ctx_ids, cont_ids in rolling_windows:
                lp, _ = self._compute_loglikelihood(ctx_ids, cont_ids)
                total_lp += lp

            loglikelihoods.append(total_lp)
            # enable partial caching for CachingLM
            self.cache_hook.add_partial(
                "loglikelihood_rolling",
                (string,),
                total_lp,
            )

        return loglikelihoods

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """
        Greedy / sampling generation using the underlying ``Foundation.run``.

        The harness passes ``(context, gen_kwargs)`` in ``Instance.args``.
        We map a minimal subset of ``gen_kwargs`` into ``GPTInput`` and
        return only the continuation (without the original context prefix).
        """
        from tqdm import tqdm  # imported lazily

        generations: list[str] = []

        for context, gen_kwargs in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running generate_until requests",
        ):
            if not isinstance(gen_kwargs, dict):
                msg = (
                    "Expected gen_kwargs to be a dict, "
                    f"but received {type(gen_kwargs)}"
                )
                raise TypeError(msg)

            max_new_tokens = int(
                gen_kwargs.get(
                    "max_gen_toks",
                    gen_kwargs.get("max_new_tokens", 256),
                )
            )
            temperature = float(gen_kwargs.get("temperature", 1.0))
            top_k_val = gen_kwargs.get("top_k")
            top_k = int(top_k_val) if top_k_val is not None else None

            gpt_input = GPTInput(
                text=context,
                ids=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            output = self.foundation.run(gpt_input)
            full_text = output.text or ""

            # Strip the prompt to return only the continuation.
            continuation = (
                full_text[len(context) :] if full_text.startswith(context) else full_text
            )

            generations.append(continuation)
            self.cache_hook.add_partial(
                "generate_until",
                (context, gen_kwargs),
                continuation,
            )

        return generations
