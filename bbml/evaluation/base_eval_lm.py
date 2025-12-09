from abc import abstractmethod
from typing import Any

import torch
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from bbml import Foundation, TrainerConfig


class BaseFoundationLM(TemplateLM):
    def __init__(
        self,
        foundation: Foundation,
        *,
        trainer_config: TrainerConfig | None = None,
    ) -> None:
        super().__init__()
        self.foundation = foundation

    # abstract methods
    @abstractmethod
    def tok_encode(
        self, string: str, add_special_tokens: bool|None = None, **kwargs
    ) -> list[int]:
        ...

    @property
    @abstractmethod
    def eot_token_id(self) -> int:
        ...

    @property
    @abstractmethod
    def max_length(self) -> int:
        ...

    # concrete integrations
    @property
    def device(self) -> torch.device:
        return self.foundation.device


    def _compute_loglikelihood(
        self,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:

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
            logits: Tensor = self.foundation(input_ids)  # [1, T-1, V]
            log_probs = F.log_softmax(logits, dim=-1)  # [1, T-1, V]

        # Gather log‑probs for the continuation next tokens.
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
        disable_tqdm: bool = False,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for (_, context_enc, continuation_enc) in tqdm(
            requests,
            disable=disable_tqdm,
            desc="Completion Loglikelihood",
        ):
            logprob, is_greedy = self._compute_loglikelihood(
                context_enc, continuation_enc
            )
            results.append((logprob, is_greedy))
        return results


    def loglikelihood_rolling(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False
    ) -> list[float]:
        loglikelihoods: list[float] = []

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Full-text loglikelihood",
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

        return loglikelihoods


    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False
    ) -> list[str]:
        generations: list[str] = []

        for context, gen_kwargs in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Text generation",
        ):

            gpt_input = self.foundation.input_model(
                text=context,
                **gen_kwargs,
            )
            output = self.foundation.run(gpt_input)
            full_text = output.text

            # Strip the prompt to return only the continuation.
            continuation = (
                full_text[len(context) :] if full_text.startswith(context) else full_text
            )

            generations.append(continuation)

        return generations
