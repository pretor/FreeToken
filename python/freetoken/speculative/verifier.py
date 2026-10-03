from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import torch


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying speculative draft tokens."""

    accepted_tokens: list[int]
    bonus_token: int | None
    num_accepted: int

    @property
    def total_emitted(self) -> int:
        """Total tokens produced in this step (accepted drafts + bonus/correction token)."""
        return self.num_accepted + (1 if self.bonus_token is not None else 0)

    @property
    def all_tokens(self) -> list[int]:
        res = list(self.accepted_tokens)
        if self.bonus_token is not None:
            res.append(self.bonus_token)
        return res


class SpeculativeVerifier:
    """Verifies candidate draft tokens against target model predictions.

    Under greedy decoding, verification ensures bit-for-bit equivalence with
    standard sequential autoregressive generation while accepting multiple tokens per step.
    """

    @staticmethod
    def verify_greedy(
        draft_tokens: Sequence[int],
        predicted_tokens: Sequence[int] | torch.Tensor,
    ) -> VerificationResult:
        """Verifies draft tokens greedily against target model predictions.

        Args:
            draft_tokens: The candidate tokens proposed by the draft provider (length K).
            predicted_tokens: Model output token predictions for each evaluated position.
                              Length must be at least K (or K + 1 if predicting the next position).

        Returns:
            A VerificationResult with accepted tokens and the bonus/correction token.
        """
        if isinstance(predicted_tokens, torch.Tensor):
            preds = predicted_tokens.tolist()
        else:
            preds = list(predicted_tokens)

        accepted: list[int] = []
        k = len(draft_tokens)

        for i in range(k):
            if i >= len(preds):
                break
            target_token = preds[i]
            draft_token = draft_tokens[i]

            if target_token == draft_token:
                accepted.append(draft_token)
            else:
                # First mismatch: target model prediction becomes the correction token
                return VerificationResult(
                    accepted_tokens=accepted,
                    bonus_token=target_token,
                    num_accepted=len(accepted),
                )

        # All draft tokens were accepted!
        # If the model evaluated K+1 positions, the (K)th prediction is the bonus token
        bonus = preds[k] if len(preds) > k else None
        return VerificationResult(
            accepted_tokens=accepted,
            bonus_token=bonus,
            num_accepted=len(accepted),
        )
