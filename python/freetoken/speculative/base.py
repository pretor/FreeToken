from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DraftCandidate:
    """Represents a set of speculative draft tokens proposed for a sequence."""

    draft_tokens: list[int]
    ngram_size: int
    match_index: int | None = None

    def __len__(self) -> int:
        return len(self.draft_tokens)

    @property
    def is_empty(self) -> bool:
        return len(self.draft_tokens) == 0

    @property
    def has_draft(self) -> bool:
        return len(self.draft_tokens) > 0


class BaseDraftProvider(ABC):
    """Abstract interface for draft token providers (Strategy pattern).

    Decouples draft generation logic (Prompt Lookup, MTP, auxiliary draft model)
    from the scheduler and engine execution pipelines.
    """

    @abstractmethod
    def propose(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        prompt_len: int,
    ) -> DraftCandidate | None:
        """Proposes candidate draft tokens given the full sequence history.

        Args:
            input_ids: The sequence of token IDs (including prompt and already generated tokens).
            prompt_len: The number of tokens in the initial user prompt.

        Returns:
            A DraftCandidate containing proposed tokens, or None if no candidate could be formed.
        """
        pass
