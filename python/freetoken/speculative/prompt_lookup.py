from __future__ import annotations

from typing import Sequence
import torch

from .base import BaseDraftProvider, DraftCandidate


class PromptLookupDraftProvider(BaseDraftProvider):
    """Prompt Lookup Decoding (PLD) draft provider.

    Implements fast N-gram pattern matching across the prompt and generated context
    to propose speculative token candidates without requiring an auxiliary draft model.

    SOLID Principles:
        - Single Responsibility Principle (SRP): Only responsible for locating and proposing N-gram drafts.
        - Open/Closed Principle (OCP): Inherits BaseDraftProvider; parameters and search scopes are extensible.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        max_draft_len: int = 4,
        min_ngram_size: int = 2,
        search_scope: str = "prompt",
    ) -> None:
        if ngram_size < min_ngram_size:
            raise ValueError(f"ngram_size ({ngram_size}) cannot be less than min_ngram_size ({min_ngram_size})")
        if max_draft_len < 1:
            raise ValueError(f"max_draft_len ({max_draft_len}) must be at least 1")
        if search_scope not in ("prompt", "all"):
            raise ValueError(f"search_scope must be prompt or all, got {search_scope}")

        self.ngram_size = ngram_size
        self.max_draft_len = max_draft_len
        self.min_ngram_size = min_ngram_size
        self.search_scope = search_scope

    def propose(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        prompt_len: int,
    ) -> DraftCandidate | None:
        if isinstance(input_ids, torch.Tensor):
            tokens = input_ids.tolist()
        else:
            tokens = list(input_ids)

        total_len = len(tokens)
        if total_len < self.min_ngram_size + 1:
            return None

        # Determine the upper bound of the reference text to match against
        if self.search_scope == "prompt":
            ref_limit = min(prompt_len, total_len)
        else:
            ref_limit = total_len

        # Try from self.ngram_size down to self.min_ngram_size
        for n in range(min(self.ngram_size, total_len - 1), self.min_ngram_size - 1, -1):
            query = tokens[-n:]
            # Search space must be strictly prior to the active query window
            search_limit = min(ref_limit, total_len - n)

            candidate = self._find_ngram_draft(tokens, query, n, search_limit, ref_limit)
            if candidate is not None:
                return candidate

        return None

    def _find_ngram_draft(
        self,
        tokens: list[int],
        query: list[int],
        n: int,
        search_limit: int,
        ref_limit: int,
    ) -> DraftCandidate | None:
        q_tuple = tuple(query)
        # Search backwards for recency preference
        for i in range(search_limit - n, -1, -1):
            if tuple(tokens[i : i + n]) == q_tuple:
                draft_start = i + n
                draft_end = min(draft_start + self.max_draft_len, ref_limit)
                if draft_end > draft_start:
                    draft_tokens = tokens[draft_start:draft_end]
                    return DraftCandidate(
                        draft_tokens=draft_tokens,
                        ngram_size=n,
                        match_index=i,
                    )
        return None
