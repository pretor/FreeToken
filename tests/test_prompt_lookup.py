import unittest
import torch

from freetoken.speculative import (
    DraftCandidate,
    PromptLookupDraftProvider,
    SpeculativeVerifier,
    VerificationResult,
)


class TestPromptLookupDraftProvider(unittest.TestCase):
    def setUp(self):
        self.provider = PromptLookupDraftProvider(
            ngram_size=3,
            max_draft_len=4,
            min_ngram_size=2,
        )

    def test_basic_match_in_prompt(self):
        prompt = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        tokens = prompt + [10, 20, 30]

        candidate = self.provider.propose(tokens, prompt_len=len(prompt))
        self.assertIsNotNone(candidate)
        self.assertIsInstance(candidate, DraftCandidate)
        self.assertEqual(candidate.draft_tokens, [40, 50, 60, 70])
        self.assertEqual(candidate.ngram_size, 3)
        self.assertEqual(candidate.match_index, 0)

    def test_fallback_to_smaller_ngram(self):
        prompt = [10, 20, 30, 40, 50]
        tokens = prompt + [99, 20, 30]

        candidate = self.provider.propose(tokens, prompt_len=len(prompt))
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.ngram_size, 2)
        self.assertEqual(candidate.draft_tokens, [40, 50])

    def test_no_match_returns_none(self):
        prompt = [10, 20, 30, 40]
        tokens = prompt + [90, 91, 92]

        candidate = self.provider.propose(tokens, prompt_len=len(prompt))
        self.assertIsNone(candidate)

    def test_short_sequence_returns_none(self):
        candidate = self.provider.propose([10, 20], prompt_len=2)
        self.assertIsNone(candidate)

    def test_torch_tensor_input(self):
        prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
        tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 1, 2, 3], dtype=torch.int32)

        candidate = self.provider.propose(tokens, prompt_len=7)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.draft_tokens, [4, 5, 6, 7])

    def test_recency_preference(self):
        prompt = [1, 2, 3, 10, 20, 1, 2, 3, 80, 90]
        tokens = prompt + [1, 2, 3]

        candidate = self.provider.propose(tokens, prompt_len=len(prompt))
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.match_index, 5)
        self.assertEqual(candidate.draft_tokens, [80, 90])


class TestSpeculativeVerifier(unittest.TestCase):
    def test_full_acceptance_with_bonus(self):
        draft = [10, 20, 30]
        predictions = [10, 20, 30, 40]

        res = SpeculativeVerifier.verify_greedy(draft, predictions)
        self.assertIsInstance(res, VerificationResult)
        self.assertEqual(res.accepted_tokens, [10, 20, 30])
        self.assertEqual(res.bonus_token, 40)
        self.assertEqual(res.num_accepted, 3)
        self.assertEqual(res.total_emitted, 4)
        self.assertEqual(res.all_tokens, [10, 20, 30, 40])

    def test_partial_acceptance_with_correction(self):
        draft = [10, 20, 30, 40]
        predictions = [10, 20, 99, 50, 60]

        res = SpeculativeVerifier.verify_greedy(draft, predictions)
        self.assertEqual(res.accepted_tokens, [10, 20])
        self.assertEqual(res.bonus_token, 99)
        self.assertEqual(res.num_accepted, 2)
        self.assertEqual(res.total_emitted, 3)
        self.assertEqual(res.all_tokens, [10, 20, 99])

    def test_complete_rejection(self):
        draft = [10, 20, 30]
        predictions = [99, 10, 20]

        res = SpeculativeVerifier.verify_greedy(draft, predictions)
        self.assertEqual(res.accepted_tokens, [])
        self.assertEqual(res.bonus_token, 99)
        self.assertEqual(res.num_accepted, 0)
        self.assertEqual(res.total_emitted, 1)
        self.assertEqual(res.all_tokens, [99])

    def test_tensor_predictions(self):
        draft = [1, 2]
        predictions = torch.tensor([1, 2, 3], dtype=torch.int32)

        res = SpeculativeVerifier.verify_greedy(draft, predictions)
        self.assertEqual(res.accepted_tokens, [1, 2])
        self.assertEqual(res.bonus_token, 3)
        self.assertEqual(res.num_accepted, 2)


if __name__ == "__main__":
    unittest.main()
