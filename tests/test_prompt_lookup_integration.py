import unittest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.config import SchedulerConfig
from freetoken.speculative import (
    PromptLookupDraftProvider,
    SpeculativeVerifier,
    DraftCandidate,
)


class MockCacheHandle:
    def __init__(self, cached_len=0):
        self.cached_len = cached_len


class TestPromptLookupIntegration(unittest.TestCase):
    def test_scheduler_config_defaults(self):
        from freetoken.scheduler.config import SchedulerConfig
        self.assertFalse(SchedulerConfig.enable_prompt_lookup)
        self.assertEqual(SchedulerConfig.prompt_lookup_ngram, 3)
        self.assertEqual(SchedulerConfig.prompt_lookup_max_draft, 4)

    def test_req_preserves_prompt_len(self):
        prompt_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)
        req = Req(
            input_ids=prompt_ids,
            table_idx=0,
            cached_len=0,
            output_len=50,
            uid=1,
            sampling_params=SamplingParams(),
            cache_handle=MockCacheHandle(0),
        )
        # Should record prompt_len
        self.assertEqual(req.prompt_len, 5)

        # Append generated tokens
        req.append_host(torch.tensor([10, 20, 30], dtype=torch.int32))
        self.assertEqual(req.prompt_len, 5)
        self.assertEqual(len(req.input_ids), 8)

    def test_prompt_lookup_draft_generation(self):
        provider = PromptLookupDraftProvider(ngram_size=3, max_draft_len=4)
        
        # User prompt code: "class User: def __init__(self): self.name = None"
        prompt = [100, 200, 300, 400, 500, 600, 700, 800]
        # Generated so far in decode: repeats prefix "class User: def"
        generated = [100, 200, 300]
        full_seq = prompt + generated

        draft = provider.propose(full_seq, prompt_len=len(prompt))
        self.assertIsNotNone(draft)
        self.assertEqual(draft.ngram_size, 3)
        # Should predict next 4 tokens: [400, 500, 600, 700]
        self.assertEqual(draft.draft_tokens, [400, 500, 600, 700])

        # Verify against mock model predictions
        # Suppose model agrees on first 3 tokens, diverges on 4th
        model_predictions = [400, 500, 600, 999]
        res = SpeculativeVerifier.verify_greedy(draft.draft_tokens, model_predictions)
        
        self.assertEqual(res.accepted_tokens, [400, 500, 600])
        self.assertEqual(res.bonus_token, 999)
        self.assertEqual(res.total_emitted, 4)


    def test_batch_speculative_phase(self):
        from freetoken.core import Batch
        prompt_ids = torch.tensor([10, 20, 30], dtype=torch.int32)
        req = Req(
            input_ids=prompt_ids,
            table_idx=0,
            cached_len=0,
            output_len=50,
            uid=1,
            sampling_params=SamplingParams(),
            cache_handle=MockCacheHandle(0),
        )
        batch = Batch(reqs=[req], phase="speculative")
        self.assertTrue(batch.is_speculative)
        self.assertFalse(batch.is_decode)
        self.assertFalse(batch.is_prefill)

    def test_req_pending_draft_field(self):
        prompt_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
        req = Req(
            input_ids=prompt_ids,
            table_idx=0,
            cached_len=0,
            output_len=50,
            uid=1,
            sampling_params=SamplingParams(),
            cache_handle=MockCacheHandle(0),
        )
        self.assertIsNone(req.pending_draft)
        req.pending_draft = [4, 5, 6]
        self.assertEqual(req.pending_draft, [4, 5, 6])

if __name__ == "__main__":
    unittest.main()
