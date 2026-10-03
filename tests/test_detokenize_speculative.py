import unittest
from unittest.mock import MagicMock
from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.message import DetokenizeMsg

class DummyTokenizer:
    def batch_decode(self, sequences):
        vocab = {101: " hello", 102: " world", 103: "!", 201: " foo", 202: " bar"}
        out = []
        for seq in sequences:
            out.append("".join(vocab.get(t, f"_{t}") for t in seq))
        return out

class TestDetokenizeSpeculative(unittest.TestCase):
    def test_single_uid_multiple_tokens_no_repetition(self):
        tok = DummyTokenizer()
        dm = DetokenizeManager(tok, eos_token_ids=frozenset({999}))
        msgs = [
            DetokenizeMsg(uid=1, next_token=101, finished=False),
            DetokenizeMsg(uid=1, next_token=102, finished=False),
            DetokenizeMsg(uid=1, next_token=103, finished=True),
        ]
        results = dm.detokenize(msgs)
        self.assertEqual(results, [" hello", " world", "!"])

    def test_mixed_uids_multiple_tokens(self):
        tok = DummyTokenizer()
        dm = DetokenizeManager(tok, eos_token_ids=frozenset({999}))
        msgs = [
            DetokenizeMsg(uid=1, next_token=101, finished=False),
            DetokenizeMsg(uid=2, next_token=201, finished=False),
            DetokenizeMsg(uid=1, next_token=102, finished=False),
            DetokenizeMsg(uid=2, next_token=202, finished=True),
            DetokenizeMsg(uid=1, next_token=103, finished=True),
        ]
        results = dm.detokenize(msgs)
        self.assertEqual(results, [" hello", " foo", " world", " bar", "!"])

if __name__ == "__main__":
    unittest.main()
