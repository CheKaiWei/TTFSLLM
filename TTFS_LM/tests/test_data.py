import unittest

import torch

from ttfs_lm.data import pack_texts


class DummyTokenizer:
    eos_token_id = 9
    pad_token_id = 0
    name_or_path = "dummy"

    def __init__(self):
        self.vocabulary = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    def __call__(self, text, **_kwargs):
        return {"input_ids": [self.vocabulary[word] for word in text.split()]}

    def __len__(self):
        return 10


class DataTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_pack_texts_preserves_all_next_token_targets(self):
        corpus = pack_texts(["a b", "", "c"], self.tokenizer, context_length=4)
        expected_ids = torch.tensor([[1, 2, 9, 3], [3, 9, 0, 0]])
        expected_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        torch.testing.assert_close(corpus.input_ids, expected_ids)
        torch.testing.assert_close(corpus.attention_mask, expected_mask)
        self.assertEqual(corpus.metadata["token_count"], 5)
        self.assertEqual(corpus.target_tokens, 4)
        self.assertEqual(corpus.metadata["nonempty_rows"], 2)

    def test_max_tokens_is_deterministic_and_ends_with_eos(self):
        corpus = pack_texts(["a b c", "d e"], self.tokenizer, context_length=4, max_tokens=5)
        self.assertEqual(corpus.metadata["token_count"], 5)
        self.assertTrue(corpus.metadata["truncated"])
        valid_tokens = corpus.input_ids[corpus.attention_mask.bool()]
        self.assertEqual(valid_tokens[-1].item(), self.tokenizer.eos_token_id)

    def test_batches_can_be_reiterated(self):
        corpus = pack_texts(["a b c", "d e"], self.tokenizer, context_length=4)
        first = list(corpus.batches(1, "cpu"))
        second = list(corpus.batches(1, "cpu"))
        self.assertEqual(len(first), len(second))
        torch.testing.assert_close(first[0]["input_ids"], second[0]["input_ids"])


if __name__ == "__main__":
    unittest.main()
