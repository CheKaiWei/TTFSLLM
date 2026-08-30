import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from ttfs_lm.data import CorpusSource, load_corpus, pack_texts


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

    def test_git_text_source_is_pinned_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "source"
            repository.mkdir()
            subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
            (repository / "test.txt").write_text("a b\nc d\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "test.txt"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=TTFS Test",
                    "-c",
                    "user.email=ttfs@example.invalid",
                    "commit",
                    "-m",
                    "fixture",
                ],
                check=True,
                capture_output=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            source = CorpusSource(
                name="fixture",
                source="git_text",
                id=str(repository),
                config=None,
                revision=revision,
                split="test",
                text_field="text",
                path="test.txt",
                max_tokens=None,
            )
            corpus = load_corpus(
                source,
                self.tokenizer,
                context_length=4,
                source_cache_dir=root / "cache",
            )
            self.assertEqual(corpus.metadata["dataset_revision"], revision)
            self.assertEqual(corpus.metadata["source_rows"], 2)
            self.assertEqual(corpus.metadata["source_rows_scanned"], 2)
            self.assertEqual(len(corpus.metadata["source_file_sha256"]), 64)

    def test_invalid_corpus_source_is_rejected(self):
        source = CorpusSource(
            name="invalid",
            source="unknown",
            id="example",
            revision="revision",
            split="test",
            text_field="text",
            max_tokens=100,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported corpus source"):
            source.validate()

    def test_invalid_source_file_digest_is_rejected(self):
        source = CorpusSource(
            name="invalid",
            source="http_json",
            id="https://example.invalid/data.json",
            revision="revision",
            split="test",
            text_field="text",
            max_tokens=100,
            file_sha256="short",
        )
        with self.assertRaisesRegex(ValueError, "64-character"):
            source.validate()


if __name__ == "__main__":
    unittest.main()
