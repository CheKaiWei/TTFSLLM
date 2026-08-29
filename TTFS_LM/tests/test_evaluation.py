import copy
import unittest

import torch

from ttfs_lm.config import QwenConfig
from ttfs_lm.data import PackedCorpus
from ttfs_lm.evaluation import evaluate_cache_equivalence, evaluate_model_pair
from ttfs_lm.modeling_ann import QwenForCausalLM


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        config = QwenConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            rope_theta=10_000.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            use_cache=False,
        )
        self.model = QwenForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
        self.corpus = PackedCorpus(input_ids, attention_mask, {"kind": "unit"})

    def test_identical_models_have_identical_ppl_and_logits(self):
        candidate = copy.deepcopy(self.model).eval()
        result = evaluate_model_pair(self.model, candidate, self.corpus, batch_size=1, device="cpu")
        self.assertEqual(result["reference"]["target_tokens"], self.corpus.target_tokens)
        self.assertEqual(result["comparison"]["comparison_tokens"], self.corpus.target_tokens)
        self.assertEqual(result["reference"]["perplexity"], result["candidate"]["perplexity"])
        self.assertEqual(result["comparison"]["max_abs"], 0.0)
        self.assertEqual(result["comparison"]["top1_agreement"], 1.0)
        self.assertAlmostEqual(result["comparison"]["mean_kl"], 0.0, places=7)

    def test_cache_equivalence_uses_complete_prefix(self):
        result = evaluate_cache_equivalence(
            self.model,
            self.corpus.input_ids[:1],
            self.corpus.attention_mask[:1],
        )
        self.assertLess(result["max_abs"], 1e-5)
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertEqual(result["cache_length"], self.corpus.input_ids.shape[1])


if __name__ == "__main__":
    unittest.main()
