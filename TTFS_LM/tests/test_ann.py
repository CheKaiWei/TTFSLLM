import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ttfs_lm.config import QwenConfig
from ttfs_lm.modeling_ann import QwenForCausalLM, masked_softmax
from ttfs_lm.workflow import tensor_comparison


def tiny_hf_config() -> Qwen2Config:
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return config


def make_models():
    torch.manual_seed(17)
    hf_config = tiny_hf_config()
    reference = Qwen2ForCausalLM(hf_config).eval()
    local = QwenForCausalLM(QwenConfig.from_hf_config(hf_config)).eval()
    local.load_state_dict(reference.state_dict(), strict=True)
    local.tie_weights()
    return reference, local


class ANNTest(unittest.TestCase):
    def setUp(self):
        self.reference, self.local = make_models()
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
        self.attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    def test_state_dict_and_forward_match_hugging_face(self):
        with torch.no_grad():
            expected = self.reference(
                self.input_ids, attention_mask=self.attention_mask, use_cache=False
            ).logits
            actual = self.local(
                self.input_ids, attention_mask=self.attention_mask, use_cache=False
            ).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(
            self.local.lm_head.weight.data_ptr(), self.local.model.embed_tokens.weight.data_ptr()
        )

    def test_future_tokens_do_not_change_prefix(self):
        changed = self.input_ids.clone()
        changed[0, 3:] = torch.tensor([30, 31])
        with torch.no_grad():
            original = self.local(self.input_ids[:1], use_cache=False).logits
            modified = self.local(changed[:1], use_cache=False).logits
        torch.testing.assert_close(original[:, :3], modified[:, :3], atol=1e-6, rtol=1e-6)

    def test_cache_matches_full_forward(self):
        with torch.no_grad():
            full = self.local(
                self.input_ids, attention_mask=self.attention_mask, use_cache=False
            ).logits
            cache = None
            pieces = []
            for index in range(self.input_ids.shape[1]):
                output = self.local(
                    self.input_ids[:, index : index + 1],
                    attention_mask=self.attention_mask[:, : index + 1],
                    cache=cache,
                    use_cache=True,
                )
                cache = output.cache
                pieces.append(output.logits)
        cached = torch.cat(pieces, dim=1)
        metrics = tensor_comparison(full, cached, self.attention_mask[:, :, None])
        self.assertLess(metrics["max_abs"], 1e-5)
        self.assertEqual(metrics["top1_agreement"], 1.0)

    def test_ann_cache_reorder_changes_batch_order(self):
        with torch.no_grad():
            output = self.local(
                self.input_ids[:, :2],
                attention_mask=self.attention_mask[:, :2],
                use_cache=True,
            )
        cache = output.cache
        original = cache.entries[0].key.clone()
        indices = torch.tensor([1, 0, 1])
        cache.reorder(indices)
        torch.testing.assert_close(cache.entries[0].key, original.index_select(0, indices))
        self.assertEqual(cache.entries[0].key.shape[0], 3)

    def test_masked_softmax_has_exact_zeros(self):
        logits = torch.tensor([[[[2.0, 1.0, 9.0]]]])
        valid = torch.tensor([[[[True, True, False]]]])
        probabilities = masked_softmax(logits, valid)
        self.assertEqual(probabilities[..., 2].item(), 0.0)
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1, 1, 1))

    def test_all_masked_softmax_row_is_zero_not_nan(self):
        probabilities = masked_softmax(torch.ones(1, 1, 1, 3), torch.zeros(1, 1, 1, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertEqual(probabilities.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
