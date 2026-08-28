import unittest

import torch

from ttfs_lm.calibration import RangeRegistry, RangeSpec
from ttfs_lm.config import QwenConfig, TTFSConfig
from ttfs_lm.conversion import convert_ann_to_ttfs
from ttfs_lm.modeling_ann import QwenForCausalLM
from ttfs_lm.modeling_ttfs import RangeViolationError, TTFSCodec, TTFSQwenForCausalLM
from ttfs_lm.workflow import calibrate_ann, tensor_comparison


class TTFSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        config = QwenConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10_000.0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            use_cache=False,
        )
        self.ann = QwenForCausalLM(config).eval()
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
        self.attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        self.registry = calibrate_ann(
            self.ann,
            [{"input_ids": self.input_ids, "attention_mask": self.attention_mask}],
            margin=0.1,
            include_decode=True,
        )

    def test_continuous_ttfs_matches_ann_on_valid_tokens(self):
        ttfs = TTFSQwenForCausalLM(self.ann, self.registry, TTFSConfig()).eval()
        with torch.no_grad():
            ann_logits = self.ann(
                self.input_ids, attention_mask=self.attention_mask, use_cache=False
            ).logits
            ttfs_logits = ttfs(
                self.input_ids, attention_mask=self.attention_mask, use_cache=False
            ).logits
        metrics = tensor_comparison(ann_logits, ttfs_logits, self.attention_mask[:, :, None])
        self.assertLess(metrics["max_abs"], 2e-5)
        self.assertGreater(metrics["cosine"], 0.99999)
        self.assertEqual(metrics["top1_agreement"], 1.0)
        self.assertFalse(any(item.below or item.above for item in ttfs.codec.telemetry.values()))

    def test_ttfs_cache_matches_no_cache(self):
        ttfs = TTFSQwenForCausalLM(self.ann, self.registry, TTFSConfig()).eval()
        with torch.no_grad():
            full = ttfs(self.input_ids, attention_mask=self.attention_mask, use_cache=False).logits
            cache = None
            pieces = []
            for index in range(self.input_ids.shape[1]):
                output = ttfs(
                    self.input_ids[:, index : index + 1],
                    attention_mask=self.attention_mask[:, : index + 1],
                    cache=cache,
                    use_cache=True,
                )
                cache = output.cache
                pieces.append(output.logits)
        cached = torch.cat(pieces, dim=1)
        metrics = tensor_comparison(full, cached, self.attention_mask[:, :, None])
        self.assertLess(metrics["max_abs"], 2e-5)
        self.assertEqual(metrics["top1_agreement"], 1.0)

    def test_ttfs_cache_reorder_preserves_encoded_times(self):
        ttfs = TTFSQwenForCausalLM(self.ann, self.registry, TTFSConfig()).eval()
        with torch.no_grad():
            output = ttfs(
                self.input_ids[:, :2],
                attention_mask=self.attention_mask[:, :2],
                use_cache=True,
            )
        cache = output.cache
        original = cache.entries[0].key.time.clone()
        indices = torch.tensor([1, 0, 1])
        cache.reorder(indices)
        torch.testing.assert_close(cache.entries[0].key.time, original.index_select(0, indices))
        self.assertEqual(cache.entries[0].key.range_name, "model.layers.0.self_attn.k_rope")

    def test_finite_precision_path_is_finite(self):
        ttfs = TTFSQwenForCausalLM(
            self.ann,
            self.registry,
            TTFSConfig(precision=1024, strict_ranges=False),
        ).eval()
        with torch.no_grad():
            output = ttfs(self.input_ids[:1], use_cache=False)
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_strict_codec_reports_range_violation(self):
        registry = RangeRegistry({"x": RangeSpec("x", -1.0, 1.0, 1, 2)})
        codec = TTFSCodec(registry, TTFSConfig(strict_ranges=True))
        with self.assertRaises(RangeViolationError):
            codec.encode("x", torch.tensor([0.0, 2.0]))

    def test_incomplete_registry_is_rejected_before_forward(self):
        registry = RangeRegistry({"model.embed_tokens": RangeSpec("model.embed_tokens", -1.0, 1.0, 1, 1)})
        with self.assertRaisesRegex(ValueError, "missing"):
            TTFSQwenForCausalLM(self.ann, registry)

    def test_conversion_rejects_wrong_checkpoint_ranges(self):
        self.ann.ttfs_provenance = {"model_id": "official", "revision": "abc"}
        registry = RangeRegistry(
            {name: self.registry[name] for name in self.registry.names()},
            {"model_id": "other", "revision": "abc"},
        )
        with self.assertRaisesRegex(ValueError, "model_id"):
            convert_ann_to_ttfs(self.ann, registry)

    def test_continuous_spiking_lm_head_matches_hybrid(self):
        full_ttfs = TTFSQwenForCausalLM(
            self.ann,
            self.registry,
            TTFSConfig(spike_lm_head=True),
        ).eval()
        hybrid = TTFSQwenForCausalLM(self.ann, self.registry, TTFSConfig()).eval()
        with torch.no_grad():
            expected = hybrid(self.input_ids[:1], use_cache=False).logits
            actual = full_ttfs(self.input_ids[:1], use_cache=False).logits
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
