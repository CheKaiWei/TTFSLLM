import os
import unittest

import torch

from ttfs_lm.checkpoint import load_official_pair, load_tokenizer
from ttfs_lm.config import DEFAULT_MODEL_ID, DEFAULT_REVISION
from ttfs_lm.workflow import tensor_comparison


@unittest.skipUnless(os.environ.get("TTFS_LM_RUN_OFFICIAL") == "1", "set TTFS_LM_RUN_OFFICIAL=1")
class OfficialCheckpointTest(unittest.TestCase):
    def test_official_qwen_checkpoint_matches_local_ann(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pair = load_official_pair(device=device, dtype="float32", local_files_only=True)
        tokenizer = load_tokenizer(local_files_only=True)
        inputs = tokenizer("ANN to SNN conversion with Qwen", return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items() if name in {"input_ids", "attention_mask"}}
        with torch.no_grad():
            expected = pair.reference(**inputs, use_cache=False).logits
            actual = pair.ann(**inputs, use_cache=False).logits
        metrics = tensor_comparison(expected, actual)
        self.assertEqual(pair.ann.config.num_hidden_layers, 24)
        self.assertEqual(pair.ann.config.num_attention_heads, 14)
        self.assertEqual(pair.ann.config.num_key_value_heads, 2)
        self.assertLess(metrics["max_abs"], 1e-6)
        self.assertGreater(metrics["cosine"], 0.999999)
        self.assertEqual(metrics["top1_agreement"], 1.0)
        self.assertEqual(DEFAULT_MODEL_ID, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(len(DEFAULT_REVISION), 40)


if __name__ == "__main__":
    unittest.main()
