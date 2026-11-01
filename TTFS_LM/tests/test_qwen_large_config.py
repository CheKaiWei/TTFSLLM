import unittest

from ttfs_lm.dataset_sweep import load_dataset_sweep_config


class QwenLargeConfigTest(unittest.TestCase):
    def test_qwen_1_5b_sweep_is_pinned(self):
        config, digest = load_dataset_sweep_config("TTFS_LM/configs/dataset_sweep_qwen1.5b.yaml")
        self.assertEqual(config.model.id, "Qwen/Qwen2.5-1.5B")
        self.assertEqual(config.model.revision, "b4b7d0ff85668c78171cba0bd02714ee6ebf6e69")
        self.assertEqual(config.model.dtype, "bfloat16")
        self.assertEqual(config.runtime.batch_size, 1)
        self.assertEqual(config.runtime.calibration_batch_size, 4)
        self.assertEqual(config.runtime.evaluation_batch_size, 4)
        self.assertFalse(config.runtime.compute_kl)
        self.assertEqual(config.ranges_path, "TTFS_LM/artifacts/dataset_sweep_qwen1.5b/ranges.json")
        self.assertEqual(
            [dataset.name for dataset in config.evaluation_datasets],
            ["wikitext103_test", "penn_treebank_test", "lambada_openai_test", "c4_validation_262k"],
        )
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
