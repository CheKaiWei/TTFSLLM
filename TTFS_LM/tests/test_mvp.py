import tempfile
import unittest
from pathlib import Path

import yaml

from ttfs_lm.mvp import load_mvp_config


class MVPConfigTest(unittest.TestCase):
    def test_repository_mvp_config_is_valid_and_pinned(self):
        config, digest = load_mvp_config("TTFS_LM/configs/mvp_wikitext2.yaml")
        self.assertEqual(config.model.id, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(config.dataset.id, "Salesforce/wikitext")
        self.assertEqual(config.dataset.config, "wikitext-2-raw-v1")
        self.assertEqual(config.dataset.calibration_split, "train")
        self.assertEqual(config.dataset.evaluation_split, "validation")
        self.assertEqual(config.dataset.context_length, 128)
        self.assertEqual(config.acceptance.cache_max_abs, 2e-4)
        self.assertEqual(len(digest), 64)

    def test_same_split_is_rejected(self):
        source = yaml.safe_load(Path("TTFS_LM/configs/mvp_wikitext2.yaml").read_text())
        source["dataset"]["evaluation_split"] = source["dataset"]["calibration_split"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "splits must differ"):
                load_mvp_config(path)

    def test_invalid_acceptance_threshold_is_rejected(self):
        source = yaml.safe_load(Path("TTFS_LM/configs/mvp_wikitext2.yaml").read_text())
        source["acceptance"]["cache_max_abs"] = -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "error tolerances"):
                load_mvp_config(path)


if __name__ == "__main__":
    unittest.main()
