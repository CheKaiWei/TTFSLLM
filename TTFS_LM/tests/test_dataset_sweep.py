import tempfile
import unittest
from pathlib import Path

import yaml

from ttfs_lm.dataset_sweep import load_dataset_sweep_config


class DatasetSweepConfigTest(unittest.TestCase):
    def test_repository_sweep_config_is_pinned(self):
        config, digest = load_dataset_sweep_config("TTFS_LM/configs/dataset_sweep.yaml")
        self.assertEqual(config.calibration_dataset.name, "wikitext2_train_calibration")
        self.assertEqual(len(config.evaluation_datasets), 4)
        self.assertEqual(
            [source.name for source in config.evaluation_datasets],
            [
                "wikitext103_test",
                "penn_treebank_test",
                "lambada_openai_test",
                "c4_validation_262k",
            ],
        )
        self.assertTrue(all(len(source.revision) == 40 for source in config.evaluation_datasets))
        self.assertEqual(config.evaluation_datasets[-1].max_tokens, 262420)
        self.assertEqual(len(digest), 64)

    def test_duplicate_dataset_names_are_rejected(self):
        source = yaml.safe_load(Path("TTFS_LM/configs/dataset_sweep.yaml").read_text())
        source["evaluation_datasets"][1]["name"] = source["evaluation_datasets"][0]["name"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "names must be unique"):
                load_dataset_sweep_config(path)


if __name__ == "__main__":
    unittest.main()
