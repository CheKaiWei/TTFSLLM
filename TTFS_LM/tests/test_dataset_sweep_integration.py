import os
import tempfile
import unittest

from ttfs_lm.dataset_sweep import run_dataset_sweep


@unittest.skipUnless(
    os.environ.get("TTFS_LM_RUN_DATASET_SWEEP") == "1",
    "set TTFS_LM_RUN_DATASET_SWEEP=1",
)
class DatasetSweepIntegrationTest(unittest.TestCase):
    def test_official_qwen_dataset_sweep_completes(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_dataset_sweep(
                "TTFS_LM/configs/dataset_sweep.yaml",
                output_dir=output_dir,
            )
            self.assertTrue(result["metrics"]["summary"]["execution_completed"])
            self.assertEqual(result["metrics"]["summary"]["dataset_count"], 4)


if __name__ == "__main__":
    unittest.main()
