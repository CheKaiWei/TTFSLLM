import os
import tempfile
import unittest

from ttfs_lm.mvp import run_mvp


@unittest.skipUnless(os.environ.get("TTFS_LM_RUN_MVP") == "1", "set TTFS_LM_RUN_MVP=1")
class MVPIntegrationTest(unittest.TestCase):
    def test_official_qwen_wikitext2_mvp(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_mvp(
                "TTFS_LM/configs/mvp_wikitext2.yaml",
                output_dir=output_dir,
                local_files_only=True,
            )
            self.assertTrue(result["metrics"]["passed"], result["metrics"]["acceptance"])


if __name__ == "__main__":
    unittest.main()
