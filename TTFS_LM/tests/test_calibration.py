import tempfile
import unittest
from pathlib import Path

import torch

from ttfs_lm.calibration import MinMaxCalibrator, RangeRegistry, RangeSpec


class CalibrationTest(unittest.TestCase):
    def test_masked_values_do_not_change_range(self):
        calibrator = MinMaxCalibrator()
        value = torch.tensor([[[1.0, 2.0], [1000.0, -1000.0]]])
        mask = torch.tensor([[[True], [False]]])
        calibrator.observe("hidden", value, mask)
        spec = calibrator.finalize(margin=0.0)["hidden"]
        self.assertEqual(spec.lower, 1.0)
        self.assertEqual(spec.upper, 2.0)
        self.assertEqual(spec.elements, 2)

    def test_probability_ranges_are_fixed_to_unit_interval(self):
        calibrator = MinMaxCalibrator()
        calibrator.observe("layer.attn.probs", torch.tensor([0.2, 0.8]))
        spec = calibrator.finalize()["layer.attn.probs"]
        self.assertEqual((spec.lower, spec.upper), (0.0, 1.0))

    def test_nonnegative_operator_ranges_do_not_cross_zero(self):
        calibrator = MinMaxCalibrator()
        calibrator.observe("layer.norm.variance", torch.tensor([0.1, 1.0]))
        spec = calibrator.finalize(margin=1.0)["layer.norm.variance"]
        self.assertEqual(spec.lower, 0.0)
        self.assertGreater(spec.upper, 1.0)

    def test_registry_json_round_trip(self):
        registry = RangeRegistry(
            {"x": RangeSpec("x", -1.0, 2.0, calls=3, elements=12)},
            {"model_id": "unit-test"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranges.json"
            registry.save(path)
            loaded = RangeRegistry.load(path)
        self.assertEqual(loaded["x"], registry["x"])
        self.assertEqual(loaded.metadata, registry.metadata)


if __name__ == "__main__":
    unittest.main()
