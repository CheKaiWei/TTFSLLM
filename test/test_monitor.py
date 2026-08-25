import unittest

import torch

from ttfsformer.util.monitor import MinMaxMonitor, RangeMonitor


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.batches = [
            torch.tensor([[-3.0, 1.5], [2.0, 4.0]]),
            torch.tensor([[-5.0, 0.0], [3.5, 8.0]]),
            torch.tensor([[-1.0, 2.5], [6.0, 7.0]]),
        ]

    def test_range_monitor_matches_python_reduction(self):
        monitor = RangeMonitor()
        for batch in self.batches:
            self.assertIs(monitor(batch), batch)

        expected = (
            min(batch.min().item() for batch in self.batches),
            max(batch.max().item() for batch in self.batches),
        )
        self.assertEqual(monitor.monitor_data(), expected)

    def test_minmax_monitor_matches_python_float_average(self):
        monitor = MinMaxMonitor()
        for batch in self.batches:
            self.assertIs(monitor(batch), batch)

        expected = (
            sum(batch.min().item() for batch in self.batches) / len(self.batches),
            sum(batch.max().item() for batch in self.batches) / len(self.batches),
        )
        self.assertEqual(monitor.monitor_data(), expected)

    def test_empty_monitor_has_no_data(self):
        self.assertIsNone(RangeMonitor().monitor_data())
        self.assertIsNone(MinMaxMonitor().monitor_data())


if __name__ == "__main__":
    unittest.main()
