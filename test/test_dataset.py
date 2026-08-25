import unittest

from ttfsformer.util.dataset import imagenet_loader


class DatasetShardValidationTest(unittest.TestCase):
    def test_num_shards_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "num_shards must be positive"):
            imagenet_loader("unused", "unused", 50, num_shards=0)

    def test_shard_index_must_be_in_range(self):
        with self.assertRaisesRegex(ValueError, "shard_index must be"):
            imagenet_loader("unused", "unused", 50, num_shards=2, shard_index=2)


if __name__ == "__main__":
    unittest.main()
