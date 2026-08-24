import unittest

import numpy as np

from ieee33_diffusion.train_infer import _conditioning_topology_ids


class SamplingConditionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.available = np.asarray([3, 5, 8, 13], dtype=np.int64)
        self.targets = np.asarray([3, 3, 5, 8, 13, 13], dtype=np.int64)

    def test_correct_condition_preserves_targets(self) -> None:
        result = _conditioning_topology_ids(
            self.targets, self.available, "correct", seed=2026
        )
        np.testing.assert_array_equal(result, self.targets)

    def test_base_condition_uses_topology_zero(self) -> None:
        result = _conditioning_topology_ids(
            self.targets, self.available, "base", seed=2026
        )
        np.testing.assert_array_equal(result, np.zeros_like(self.targets))

    def test_shuffled_condition_is_consistent_and_deranged(self) -> None:
        result = _conditioning_topology_ids(
            self.targets, self.available, "shuffled", seed=2026
        )
        self.assertTrue(np.all(result != self.targets))
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[-1], result[-2])
        self.assertTrue(set(result).issubset(set(self.available)))


if __name__ == "__main__":
    unittest.main()
