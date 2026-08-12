import unittest

from robotwin_critic.two_stage_rft.action_only_dataset import (
    DeterministicFractionDataset,
)


class DeterministicFractionDatasetTest(unittest.TestCase):
    def test_tenth_is_rounded_and_reproducible(self):
        source = list(range(101))
        first = DeterministicFractionDataset(source, fraction=0.1, seed=42)
        second = DeterministicFractionDataset(source, fraction=0.1, seed=42)
        self.assertEqual(len(first), 10)
        self.assertEqual(first.indices, second.indices)
        self.assertEqual([first[i] for i in range(len(first))], first.indices)

    def test_full_fraction_preserves_order(self):
        source = list(range(7))
        subset = DeterministicFractionDataset(source, fraction=1.0, seed=42)
        self.assertEqual(subset.indices, source)

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            DeterministicFractionDataset([1], fraction=0.0)


if __name__ == "__main__":
    unittest.main()
