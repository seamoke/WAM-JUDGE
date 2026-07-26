import unittest

from .common import spearman_order


class SpearmanOrderTest(unittest.TestCase):
    def test_monotonic_orders(self):
        self.assertAlmostEqual(spearman_order([0.0, 1.0, 2.0]), 1.0)
        self.assertAlmostEqual(spearman_order([2.0, 1.0, 0.0]), -1.0)

    def test_constant_scores_are_neutral(self):
        self.assertEqual(spearman_order([0.0, 0.0, 0.0]), 0.0)

    def test_ties_use_average_ranks(self):
        self.assertAlmostEqual(
            spearman_order([0.0, 0.0, 1.0]),
            0.8660254037844387,
        )


if __name__ == "__main__":
    unittest.main()
