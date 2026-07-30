from __future__ import annotations

import unittest

from robotwin_critic.two_stage_rft.select_rft_candidates import sigmoid


class CandidateSelectionTest(unittest.TestCase):
    def test_sigmoid_is_stable_for_large_process_deltas(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(1000.0), 0.999)
        self.assertLess(sigmoid(-1000.0), 0.001)


if __name__ == "__main__":
    unittest.main()
