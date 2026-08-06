from __future__ import annotations

import unittest

from robotwin_critic.two_stage_rft.run_online_rft_swanlab import (
    parse_metric_event,
)


class OnlineRFTSwanLabDriverTest(unittest.TestCase):
    def test_parse_training_metric_event(self) -> None:
        parsed = parse_metric_event(
            'SWANLAB_METRIC_EVENT {"step": 12, "metrics": {"loss": 1.5}}\n'
        )
        self.assertEqual(parsed, ({"loss": 1.5}, 12))

    def test_ignore_regular_output(self) -> None:
        self.assertIsNone(parse_metric_event("ordinary trainer output\n"))


if __name__ == "__main__":
    unittest.main()
