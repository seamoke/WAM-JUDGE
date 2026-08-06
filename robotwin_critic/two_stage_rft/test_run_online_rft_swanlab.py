from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robotwin_critic.two_stage_rft.run_online_rft_swanlab import (
    parse_metric_event,
    replay_completed_training_metrics,
)


class OnlineRFTSwanLabDriverTest(unittest.TestCase):
    def test_parse_training_metric_event(self) -> None:
        parsed = parse_metric_event(
            'SWANLAB_METRIC_EVENT {"step": 12, "metrics": {"loss": 1.5}}\n'
        )
        self.assertEqual(parsed, ({"loss": 1.5}, 12))

    def test_parse_torchrun_prefixed_training_metric_event(self) -> None:
        parsed = parse_metric_event(
            '[default0]:SWANLAB_METRIC_EVENT {"step": 7, "metrics": {"train/loss": 0.25}}\n'
        )
        self.assertEqual(parsed, ({"train/loss": 0.25}, 7))

    def test_ignore_regular_output(self) -> None:
        self.assertIsNone(parse_metric_event("ordinary trainer output\n"))

    def test_replay_completed_training_metrics_once(self) -> None:
        class FakeSwanLab:
            def __init__(self) -> None:
                self.events = []

            def log(self, metrics, step=None) -> None:
                self.events.append((metrics, step))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            update = root / "updates" / "update_000000"
            (update / "model").mkdir(parents=True)
            (update / "train").mkdir()
            (update / "train" / "train.log").write_text(
                '[default0]:SWANLAB_METRIC_EVENT {"step": 2, "metrics": {"train/loss": 0.5}}\n',
                encoding="utf-8",
            )
            state_path = root / "metric_state.json"
            swanlab = FakeSwanLab()
            self.assertEqual(
                replay_completed_training_metrics(swanlab, root, state_path), [0]
            )
            self.assertEqual(swanlab.events, [({"train/loss": 0.5}, 2)])
            self.assertEqual(
                replay_completed_training_metrics(swanlab, root, state_path), []
            )


if __name__ == "__main__":
    unittest.main()
