from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

from robotwin_critic.two_stage_rft.log_online_collection_swanlab import (
    OnlineCollectionLogger,
    build_collect_metrics,
    generated_visual_metrics,
    numeric_metrics,
)


class CollectionSwanLabMetricsTest(unittest.TestCase):
    def test_numeric_metrics_contains_distribution_quantiles(self) -> None:
        metrics = numeric_metrics("score", [0.0, 10.0, 20.0])
        self.assertEqual(metrics["score/mean"], 10.0)
        self.assertEqual(metrics["score/median"], 10.0)
        self.assertEqual(metrics["score/max"], 20.0)

    def test_collect_metrics_contains_task_and_critic_records(self) -> None:
        generated = [
            {
                "task": "pick",
                "domain": "clean",
                "context_id": "q0",
                "candidate_id": "q0/0",
                "process_score": 10.0,
                "process_critic": {"numeric_parsed": True},
                "action_critic": {
                    "accepted": True,
                    "action_score": 0.8,
                    "gate_violations": [],
                    "hard_violations": ["left.jerk"],
                },
            },
            {
                "task": "pick",
                "domain": "clean",
                "context_id": "q0",
                "candidate_id": "q0/1",
                "process_score": -10.0,
                "process_critic": {"numeric_parsed": True},
                "action_critic": {
                    "accepted": False,
                    "action_score": 0.2,
                    "gate_violations": ["workspace"],
                    "hard_violations": [],
                },
            },
        ]
        metrics = build_collect_metrics(
            generated,
            [generated[0]],
            {
                "pending_after_commit": 1,
                "buffer_capacity": 512,
                "action_rejected": 1,
                "process_rejected": 0,
            },
            generated,
            [generated[0]],
        )
        self.assertEqual(metrics["collect/generated_qa_pairs"], 2.0)
        self.assertEqual(metrics["collect/retained_qa_pairs"], 1.0)
        self.assertEqual(metrics["critic/process_score/mean"], 0.0)
        self.assertEqual(metrics["critic/retained_process_score/mean"], 10.0)
        self.assertEqual(metrics["collect/retained_qa_per_retained_q"], 1.0)
        self.assertEqual(metrics["task/pick/qa_retention_rate"], 0.5)
        self.assertEqual(metrics["action_violation/hard.left.jerk"], 1.0)

    @unittest.skipIf(
        Image is None, "Pillow is not installed in the test interpreter"
    )
    def test_generated_visual_metrics_detects_black_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            black = root / "black.png"
            white = root / "white.png"
            Image.new("RGB", (32, 32), "black").save(black)
            Image.new("RGB", (32, 32), "white").save(white)
            metrics = generated_visual_metrics(
                [{"generated_image": str(black)}, {"generated_image": str(white)}]
            )
            self.assertEqual(metrics["visual/generated_images_checked"], 2.0)
            self.assertEqual(metrics["visual/generated_near_black_rate"], 0.5)
            self.assertAlmostEqual(
                metrics["visual/generated_black_fraction/mean"], 0.5
            )

    def test_continuous_logger_uses_existing_run_without_finishing(self) -> None:
        class FakeSwanLab:
            def __init__(self) -> None:
                self.records = []

            def log(self, metrics, step=None) -> None:
                self.records.append((metrics, step))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect = root / "collect" / "collect_000000"
            collect.mkdir(parents=True)
            generated = {
                "task": "pick",
                "domain": "clean",
                "context_id": "q0",
                "candidate_id": "q0/0",
                "process_score": 10.0,
                "process_critic": {"numeric_parsed": True},
                "action_critic": {
                    "accepted": True,
                    "action_score": 0.8,
                    "gate_violations": [],
                    "hard_violations": [],
                },
            }
            (collect / "dual_scored.jsonl").write_text(
                json.dumps(generated) + "\n", encoding="utf-8"
            )
            (collect / "selected_winners.jsonl").write_text(
                json.dumps(generated) + "\n", encoding="utf-8"
            )
            (collect / "selection_summary.json").write_text(
                json.dumps({"buffer_capacity": 512, "pending_after_commit": 1}),
                encoding="utf-8",
            )
            fake = FakeSwanLab()
            logger = OnlineCollectionLogger(
                fake,
                root,
                root / "upload_state.json",
                "run-1",
                max_images_per_collect=0,
            )
            self.assertEqual(logger.log_completed(), 1)
            self.assertEqual(logger.log_completed(), 0)
            self.assertEqual(len(fake.records), 1)
            self.assertEqual(fake.records[0][1], 0)


if __name__ == "__main__":
    unittest.main()
