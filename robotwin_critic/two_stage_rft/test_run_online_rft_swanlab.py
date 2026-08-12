from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from robotwin_critic.two_stage_rft.run_online_rft_swanlab import (
    acquire_parent_lock,
    build_runtime_config,
    log_startup_status,
    parse_metric_event,
    replay_completed_training_metrics,
)


class OnlineRFTSwanLabDriverTest(unittest.TestCase):
    def test_parent_lock_rejects_concurrent_driver(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = acquire_parent_lock(root)
            try:
                with self.assertRaises(RuntimeError):
                    acquire_parent_lock(root)
            finally:
                first.close()

    def test_runtime_config_contains_derived_training_shape(self) -> None:
        environment = {
            "INFER_GPU_IDS": "0,1",
            "REMOTE_INFER_WORKERS": "2",
            "REMOTE_GPU_IDS": "0,1",
            "Q_PER_ROUND": "320",
            "BUFFER_CAPACITY": "1024",
            "TRAIN_BATCH_SIZE_PER_GPU": "32",
            "TRAIN_GLOBAL_BATCH": "128",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "TRAIN_NNODES": "2",
            "TRAIN_LOCAL_NGPU": "2",
            "TRAIN_MASTER_ADDR": "10.0.0.1",
            "PSEUDO_EPOCHS_PER_UPDATE": "3",
            "REAL_FRACTION": "0.7",
            "ALLOW_MISSING_LATENT_SEGMENTS": "19",
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            config = build_runtime_config(Path("/tmp/online"), "run-1")
        self.assertEqual(config["sampling.local_num_gpus"], 2)
        self.assertEqual(config["sampling.num_gpus"], 4)
        self.assertEqual(config["sampling.q_per_gpu"], 80)
        self.assertEqual(config["training.world_size"], 4)
        self.assertEqual(config["training.gradient_accumulation_steps"], 1)
        self.assertEqual(config["training.effective_update_steps"], 80)
        self.assertEqual(config["replay.buffer_capacity"], 1024)
        self.assertEqual(config["data.allow_missing_latent_segments"], 19)

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

    def test_log_startup_status(self) -> None:
        class FakeSwanLab:
            def __init__(self) -> None:
                self.events = []

            def log(self, metrics, step=None) -> None:
                self.events.append((metrics, step))

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state.json").write_text(
                '{"update_index": 3, "collect_index": 5, "accepted_total": 70, "consumed_total": 64}',
                encoding="utf-8",
            )
            swanlab = FakeSwanLab()
            metrics = log_startup_status(
                swanlab, root, {"replay.buffer_capacity": 1024}
            )
            self.assertEqual(metrics["online/status/update_index"], 3.0)
            self.assertEqual(metrics["rft/update_round"], 3.0)
            self.assertEqual(metrics["online/status/collect_index"], 5.0)
            self.assertEqual(metrics["online/status/accepted_total"], 70.0)
            self.assertEqual(
                metrics["online/config/replay.buffer_capacity"], 1024.0
            )
            self.assertEqual(swanlab.events, [(metrics, None)])


if __name__ == "__main__":
    unittest.main()
