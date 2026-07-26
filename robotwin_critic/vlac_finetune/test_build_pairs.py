from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from robotwin_critic.vlac_finetune.build_pairs import build
from robotwin_critic.vlac_finetune.common import DEFAULT_CAMERAS, write_jsonl


class BuildPairsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.index = self.root / "index.jsonl"
        rows = []
        for task_index, task in enumerate(("task_a", "task_b")):
            task_dir = self.root / "dataset" / task
            chunk = "chunk-000"
            (task_dir / "data" / chunk).mkdir(parents=True)
            for episode in range(4):
                parquet = task_dir / "data" / chunk / f"episode_{episode:06d}.parquet"
                parquet.touch()
                for camera_index, camera in enumerate(DEFAULT_CAMERAS):
                    path = (
                        task_dir
                        / "videos"
                        / chunk
                        / camera
                        / f"episode_{episode:06d}.mp4"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.write_video(
                        path,
                        seed=1000 * task_index + 100 * episode + camera_index,
                    )
                rows.append(
                    {
                        "dataset_split": "synthetic",
                        "task_dir": str(task_dir),
                        "task": task,
                        "task_name": task,
                        "episode_index": episode,
                        "length": 16,
                        "text": task,
                        "parquet_path": str(parquet),
                    }
                )
        corrupt = (
            self.root
            / "dataset"
            / "task_a"
            / "videos"
            / "chunk-000"
            / DEFAULT_CAMERAS[0]
            / "episode_000000.mp4"
        )
        corrupt.write_bytes(b"not an mp4")
        write_jsonl(self.index, rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_video(path: Path, seed: int) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (48, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV cannot create the synthetic MP4")
        rng = np.random.default_rng(seed)
        for _ in range(16):
            frame = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_corrupt_episode_is_skipped_transactionally(self) -> None:
        output = self.root / "output"
        args = SimpleNamespace(
            index=str(self.index),
            output_dir=str(output),
            tasks=None,
            max_tasks=0,
            episodes_per_task=0,
            groups_per_episode=2,
            min_long_gap=4,
            adjacent_stride=1,
            eval_frames=4,
            val_ratio=0.25,
            pixel_static_threshold=0.0,
            image_width=64,
            jpeg_quality=90,
            trainer_val_samples=8,
            workers=2,
            seed=42,
            cameras=list(DEFAULT_CAMERAS),
        )
        counts = build(args)
        self.assertEqual(counts["episodes"], 7)
        self.assertEqual(counts["skipped_episodes"], 1)
        self.assertEqual(counts["train_samples"] + counts["val_samples"], 56)
        self.assertEqual(counts["trainer_val_samples"], 8)
        with (output / "val_train.jsonl").open() as handle:
            trainer_val = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(
            {row["metadata"]["task"] for row in trainer_val},
            {"task_a", "task_b"},
        )
        with (output / "build_summary.json").open() as handle:
            summary = json.load(handle)
        skipped = summary["skipped_episode_summary"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["task"], "task_a")
        self.assertEqual(skipped[0]["episode_index"], 0)
        self.assertIn(skipped[0]["error_type"], {"FileNotFoundError", "VideoDecodeError"})


if __name__ == "__main__":
    unittest.main()
