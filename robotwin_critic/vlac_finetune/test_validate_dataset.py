from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from robotwin_critic.vlac_finetune.validate_dataset import validate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def pair_row(
    first: Path,
    second: Path,
    *,
    episode: int,
    i: int,
    j: int,
    target: float,
    task_dir: str,
) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "compare"},
            {"role": "assistant", "content": f"{target:+.1f}"},
        ],
        "images": [str(first), str(second)],
        "metadata": {
            "task": "synthetic",
            "dataset_split": "clean",
            "task_dir": task_dir,
            "episode_index": episode,
            "pair_kind": "long",
            "i": i,
            "j": j,
            "target": target,
            "goal_image": str(second),
        },
    }


class ValidateDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        self.images = []
        for index in range(4):
            path = self.root / f"state_{index}.jpg"
            image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            self.images.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifests(self) -> tuple[list[dict], list[dict]]:
        train_forward = pair_row(
            self.images[0],
            self.images[1],
            episode=1,
            i=2,
            j=10,
            target=25.0,
            task_dir="/dataset/train_task",
        )
        train_reverse = pair_row(
            self.images[1],
            self.images[0],
            episode=1,
            i=10,
            j=2,
            target=-25.0,
            task_dir="/dataset/train_task",
        )
        val_forward = pair_row(
            self.images[2],
            self.images[3],
            episode=2,
            i=3,
            j=11,
            target=30.0,
            task_dir="/dataset/val_task",
        )
        val_reverse = pair_row(
            self.images[3],
            self.images[2],
            episode=2,
            i=11,
            j=3,
            target=-30.0,
            task_dir="/dataset/val_task",
        )
        return [train_forward, train_reverse], [val_forward, val_reverse]

    def write_dataset(self, train: list[dict], val: list[dict]) -> None:
        write_jsonl(self.root / "train.jsonl", train)
        write_jsonl(self.root / "val.jsonl", val)
        write_jsonl(
            self.root / "val_trajectories.jsonl",
            [
                {
                    "task": "synthetic",
                    "episode_index": 2,
                    "images": [str(self.images[2]), str(self.images[3])],
                }
            ],
        )
        state_train = [
            {
                "state_i": str(self.images[0]),
                "state_j": str(self.images[1]),
                "goal": str(self.images[1]),
                "target_delta": 25.0,
                "label": 1,
            }
        ]
        state_val = [
            {
                "state_i": str(self.images[3]),
                "state_j": str(self.images[2]),
                "goal": str(self.images[3]),
                "target_delta": -30.0,
                "label": -1,
            }
        ]
        write_jsonl(self.root / "state_score_train.jsonl", state_train)
        write_jsonl(self.root / "state_score_val.jsonl", state_val)

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            data_dir=str(self.root),
            output=str(self.root / "validation.jpg"),
            samples=4,
        )

    def test_valid_manifests_pass(self) -> None:
        train, val = self.manifests()
        self.write_dataset(train, val)
        summary = validate(self.args())
        self.assertEqual(summary["episode_overlap"], 0)
        self.assertTrue(summary["all_pairs_antisymmetric"])
        self.assertEqual(summary["train_pair_groups"], 1)
        self.assertEqual(summary["val_pair_groups"], 1)

    def test_episode_leakage_is_rejected(self) -> None:
        train, val = self.manifests()
        for row in val:
            row["metadata"]["episode_index"] = 1
            row["metadata"]["task_dir"] = "/dataset/train_task"
        self.write_dataset(train, val)
        with self.assertRaisesRegex(RuntimeError, "Episode-level leakage"):
            validate(self.args())

    def test_bad_reverse_pair_is_rejected(self) -> None:
        train, val = self.manifests()
        train[1]["metadata"]["i"] = 9
        self.write_dataset(train, val)
        with self.assertRaisesRegex(RuntimeError, "frame indices are not reversed"):
            validate(self.args())


if __name__ == "__main__":
    unittest.main()
