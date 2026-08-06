from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path


class PrepareActionVisibleRealTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            self.skipTest("pyarrow is required")

    @staticmethod
    def write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def make_source_repo(self, root: Path, domain_dir: str, task: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        repo = root / domain_dir / task
        episodes = []
        for index in (2, 7, 9):
            episodes.append(
                {
                    "episode_index": index,
                    "length": 1,
                    "action_config": [{"start_frame": 0, "end_frame": 1}],
                }
            )
            parquet = repo / "data/chunk-000" / f"episode_{index:06d}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [index],
                        "frame_index": [0],
                        "action": [[float(index), float(index + 1)]],
                    }
                ),
                parquet,
            )
            for camera in (
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            ):
                latent = (
                    repo
                    / "latents/chunk-000"
                    / camera
                    / f"episode_{index:06d}_0_1.pth"
                )
                latent.parent.mkdir(parents=True, exist_ok=True)
                latent.write_bytes(f"latent-{index}-{camera}".encode())

        self.write_jsonl(repo / "meta/episodes.jsonl", episodes)
        self.write_json(
            repo / "meta/info.json",
            {
                "chunks_size": 1000,
                "features": {"action": {"dtype": "float32", "shape": [2]}},
            },
        )

    def test_restores_exact_selected_union_with_action(self) -> None:
        import pyarrow.parquet as pq

        from robotwin_critic.two_stage_rft.prepare_action_visible_real import (
            COMPLETE_NAME,
            prepare,
        )
        from script.prepare_robotwin_two_stage_dataset import sha256_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            prepared = root / "prepared"
            output = prepared / "action_visible_real"
            source.mkdir()
            (prepared / "stage1").mkdir(parents=True)
            (prepared / "stage2").mkdir()
            (source / "empty_emb.pt").write_bytes(b"embedding")
            domains = {
                "clean": "lerobot_robotwin_eef_clean_50",
                "randomized": "lerobot_robotwin_eef_aug_500",
            }
            for directory_name in domains.values():
                self.make_source_repo(source, directory_name, "task_a")

            manifest = {
                "source_root": "/stale/path/on/another/machine",
                "domains": domains,
                "tasks": [
                    {
                        "task": "task_a",
                        "domains": {
                            domain: {
                                "selected_source_episode_indices_ranked": [7, 2],
                                "stage1_source_episode_indices": [2],
                                "stage2_source_episode_indices": [7],
                            }
                            for domain in domains
                        },
                    }
                ],
            }
            split_manifest = prepared / "split_manifest.json"
            self.write_json(split_manifest, manifest)
            self.write_json(
                prepared / "PREPARATION_COMPLETE.json",
                {"manifest_sha256": sha256_file(split_manifest)},
            )

            result = prepare(
                argparse.Namespace(
                    prepared_root=prepared,
                    source_root=source,
                    output_root=output,
                    link_mode="copy",
                    allow_missing_latent_segments=0,
                    verify_only=False,
                )
            )
            self.assertEqual(result["summary"]["episodes"], 4)
            self.assertTrue(result["summary"]["action_visible"])
            self.assertTrue((output / COMPLETE_NAME).is_file())

            for prefix in ("clean", "aug"):
                repo = output / f"lerobot_robotwin_eef_{prefix}_real_2/task_a"
                first = pq.read_table(repo / "data/chunk-000/episode_000000.parquet")
                second = pq.read_table(repo / "data/chunk-000/episode_000001.parquet")
                self.assertEqual(first["action"].to_pylist(), [[7.0, 8.0]])
                self.assertEqual(second["action"].to_pylist(), [[2.0, 3.0]])
                self.assertTrue(
                    (repo / "latents/chunk-000/observation.images.cam_high/episode_000000_0_1.pth").is_file()
                )


if __name__ == "__main__":
    unittest.main()
