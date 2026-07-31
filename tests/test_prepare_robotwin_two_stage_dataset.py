import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from script.prepare_robotwin_two_stage_dataset import (
    CAMERA_KEYS,
    DOMAIN_DIRS,
    audit_prepared_root,
    canonical_task_name,
    prepare_dataset,
    read_json,
    read_jsonl,
)


class PrepareRoboTwinTwoStageDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.output = self.root / "prepared"
        (self.source / "empty_emb.pt").parent.mkdir(parents=True)
        (self.source / "empty_emb.pt").write_bytes(b"empty-embedding")

        for domain, directory in DOMAIN_DIRS.items():
            episode_count = 5 if domain == "clean" else 7
            for task in ("task_a", "task_b"):
                self._create_task(
                    self.source / directory / task,
                    episode_count=episode_count,
                )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _create_task(repository: Path, episode_count: int):
        meta = repository / "meta"
        meta.mkdir(parents=True)
        info = {
            "chunks_size": 1000,
            "total_episodes": episode_count,
            "total_frames": episode_count * 4,
            "total_videos": episode_count * len(CAMERA_KEYS),
            "total_chunks": 1,
            "splits": {"train": f"0:{episode_count}"},
        }
        (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": repository.name}) + "\n",
            encoding="utf-8",
        )

        episodes = []
        for episode_index in range(episode_count):
            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [repository.name],
                    "length": 4,
                    "action_config": [{"start_frame": 0, "end_frame": 4}],
                }
            )
            episode_tag = f"episode_{episode_index:06d}"
            parquet = repository / "data" / "chunk-000" / f"{episode_tag}.parquet"
            parquet.parent.mkdir(parents=True, exist_ok=True)
            parquet.write_bytes(f"parquet-{episode_index}".encode())

            for camera in CAMERA_KEYS:
                video = (
                    repository
                    / "videos"
                    / "chunk-000"
                    / camera
                    / f"{episode_tag}.mp4"
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(f"video-{episode_index}-{camera}".encode())

                latent = (
                    repository
                    / "latents"
                    / "chunk-000"
                    / camera
                    / f"{episode_tag}_0_4.pth"
                )
                latent.parent.mkdir(parents=True, exist_ok=True)
                latent.write_bytes(f"latent-{episode_index}-{camera}".encode())

        (meta / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in episodes),
            encoding="utf-8",
        )

    def _arguments(self, *, link_mode="symlink", output=None):
        return SimpleNamespace(
            source_root=self.source,
            output_root=output or self.output,
            seed=42,
            expected_tasks=2,
            per_domain_total=4,
            stage1_per_domain=3,
            link_mode=link_mode,
            allow_missing_latent_segments=0,
        )

    def test_canonicalizes_published_repository_names(self):
        self.assertEqual(
            canonical_task_name(
                "clean", "adjust_bottle-demo_clean_collect_200-50"
            ),
            "adjust_bottle",
        )
        self.assertEqual(
            canonical_task_name(
                "clean", "put_bottles_dustbin-piper_clean_50-50"
            ),
            "put_bottles_dustbin",
        )
        self.assertEqual(
            canonical_task_name(
                "randomized", "adjust_bottle-aloha-agilex_randomized_500-1000"
            ),
            "adjust_bottle",
        )
        self.assertEqual(
            canonical_task_name("randomized", "scan_object"),
            "scan_object",
        )

    def test_prepares_disjoint_stage_views_and_refuses_overwrite(self):
        complete = prepare_dataset(self._arguments())
        self.assertEqual(complete["summary"]["stage1_episodes"], 12)
        self.assertEqual(complete["summary"]["stage2_episodes"], 4)

        manifest = read_json(self.output / "split_manifest.json")
        summary = audit_prepared_root(
            self.output,
            manifest,
            allow_missing_latent_segments=0,
            require_complete_marker=True,
        )
        self.assertEqual(summary["total_output_episodes"], 16)
        self.assertEqual(summary["stage1_segments"], 12)
        self.assertEqual(summary["stage1_valid_segments"], 12)
        self.assertEqual(summary["stage2_segments"], 4)
        self.assertEqual(summary["stage2_valid_segments"], 4)

        for task_row in manifest["tasks"]:
            for domain_row in task_row["domains"].values():
                stage1 = set(domain_row["stage1_source_episode_indices"])
                stage2 = set(domain_row["stage2_source_episode_indices"])
                self.assertEqual(len(stage1), 3)
                self.assertEqual(len(stage2), 1)
                self.assertFalse(stage1 & stage2)
                self.assertEqual(len(stage1 | stage2), 4)

                stage1_repo = self.output / domain_row["stage1_output_repo"]
                stage2_repo = self.output / domain_row["stage2_output_repo"]
                self.assertEqual(len(read_jsonl(stage1_repo / "meta/episodes.jsonl")), 3)
                self.assertEqual(len(read_jsonl(stage2_repo / "meta/episodes.jsonl")), 1)

        with self.assertRaises(FileExistsError):
            prepare_dataset(self._arguments())

    def test_hardlink_mode_reuses_source_payload_inode(self):
        output = self.root / "prepared-hardlink"
        prepare_dataset(self._arguments(link_mode="hardlink", output=output))

        manifest = read_json(output / "split_manifest.json")
        domain_row = manifest["tasks"][0]["domains"]["clean"]
        source_index = domain_row["stage1_source_episode_indices"][0]
        destination_index = int(
            domain_row["stage1_output"]["source_to_destination_index"][
                str(source_index)
            ]
        )
        source_repo = Path(domain_row["source_repo"])
        destination_repo = output / domain_row["stage1_output_repo"]
        source_parquet = (
            source_repo / "data/chunk-000" / f"episode_{source_index:06d}.parquet"
        )
        destination_parquet = (
            destination_repo
            / "data/chunk-000"
            / f"episode_{destination_index:06d}.parquet"
        )
        self.assertEqual(source_parquet.stat().st_ino, destination_parquet.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
