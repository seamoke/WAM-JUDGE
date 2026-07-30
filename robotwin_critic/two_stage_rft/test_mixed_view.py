from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotwin_critic.two_stage_rft.build_mixed_view import build_view


def make_repo(root: Path, name: str, item_count: int) -> Path:
    repo = root / "split" / name
    (repo / "meta").mkdir(parents=True)
    (repo / "meta" / "info.json").write_text("{}")
    (repo / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "action_config": [
                    {"start_frame": index, "end_frame": index + 1}
                    for index in range(item_count)
                ],
            }
        )
        + "\n"
    )
    for payload in ("data", "latents"):
        (repo / payload).mkdir()
    for camera in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        camera_root = repo / "latents" / "chunk-000" / camera
        camera_root.mkdir(parents=True)
        for index in range(item_count):
            (camera_root / f"episode_000000_{index}_{index + 1}.pth").touch()
    return repo


class MixedViewTest(unittest.TestCase):
    def test_builds_weighted_view_without_copying_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared"
            stage1 = prepared / "stage1"
            rft = root / "rft"
            prepared.mkdir()
            rft.mkdir()
            (prepared / "split_manifest.json").write_text("{}")
            (rft / "rft_manifest.json").write_text("{}")
            (stage1).mkdir()
            (stage1 / "empty_emb.pt").write_bytes(b"embedding")
            stage1_repo = make_repo(stage1, "task-a", 3)
            rft_repo = make_repo(rft, "task-a", 1)
            output = root / "mixed"
            manifest = build_view(
                stage1,
                rft,
                output,
                rft_target_fraction=0.25,
            )
            self.assertEqual(manifest["stage1_items"], 3)
            self.assertEqual(manifest["rft_effective_items"], 1)
            self.assertAlmostEqual(manifest["effective_rft_fraction"], 0.25)
            linked_data = next((output / "stage1").glob("**/data"))
            self.assertTrue(linked_data.is_symlink())
            self.assertEqual(linked_data.resolve(), (stage1_repo / "data").resolve())
            linked_rft = next((output / "rft").glob("**/latents"))
            self.assertEqual(linked_rft.resolve(), (rft_repo / "latents").resolve())


if __name__ == "__main__":
    unittest.main()
