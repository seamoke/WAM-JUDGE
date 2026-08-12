import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from robotwin_critic.two_stage_rft.finalize_one_shot_buffer import (
    finalize_buffer,
    proportional_quotas,
)


class OneShotBufferTest(unittest.TestCase):
    def test_proportional_quotas_preserve_target(self):
        self.assertEqual(
            proportional_quotas({"a": 1, "b": 2, "c": 1}, 10),
            {"a": 3, "b": 5, "c": 2},
        )

    def test_balances_groups_progress_and_rejects_black_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.jpg"
            black = root / "black.jpg"
            image = Image.new("L", (16, 16), color=80)
            for x in range(8, 16):
                for y in range(16):
                    image.putpixel((x, y), 140)
            image.save(valid)
            Image.new("L", (16, 16), color=0).save(black)
            rows = []
            candidate = 0
            for group, weight in (("task_a/clean", 1), ("task_b/randomized", 1)):
                task, domain = group.split("/")
                for episode in range(2):
                    for progress in (0.1, 0.8):
                        candidate += 1
                        diagnostics = {
                            "velocity": {
                                "maximum": candidate,
                                "hard": 100.0,
                            }
                        }
                        rows.append(
                            {
                                "candidate_id": f"c{candidate}",
                                "context_id": f"{group}/{episode}/{progress}@e0",
                                "source_context_id": f"{group}/{episode}/{progress}",
                                "task": task,
                                "domain": domain,
                                "source_episode_index": episode,
                                "progress_fraction": progress,
                                "generated_image": str(valid),
                                "process_score": float(candidate),
                                "action_critic": {
                                    "action_score": 0.9,
                                    "diagnostics": diagnostics,
                                },
                                "rft_selection": {"mode": "dual"},
                            }
                        )
            rejected = dict(rows[0])
            rejected["candidate_id"] = "black"
            rejected["generated_image"] = str(black)
            rows.append(rejected)
            selected, summary = finalize_buffer(
                rows,
                {"task_a/clean": 1, "task_b/randomized": 1},
                target=4,
                visual_cache_path=root / "cache.jsonl",
                max_per_context=1,
                max_per_episode=2,
                min_action_distance=0.0,
            )
            self.assertTrue(summary["ready"])
            self.assertEqual(len(selected), 4)
            self.assertEqual(summary["visual_rejected"], 1)
            self.assertEqual(summary["selected_by_group"], {
                "task_a/clean": 2,
                "task_b/randomized": 2,
            })
            self.assertEqual(set(summary["selected_by_progress_bin"]), {"0", "4"})
            self.assertTrue(all(row["rft_selection"]["one_shot"] for row in selected))

    def test_backfills_an_unreachable_group_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.jpg"
            image = Image.new("L", (16, 16), color=80)
            for x in range(8, 16):
                for y in range(16):
                    image.putpixel((x, y), 140)
            image.save(valid)
            rows = []
            for group, count in (("task_a/clean", 1), ("task_b/clean", 5)):
                task, domain = group.split("/")
                for index in range(count):
                    rows.append(
                        {
                            "candidate_id": f"{task}-{index}",
                            "context_id": f"{group}/{index}",
                            "task": task,
                            "domain": domain,
                            "source_episode_index": index,
                            "progress_fraction": index / max(count, 1),
                            "generated_image": str(valid),
                            "process_score": float(index),
                            "action_critic": {"action_score": 0.9},
                            "rft_selection": {"mode": "dual"},
                        }
                    )
            selected, summary = finalize_buffer(
                rows,
                {"task_a/clean": 1, "task_b/clean": 1},
                target=4,
                visual_cache_path=root / "cache.jsonl",
                max_per_context=1,
                max_per_episode=1,
                min_action_distance=0.0,
            )
            self.assertTrue(summary["ready"])
            self.assertEqual(len(selected), 4)
            self.assertEqual(summary["selected_by_group"]["task_a/clean"], 1)
            self.assertEqual(summary["selected_by_group"]["task_b/clean"], 3)
            self.assertEqual(
                summary["quota_shortfalls_before_backfill"]["task_a/clean"]["missing"],
                1,
            )
            self.assertEqual(summary["quota_overfill_after_backfill"], {"task_b/clean": 1})


if __name__ == "__main__":
    unittest.main()
