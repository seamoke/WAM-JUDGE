from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robotwin_critic.two_stage_rft.build_video_contexts import (
    coverage_frames,
    episode_transition_chunks,
)
from robotwin_critic.two_stage_rft.count_pseudo_budget import count_budget
from robotwin_critic.two_stage_rft.data_access import verified_eef_state_indices


class ChunkBudgetTest(unittest.TestCase):
    def test_budget_counts_executable_transitions_not_trajectory_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "split_manifest.json").write_text("{}\n", encoding="utf-8")
            ref = SimpleNamespace(
                repo=root / "repo",
                task="task",
                domain="clean",
                output_episode_index=0,
            )
            episode = {
                "episode_index": 0,
                "action_config": [
                    {"start_frame": 0, "end_frame": 100},
                    {"start_frame": 100, "end_frame": 200},
                ],
            }
            with (
                patch(
                    "robotwin_critic.two_stage_rft.count_pseudo_budget.iter_episode_refs",
                    return_value=[ref],
                ),
                patch(
                    "robotwin_critic.two_stage_rft.count_pseudo_budget.episode_metadata",
                    return_value={0: episode},
                ),
                patch(
                    "robotwin_critic.two_stage_rft.count_pseudo_budget.latent_segment_exists",
                    return_value=True,
                ),
                patch(
                    "robotwin_critic.two_stage_rft.count_pseudo_budget.latent_segment_num_frames",
                    side_effect=[9, 8],
                ),
            ):
                result = count_budget(root, root / "budget.json")
            self.assertEqual(result["segments_by_group"]["task/clean"], 2)
            self.assertEqual(result["groups"]["task/clean"], 15)

    def test_context_coverage_keeps_five_anchors_and_meets_chunk_count(self) -> None:
        frames = coverage_frames(143, (0.1, 0.3, 0.5, 0.7, 0.9), 15)
        self.assertEqual(len(frames), 15)
        self.assertEqual(len(set(frames)), 15)
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.assertIn(round(fraction * 142), frames)

    def test_context_chunk_count_uses_same_length_filter_as_budget(self) -> None:
        episode = {
            "episode_index": 0,
            "action_config": [
                {"start_frame": 0, "end_frame": 100},
                {"start_frame": 100, "end_frame": 800},
            ],
        }
        with (
            patch(
                "robotwin_critic.two_stage_rft.build_video_contexts.latent_segment_exists",
                return_value=True,
            ),
            patch(
                "robotwin_critic.two_stage_rft.build_video_contexts.latent_segment_num_frames",
                return_value=9,
            ),
        ):
            self.assertEqual(
                episode_transition_chunks(episode, Path("repo"), 500),
                8,
            )

    def test_eef_mapping_requires_exact_named_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "meta").mkdir()
            names = [
                "left_x", "left_y", "left_z", "left_q1", "left_q2",
                "left_q3", "left_q4", "left_gripper", "right_x",
                "right_y", "right_z", "right_q1", "right_q2",
                "right_q3", "right_q4", "right_gripper",
            ]
            (repo / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "features": {
                            "observation.state": {
                                "shape": [16],
                                "names": [names],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                verified_eef_state_indices(repo, "observation.state"),
                tuple(range(16)),
            )


if __name__ == "__main__":
    unittest.main()
