from __future__ import annotations

import unittest

import numpy as np

from robotwin_critic.two_stage_rft.evaluate_action_critic import binary_auc, corrupt
from robotwin_critic.two_stage_rft.kinematic_action_critic import (
    calibrate_kinematic_profile,
    quaternion_step_rotvecs,
    score_relative_actions,
    to_relative_actions,
)


def smooth_actions(frames: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames)
    actions = np.zeros((frames, 16), dtype=np.float32)
    actions[:, 0] = 0.4 + 0.05 * t
    actions[:, 1] = 0.1 + 0.01 * np.sin(np.pi * t)
    actions[:, 2] = 0.3
    actions[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0])
    actions[:, 8] = -0.4 - 0.04 * t
    actions[:, 9] = -0.1
    actions[:, 10] = 0.3 + 0.01 * np.sin(np.pi * t)
    actions[:, 11:15] = np.array([0.0, 0.0, 0.0, 1.0])
    actions[:, 7] = (t > 0.6).astype(np.float32)
    actions[:, 15] = (t > 0.4).astype(np.float32)
    return actions


def profile():
    training = []
    base = smooth_actions()
    for scale in np.linspace(0.95, 1.05, 12):
        sample = base.copy()
        sample[:, [0, 1, 8, 10]] *= scale
        training.append(sample)
    return calibrate_kinematic_profile(
        training,
        episode_ids=[f"stage1-{index}" for index in range(len(training))],
        fps=30.0,
        split_manifest_sha256="abc",
        soft_quantile=0.98,
        hard_quantile=0.995,
    )


class ActionCriticTest(unittest.TestCase):
    def test_quaternion_sign_is_invariant(self) -> None:
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
        quaternions[1] *= -1
        quaternions[3] *= -1
        np.testing.assert_allclose(quaternion_step_rotvecs(quaternions), 0.0)

    def test_spike_is_rejected_but_smooth_chunk_is_kept(self) -> None:
        critic = profile()
        smooth = score_relative_actions(
            to_relative_actions(smooth_actions()), critic
        )
        spiky_actions = to_relative_actions(smooth_actions())
        spiky_actions[48, 0] += 1.0
        spiky = score_relative_actions(spiky_actions, critic)
        self.assertTrue(smooth["accepted"])
        self.assertFalse(spiky["accepted"])
        self.assertGreater(smooth["action_score"], spiky["action_score"])

    def test_profile_records_stage1_provenance(self) -> None:
        critic = profile()
        self.assertEqual(critic.calibration_scope, "stage1_action_only")
        self.assertEqual(critic.split_manifest_sha256, "abc")
        self.assertTrue(all(value.startswith("stage1-") for value in critic.calibration_episode_ids))

    def test_corruption_benchmark_separates_obvious_outliers(self) -> None:
        critic = profile()
        relative = to_relative_actions(smooth_actions())
        positive = [score_relative_actions(relative, critic)["action_score"]]
        negative = [
            score_relative_actions(corrupt(relative, kind), critic)["action_score"]
            for kind in (
                "position_spike",
                "high_frequency_jitter",
                "orientation_spike",
                "excess_speed",
            )
        ]
        self.assertEqual(binary_auc(positive, negative), 1.0)


if __name__ == "__main__":
    unittest.main()
