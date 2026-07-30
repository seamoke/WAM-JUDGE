from __future__ import annotations

import unittest

import numpy as np

from robotwin_critic.two_stage_rft.action_critic import (
    calibrate,
    quaternion_step_angles,
    score_actions,
)
from robotwin_critic.two_stage_rft.evaluate_action_critic import binary_auc, corrupt


def smooth_actions(frames: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 1.0, frames)
    actions = np.zeros((frames, 16), dtype=np.float32)
    actions[:, 0] = 0.05 * t
    actions[:, 1] = 0.01 * np.sin(np.pi * t)
    actions[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0])
    actions[:, 8] = -0.04 * t
    actions[:, 10] = 0.01 * np.sin(np.pi * t)
    actions[:, 11:15] = np.array([0.0, 0.0, 0.0, 1.0])
    actions[:, 7] = (t > 0.6).astype(np.float32)
    actions[:, 15] = (t > 0.4).astype(np.float32)
    return actions


class ActionCriticTest(unittest.TestCase):
    def test_quaternion_sign_is_invariant(self) -> None:
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
        quaternions[1] *= -1
        quaternions[3] *= -1
        np.testing.assert_allclose(quaternion_step_angles(quaternions), 0.0)

    def test_spike_is_rejected_but_smooth_chunk_is_kept(self) -> None:
        training = []
        base = smooth_actions()
        for scale in np.linspace(0.95, 1.05, 12):
            sample = base.copy()
            sample[:, [0, 1, 8, 10]] *= scale
            training.append(sample)
        profile = calibrate(training, fps=30.0, soft_quantile=0.98, hard_quantile=0.995)
        smooth = score_actions(smooth_actions(), profile)
        spiky_actions = smooth_actions()
        spiky_actions[48, 0] += 1.0
        spiky = score_actions(spiky_actions, profile)
        self.assertTrue(smooth["accepted"])
        self.assertFalse(spiky["accepted"])
        self.assertGreater(smooth["action_score"], spiky["action_score"])

    def test_gripper_switch_is_not_a_kinematic_rejection(self) -> None:
        profile = calibrate([smooth_actions() for _ in range(8)], fps=30.0)
        actions = smooth_actions()
        actions[:, 7] = np.arange(len(actions)) % 2
        result = score_actions(actions, profile)
        self.assertTrue(result["accepted"])
        self.assertGreater(result["gripper_switch_rate"], 0.45)

    def test_corruption_benchmark_separates_obvious_outliers(self) -> None:
        profile = calibrate([smooth_actions() for _ in range(8)], fps=30.0)
        positive = [score_actions(smooth_actions(), profile)["action_score"]]
        negative = [
            score_actions(corrupt(smooth_actions(), kind), profile)["action_score"]
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
