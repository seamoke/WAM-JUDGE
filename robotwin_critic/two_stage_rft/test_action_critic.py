from __future__ import annotations

import unittest

import numpy as np

from robotwin_critic.two_stage_rft.evaluate_action_critic import binary_auc, corrupt
from robotwin_critic.two_stage_rft.kinematic_action_critic import (
    action_gate_violations,
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
    def test_score_with_safety_gates_keeps_kinematics_soft(self) -> None:
        violations = [
            "left.linear_acceleration",
            "left.linear_jerk",
            "right.eef_workspace",
            "left.gripper_limit",
        ]
        self.assertEqual(
            action_gate_violations(violations, "score_with_safety_gates"),
            ["left.gripper_limit", "right.eef_workspace"],
        )
        self.assertEqual(action_gate_violations(violations, "strict"), sorted(violations))

    def test_unknown_workspace_scope_is_rejected(self) -> None:
        critic = profile()
        relative = to_relative_actions(smooth_actions()[:16])
        with self.assertRaisesRegex(ValueError, "workspace_scope"):
            score_relative_actions(relative, critic, workspace_scope="unknown")

    def test_quaternion_sign_is_invariant(self) -> None:
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
        quaternions[1] *= -1
        quaternions[3] *= -1
        np.testing.assert_allclose(quaternion_step_rotvecs(quaternions), 0.0)

    def test_spike_is_rejected_but_smooth_chunk_is_kept(self) -> None:
        critic = profile()
        calibrated_smooth = smooth_actions()
        calibrated_smooth[:, [0, 1, 8, 10]] *= 0.95
        smooth = score_relative_actions(
            to_relative_actions(calibrated_smooth[:16]), critic
        )
        spiky_actions = to_relative_actions(calibrated_smooth[:16])
        spiky_actions[8, 0] += 1.0
        spiky = score_relative_actions(spiky_actions, critic)
        self.assertTrue(smooth["accepted"])
        self.assertFalse(spiky["accepted"])
        self.assertGreater(smooth["action_score"], spiky["action_score"])

    def test_executable_chunk_without_condition_block_is_checked_from_reference(self) -> None:
        critic = profile()
        executable = to_relative_actions(smooth_actions()[1:17])
        kept = score_relative_actions(executable, critic)
        jumped = executable.copy()
        jumped[0, 0] += 0.25
        rejected = score_relative_actions(jumped, critic)
        self.assertTrue(kept["condition_reference_checked"])
        self.assertTrue(kept["accepted"])
        self.assertFalse(rejected["accepted"])

    def test_gripper_jump_is_measured_against_current_state(self) -> None:
        critic = profile()
        executable = to_relative_actions(smooth_actions()[70:86])
        executable[:, 7] = 0.0
        executable[:, 15] = 0.0
        start_state = smooth_actions()[70]
        without_state = score_relative_actions(executable, critic)
        with_state = score_relative_actions(
            executable,
            critic,
            start_state=start_state,
        )
        self.assertEqual(
            without_state["diagnostics"]["left.gripper_velocity"]["maximum"],
            0.0,
        )
        self.assertGreater(
            with_state["diagnostics"]["left.gripper_velocity"]["maximum"],
            0.0,
        )

    def test_profile_records_stage1_provenance(self) -> None:
        critic = profile()
        self.assertEqual(critic.calibration_scope, "stage1_action_only")
        self.assertEqual(critic.split_manifest_sha256, "abc")
        self.assertTrue(all(value.startswith("stage1-") for value in critic.calibration_episode_ids))

    def test_corruption_benchmark_separates_obvious_outliers(self) -> None:
        critic = profile()
        relative = to_relative_actions(smooth_actions()[:16])
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
