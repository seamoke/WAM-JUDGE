"""Stage-1-only kinematic critic for RoboTwin dual-arm action chunks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ARM_LAYOUT = {
    "left": (slice(0, 3), slice(3, 7), 7),
    "right": (slice(8, 11), slice(11, 15), 15),
}
SERIES_NAMES = (
    "linear_velocity",
    "angular_velocity",
    "linear_displacement",
    "angular_displacement",
    "linear_acceleration",
    "angular_acceleration",
    "linear_jerk",
    "angular_jerk",
    "gripper_velocity",
    "gripper_acceleration",
    "gripper_jerk",
)
GATE_POLICIES = ("strict", "score_with_safety_gates")
WORKSPACE_SCOPES = ("task", "global")


def action_gate_violations(
    violations: Iterable[str], gate_policy: str
) -> list[str]:
    """Select which diagnostics are hard gates under the requested policy."""
    violations = sorted(set(str(value) for value in violations))
    if gate_policy == "strict":
        return violations
    if gate_policy == "score_with_safety_gates":
        return [
            value
            for value in violations
            if value.endswith(".eef_workspace")
            or value.endswith(".gripper_limit")
        ]
    raise ValueError(
        f"Unknown gate_policy={gate_policy!r}; expected one of {GATE_POLICIES}"
    )


def _actions(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 16:
        raise ValueError(f"Expected [T,16] RoboTwin actions, got {value.shape}")
    if len(value) < 4:
        raise ValueError("A kinematic chunk needs at least four action steps")
    if not np.all(np.isfinite(value)):
        raise ValueError("Action chunk contains NaN or infinity")
    return value


def _normalized_quaternions(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Action chunk contains a zero quaternion")
    return value / norms


def _quaternion_conjugate(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result[..., :3] *= -1.0
    return result


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + np.cross(left_xyz, right_xyz)
    )
    w = left_w * right_w - np.sum(left_xyz * right_xyz, axis=-1, keepdims=True)
    return np.concatenate([xyz, w], axis=-1)


def quaternion_step_rotvecs(value: np.ndarray) -> np.ndarray:
    quaternions = _normalized_quaternions(np.asarray(value, dtype=np.float64))
    current = quaternions[1:].copy()
    previous = quaternions[:-1]
    current[np.sum(current * previous, axis=-1) < 0.0] *= -1.0
    relative = _normalized_quaternions(
        _quaternion_multiply(_quaternion_conjugate(previous), current)
    )
    relative[relative[:, 3] < 0.0] *= -1.0
    vector = relative[:, :3]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, relative[:, 3:4])
    axis = np.divide(
        vector,
        vector_norm,
        out=np.zeros_like(vector),
        where=vector_norm > 1e-12,
    )
    return axis * angle


def quaternion_reference_rotvecs(value: np.ndarray) -> np.ndarray:
    quaternions = _normalized_quaternions(np.asarray(value, dtype=np.float64))
    reference = np.repeat(quaternions[:1], len(quaternions), axis=0)
    current = quaternions.copy()
    current[np.sum(current * reference, axis=-1) < 0.0] *= -1.0
    relative = _normalized_quaternions(
        _quaternion_multiply(_quaternion_conjugate(reference), current)
    )
    relative[relative[:, 3] < 0.0] *= -1.0
    vector_norm = np.linalg.norm(relative[:, :3], axis=-1)
    return 2.0 * np.arctan2(vector_norm, relative[:, 3])


def to_relative_actions(absolute_actions: np.ndarray) -> np.ndarray:
    """Match ``LatentLeRobotDataset._action_post_process`` without reading Stage 2."""
    absolute_actions = _actions(absolute_actions)
    relative = absolute_actions.copy()
    for _, (position_slice, quaternion_slice, _) in ARM_LAYOUT.items():
        position = absolute_actions[:, position_slice]
        quaternion = _normalized_quaternions(absolute_actions[:, quaternion_slice])
        relative[:, position_slice] = position - position[:1]
        initial_inverse = _quaternion_conjugate(
            np.repeat(quaternion[:1], len(quaternion), axis=0)
        )
        relative[:, quaternion_slice] = _normalized_quaternions(
            _quaternion_multiply(initial_inverse, quaternion)
        )
    return relative


def _prepend_condition_reference(
    relative_actions: np.ndarray,
    *,
    start_state: np.ndarray | None = None,
) -> np.ndarray:
    """Add the physical pose represented by VA_Server's zero condition block."""
    relative_actions = _actions(relative_actions)
    reference = np.zeros((1, 16), dtype=np.float64)
    if start_state is not None:
        start_state = np.asarray(start_state, dtype=np.float64).reshape(-1)
        if start_state.size < 16 or not np.all(np.isfinite(start_state[:16])):
            raise ValueError("start_state must contain a finite 16-D RoboTwin EEF state")
    for _, (_, quaternion_slice, gripper_column) in ARM_LAYOUT.items():
        reference[0, quaternion_slice.stop - 1] = 1.0
        reference[0, gripper_column] = (
            relative_actions[0, gripper_column]
            if start_state is None
            else start_state[gripper_column]
        )
    return np.concatenate([reference, relative_actions], axis=0)


def kinematic_series(relative_actions: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    relative_actions = _actions(relative_actions)
    if fps <= 0:
        raise ValueError("fps must be positive")
    result: dict[str, np.ndarray] = {}
    for arm, (position_slice, quaternion_slice, _) in ARM_LAYOUT.items():
        positions = relative_actions[:, position_slice]
        quaternions = relative_actions[:, quaternion_slice]
        linear_velocity_vector = np.diff(positions, axis=0) * fps
        angular_velocity_vector = quaternion_step_rotvecs(quaternions) * fps
        linear_acceleration_vector = np.diff(linear_velocity_vector, axis=0) * fps
        angular_acceleration_vector = np.diff(angular_velocity_vector, axis=0) * fps
        values = {
            "linear_velocity": np.linalg.norm(linear_velocity_vector, axis=-1),
            "angular_velocity": np.linalg.norm(angular_velocity_vector, axis=-1),
            "linear_displacement": np.linalg.norm(
                positions - positions[:1], axis=-1
            ),
            "angular_displacement": quaternion_reference_rotvecs(quaternions),
            "linear_acceleration": np.linalg.norm(
                linear_acceleration_vector, axis=-1
            ),
            "angular_acceleration": np.linalg.norm(
                angular_acceleration_vector, axis=-1
            ),
            "linear_jerk": np.linalg.norm(
                np.diff(linear_acceleration_vector, axis=0) * fps, axis=-1
            ),
            "angular_jerk": np.linalg.norm(
                np.diff(angular_acceleration_vector, axis=0) * fps, axis=-1
            ),
        }
        gripper = relative_actions[:, ARM_LAYOUT[arm][2]]
        gripper_velocity = np.abs(np.diff(gripper)) * fps
        gripper_acceleration = np.abs(np.diff(np.diff(gripper) * fps)) * fps
        values.update(
            {
                "gripper_velocity": gripper_velocity,
                "gripper_acceleration": gripper_acceleration,
                "gripper_jerk": np.abs(np.diff(gripper_acceleration)) * fps,
            }
        )
        for name, value in values.items():
            result[f"{arm}.{name}"] = value
    return result


@dataclass(frozen=True)
class RobustThreshold:
    median: float
    scale: float
    soft: float
    hard: float


@dataclass(frozen=True)
class Workspace:
    low: list[float]
    high: list[float]


@dataclass(frozen=True)
class KinematicProfile:
    schema_version: int
    fps: float
    thresholds: dict[str, RobustThreshold]
    workspaces: dict[str, Workspace]
    task_thresholds: dict[str, dict[str, RobustThreshold]]
    task_workspaces: dict[str, dict[str, Workspace]]
    calibration_trajectories: int
    calibration_chunks: int
    calibration_frames: int
    calibration_scope: str
    split_manifest_sha256: str
    calibration_episode_ids: list[str]
    soft_quantile: float
    hard_quantile: float
    minimum_score: float
    action_chunk_steps: int

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> "KinematicProfile":
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value.get("schema_version", -1)) != 6:
            raise ValueError(
                "Kinematic profile is obsolete; recalibrate with schema_version=6 "
                "to use expert-margin displacement-aware thresholds"
            )
        value["thresholds"] = {
            key: RobustThreshold(**row)
            for key, row in value["thresholds"].items()
        }
        value["workspaces"] = {
            key: Workspace(**row) for key, row in value["workspaces"].items()
        }
        value["task_thresholds"] = {
            task: {
                key: RobustThreshold(**row) for key, row in thresholds.items()
            }
            for task, thresholds in value["task_thresholds"].items()
        }
        value["task_workspaces"] = {
            task: {key: Workspace(**row) for key, row in spaces.items()}
            for task, spaces in value["task_workspaces"].items()
        }
        return cls(**value)


def _threshold(values: np.ndarray, soft: float, hard: float) -> RobustThreshold:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-8)
    hard_statistic = max(
        float(np.quantile(values, hard)), median + 6.0 * scale
    )
    hard_with_expert_margin = float(np.max(values)) * 1.05 + 1e-8
    return RobustThreshold(
        median=median,
        scale=scale,
        soft=max(float(np.quantile(values, soft)), median + 3.0 * scale),
        hard=max(hard_statistic, hard_with_expert_margin),
    )


def calibrate_kinematic_profile(
    absolute_trajectories: Iterable[np.ndarray],
    *,
    episode_ids: Iterable[str],
    fps: float,
    split_manifest_sha256: str,
    soft_quantile: float = 0.99,
    hard_quantile: float = 0.999,
    minimum_score: float = 0.5,
    workspace_quantile: float = 0.001,
    action_chunk_steps: int = 16,
    task_keys: Iterable[str] | None = None,
) -> KinematicProfile:
    trajectories = [_actions(value) for value in absolute_trajectories]
    ids = list(episode_ids)
    tasks = list(task_keys) if task_keys is not None else [""] * len(trajectories)
    if not trajectories or len(ids) != len(trajectories):
        raise ValueError("Calibration trajectories and episode ids must be non-empty")
    if len(tasks) != len(trajectories):
        raise ValueError("task_keys must align with calibration trajectories")
    if not 0.0 <= workspace_quantile < 0.5:
        raise ValueError("workspace_quantile must be in [0, 0.5)")
    if action_chunk_steps < 4:
        raise ValueError("action_chunk_steps must be at least four")
    collected: dict[str, list[np.ndarray]] = {}
    workspace_values = {arm: [] for arm in ARM_LAYOUT}
    task_collected: dict[str, dict[str, list[np.ndarray]]] = {}
    task_workspace_values: dict[str, dict[str, list[np.ndarray]]] = {}
    for absolute, task in zip(trajectories, tasks):
        if task:
            task_collected.setdefault(task, {})
            task_workspace_values.setdefault(
                task, {arm: [] for arm in ARM_LAYOUT}
            )
        for start in range(0, len(absolute) - action_chunk_steps + 1, action_chunk_steps):
            window = absolute[start : start + action_chunk_steps]
            relative = _prepend_condition_reference(
                to_relative_actions(window), start_state=window[0]
            )
            for key, values in kinematic_series(relative, fps).items():
                peak = np.asarray([np.max(values)], dtype=np.float64)
                collected.setdefault(key, []).append(peak)
                if task:
                    task_collected[task].setdefault(key, []).append(peak)
        for arm, (position_slice, _, _) in ARM_LAYOUT.items():
            workspace_values[arm].append(absolute[:, position_slice])
            if task:
                task_workspace_values[task][arm].append(
                    absolute[:, position_slice]
                )
    thresholds = {
        key: _threshold(np.concatenate(values), soft_quantile, hard_quantile)
        for key, values in collected.items()
    }
    if not thresholds:
        raise ValueError("No complete action chunks were available for calibration")
    def make_workspaces(values_by_arm: dict[str, list[np.ndarray]]):
        result = {}
        for arm, arrays in values_by_arm.items():
            values = np.concatenate(arrays, axis=0)
            low = np.quantile(values, workspace_quantile, axis=0)
            high = np.quantile(values, 1.0 - workspace_quantile, axis=0)
            margin = np.maximum((high - low) * 0.05, 1e-3)
            result[arm] = Workspace(
                low=(low - margin).tolist(), high=(high + margin).tolist()
            )
        return result

    workspaces = make_workspaces(workspace_values)
    task_thresholds = {
        task: {
            key: _threshold(np.concatenate(values), soft_quantile, hard_quantile)
            for key, values in metrics.items()
        }
        for task, metrics in task_collected.items()
    }
    task_workspaces = {
        task: make_workspaces(values)
        for task, values in task_workspace_values.items()
    }
    return KinematicProfile(
        schema_version=6,
        fps=float(fps),
        thresholds=thresholds,
        workspaces=workspaces,
        task_thresholds=task_thresholds,
        task_workspaces=task_workspaces,
        calibration_trajectories=len(trajectories),
        calibration_chunks=sum(
            len(value) // action_chunk_steps for value in trajectories
        ),
        calibration_frames=sum(len(value) for value in trajectories),
        calibration_scope="stage1_action_only",
        split_manifest_sha256=split_manifest_sha256,
        calibration_episode_ids=ids,
        soft_quantile=float(soft_quantile),
        hard_quantile=float(hard_quantile),
        minimum_score=float(minimum_score),
        action_chunk_steps=int(action_chunk_steps),
    )


def score_relative_actions(
    relative_actions: np.ndarray,
    profile: KinematicProfile,
    *,
    start_state: np.ndarray | None = None,
    task: str | None = None,
    gate_policy: str = "strict",
    workspace_scope: str = "task",
) -> dict:
    relative_actions = _actions(relative_actions)
    if len(relative_actions) != profile.action_chunk_steps:
        raise ValueError(
            f"Expected exactly {profile.action_chunk_steps} executable actions, "
            f"got {len(relative_actions)}"
        )
    series = kinematic_series(
        _prepend_condition_reference(relative_actions, start_state=start_state),
        profile.fps,
    )
    diagnostics = {}
    penalties: list[float] = []
    violations: list[str] = []
    thresholds = profile.task_thresholds.get(task or "", profile.thresholds)
    if workspace_scope not in WORKSPACE_SCOPES:
        raise ValueError(
            f"Unknown workspace_scope={workspace_scope!r}; "
            f"expected one of {WORKSPACE_SCOPES}"
        )
    workspaces = (
        profile.task_workspaces.get(task or "", profile.workspaces)
        if workspace_scope == "task"
        else profile.workspaces
    )
    threshold_scope = f"task:{task}" if task in profile.task_thresholds else "global"
    for key, threshold in thresholds.items():
        values = series[key]
        denominator = max(threshold.hard - threshold.soft, threshold.scale)
        normalized = np.maximum(values - threshold.soft, 0.0) / denominator
        penalty = float(np.mean(np.minimum(normalized, 4.0)))
        hard = bool(np.any(values > threshold.hard))
        penalties.append(penalty)
        if hard:
            violations.append(key)
        diagnostics[key] = {
            "maximum": float(np.max(values)),
            "soft": threshold.soft,
            "hard": threshold.hard,
            "penalty": penalty,
            "hard_violation": hard,
        }

    gripper_violations = []
    for arm, (_, _, gripper_column) in ARM_LAYOUT.items():
        gripper = relative_actions[:, gripper_column]
        if np.any((gripper < -0.05) | (gripper > 1.05)):
            gripper_violations.append(f"{arm}.gripper_limit")
    violations.extend(gripper_violations)

    workspace_diagnostics = {}
    if start_state is not None:
        start_state = np.asarray(start_state, dtype=np.float64).reshape(-1)
        if start_state.size < 16 or not np.all(np.isfinite(start_state[:16])):
            raise ValueError("start_state must contain a finite 16-D RoboTwin EEF state")
        for arm, (position_slice, _, _) in ARM_LAYOUT.items():
            absolute_positions = (
                start_state[position_slice][None]
                + relative_actions[:, position_slice]
            )
            workspace = workspaces[arm]
            low = np.asarray(workspace.low)
            high = np.asarray(workspace.high)
            outside = (absolute_positions < low) | (absolute_positions > high)
            if np.any(outside):
                violations.append(f"{arm}.eef_workspace")
            workspace_diagnostics[arm] = {
                "minimum": absolute_positions.min(axis=0).tolist(),
                "maximum": absolute_positions.max(axis=0).tolist(),
                "low": workspace.low,
                "high": workspace.high,
                "hard_violation": bool(np.any(outside)),
            }

    if penalties:
        strongest = sorted(penalties, reverse=True)[: min(3, len(penalties))]
        penalty = float(np.mean(strongest))
    else:
        penalty = 0.0
    score = float(np.exp(-penalty))
    gate_violations = action_gate_violations(violations, gate_policy)
    accepted = not gate_violations and score >= profile.minimum_score
    return {
        "action_score": score,
        "accepted": accepted,
        "hard_violations": sorted(set(violations)),
        "gate_violations": gate_violations,
        "gate_policy": gate_policy,
        "workspace_scope": workspace_scope,
        "diagnostics": diagnostics,
        "workspace": workspace_diagnostics,
        "condition_reference_checked": True,
        "threshold_scope": threshold_scope,
        "calibration_scope": profile.calibration_scope,
    }
