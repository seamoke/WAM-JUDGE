"""Analytic RoboTwin action reasonableness critic.

The critic is calibrated on real Stage-1 trajectories and scores generated
chunks using translational speed, angular speed, acceleration, and jerk.
Quaternion sign flips are handled exactly and gripper switches are reported
separately rather than treated as kinematic outliers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ARM_SLICES = ((slice(0, 3), slice(3, 7)), (slice(8, 11), slice(11, 15)))
GRIPPER_COLUMNS = (7, 15)
KINEMATIC_KEYS = (
    "linear_speed",
    "angular_speed",
    "linear_acceleration",
    "angular_acceleration",
    "linear_jerk",
    "angular_jerk",
)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Action contains an invalid zero quaternion")
    return quaternions / norms


def quaternion_step_angles(quaternions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(quaternion_step_rotvecs(quaternions), axis=-1)


def quaternion_step_rotvecs(quaternions: np.ndarray) -> np.ndarray:
    q = _normalize_quaternions(np.asarray(quaternions, dtype=np.float64))
    current = q[1:].copy()
    previous = q[:-1]
    # q and -q denote the same rotation; choose the shorter geodesic.
    current[np.sum(current * previous, axis=-1) < 0.0] *= -1.0
    px, py, pz, pw = np.moveaxis(previous, -1, 0)
    cx, cy, cz, cw = np.moveaxis(current, -1, 0)
    # conjugate(previous) * current, in scipy's xyzw convention.
    vector = np.stack(
        (
            pw * cx - px * cw - py * cz + pz * cy,
            pw * cy + px * cz - py * cw - pz * cx,
            pw * cz - px * cy + py * cx - pz * cw,
        ),
        axis=-1,
    )
    scalar = pw * cw + px * cx + py * cy + pz * cz
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(scalar, 0.0, None))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.zeros_like(angle),
        where=vector_norm > 1e-12,
    )
    return vector * scale[:, None]


def kinematic_series(actions: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16:
        raise ValueError(f"Expected actions with shape [T,16], got {actions.shape}")
    if len(actions) < 4:
        raise ValueError("At least four action frames are required")
    if fps <= 0:
        raise ValueError("fps must be positive")

    arm_metrics: dict[str, list[np.ndarray]] = {
        key: [] for key in KINEMATIC_KEYS
    }
    for position_slice, quaternion_slice in ARM_SLICES:
        position = actions[:, position_slice]
        linear_velocity = np.diff(position, axis=0) * fps
        linear_speed = np.linalg.norm(linear_velocity, axis=-1)
        angular_velocity = (
            quaternion_step_rotvecs(actions[:, quaternion_slice]) * fps
        )
        angular_speed = np.linalg.norm(angular_velocity, axis=-1)
        linear_acceleration = np.linalg.norm(
            np.diff(linear_velocity, axis=0) * fps, axis=-1
        )
        angular_acceleration_vector = np.diff(angular_velocity, axis=0) * fps
        angular_acceleration = np.linalg.norm(
            angular_acceleration_vector, axis=-1
        )
        linear_jerk = np.linalg.norm(
            np.diff(np.diff(linear_velocity, axis=0) * fps, axis=0) * fps,
            axis=-1,
        )
        angular_jerk = np.linalg.norm(
            np.diff(angular_acceleration_vector, axis=0) * fps, axis=-1
        )
        values = (
            linear_speed,
            angular_speed,
            linear_acceleration,
            angular_acceleration,
            linear_jerk,
            angular_jerk,
        )
        for key, value in zip(KINEMATIC_KEYS, values):
            arm_metrics[key].append(value)

    result = {
        key: np.concatenate(values) if values else np.empty(0)
        for key, values in arm_metrics.items()
    }
    gripper = actions[:, GRIPPER_COLUMNS]
    result["gripper_switch_rate"] = np.asarray(
        [np.mean(np.abs(np.diff(gripper, axis=0)) > 0.5)]
    )
    return result


@dataclass(frozen=True)
class Threshold:
    median: float
    scale: float
    soft: float
    hard: float


@dataclass(frozen=True)
class ActionCriticProfile:
    fps: float
    soft_quantile: float
    hard_quantile: float
    thresholds: dict[str, Threshold]
    calibration_frames: int
    calibration_trajectories: int
    calibration_scope: str = "stage1_only"

    def to_json(self, path: Path) -> None:
        value = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def from_json(cls, path: Path) -> "ActionCriticProfile":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        value["thresholds"] = {
            key: Threshold(**threshold)
            for key, threshold in value["thresholds"].items()
        }
        return cls(**value)


def calibrate(
    trajectories: list[np.ndarray],
    *,
    fps: float,
    soft_quantile: float = 0.99,
    hard_quantile: float = 0.999,
) -> ActionCriticProfile:
    if not trajectories:
        raise ValueError("No calibration trajectories")
    collected: dict[str, list[np.ndarray]] = {key: [] for key in KINEMATIC_KEYS}
    total_frames = 0
    for actions in trajectories:
        total_frames += len(actions)
        series = kinematic_series(actions, fps)
        for key in KINEMATIC_KEYS:
            collected[key].append(series[key])
    thresholds = {}
    for key, arrays in collected.items():
        values = np.concatenate(arrays)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = max(1.4826 * mad, 1e-8)
        thresholds[key] = Threshold(
            median=median,
            scale=robust_scale,
            soft=float(np.quantile(values, soft_quantile)),
            hard=float(np.quantile(values, hard_quantile)),
        )
    return ActionCriticProfile(
        fps=float(fps),
        soft_quantile=float(soft_quantile),
        hard_quantile=float(hard_quantile),
        thresholds=thresholds,
        calibration_frames=total_frames,
        calibration_trajectories=len(trajectories),
    )


def score_actions(
    actions: np.ndarray, profile: ActionCriticProfile
) -> dict:
    series = kinematic_series(actions, profile.fps)
    diagnostics = {}
    penalties = []
    hard_violation = False
    for key in KINEMATIC_KEYS:
        values = series[key]
        threshold = profile.thresholds[key]
        exceedance = np.maximum(values - threshold.soft, 0.0)
        denominator = max(threshold.hard - threshold.soft, threshold.scale)
        normalized = exceedance / denominator
        metric_penalty = float(np.mean(np.minimum(normalized, 2.0)))
        metric_hard = bool(np.any(values > threshold.hard))
        penalties.append(metric_penalty)
        hard_violation = hard_violation or metric_hard
        diagnostics[key] = {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "soft_threshold": threshold.soft,
            "hard_threshold": threshold.hard,
            "hard_violation": metric_hard,
            "penalty": metric_penalty,
        }
    penalty = float(np.mean(penalties))
    score = float(np.exp(-penalty))
    return {
        "action_score": score,
        "accepted": bool(not hard_violation and score >= 0.5),
        "hard_violation": hard_violation,
        "gripper_switch_rate": float(series["gripper_switch_rate"][0]),
        "diagnostics": diagnostics,
    }
