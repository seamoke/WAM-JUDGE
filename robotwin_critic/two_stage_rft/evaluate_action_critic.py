"""Sanity-check Action Critic using Stage-1 actions and synthetic corruption."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from robotwin_critic.two_stage_rft.kinematic_action_critic import (
    KinematicProfile,
    score_relative_actions,
    to_relative_actions,
)
from robotwin_critic.two_stage_rft.calibrate_action_critic import (
    find_parquet,
    load_actions,
)
from robotwin_critic.two_stage_rft.protocol import iter_episode_refs


def corrupt(actions: np.ndarray, kind: str) -> np.ndarray:
    value = actions.copy()
    middle = len(value) // 2
    if kind == "position_spike":
        value[middle, [0, 8]] += 0.25
    elif kind == "high_frequency_jitter":
        sign = (1 - 2 * (np.arange(len(value)) % 2)).astype(np.float32)
        value[:, 0] += 0.04 * sign
        value[:, 8] -= 0.04 * sign
    elif kind == "orientation_spike":
        value[middle, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        value[middle, 11:15] = np.array([0.0, 1.0, 0.0, 0.0])
    elif kind == "excess_speed":
        for columns in (slice(0, 3), slice(8, 11)):
            origin = value[:1, columns]
            value[:, columns] = origin + 5.0 * (value[:, columns] - origin)
    else:
        raise KeyError(kind)
    return value


def binary_auc(positive: list[float], negative: list[float]) -> float:
    comparisons = [
        float(pos > neg) + 0.5 * float(pos == neg)
        for pos in positive
        for neg in negative
    ]
    return float(np.mean(comparisons))


def evaluate(
    prepared_root: Path,
    profile_path: Path,
    *,
    max_segments: int,
) -> dict:
    profile = KinematicProfile.from_json(profile_path)
    refs = list(iter_episode_refs(prepared_root, stages=("stage1",)))
    real_scores = []
    real_accepts = []
    corrupt_scores: dict[str, list[float]] = defaultdict(list)
    corrupt_accepts: dict[str, list[bool]] = defaultdict(list)
    kinds = (
        "position_spike",
        "high_frequency_jitter",
        "orientation_spike",
        "excess_speed",
    )
    for ref in refs:
        actions = load_actions(find_parquet(ref.repo, ref.output_episode_index))
        if len(actions) < 8:
            continue
        segment = to_relative_actions(actions)
        result = score_relative_actions(segment, profile)
        real_scores.append(result["action_score"])
        real_accepts.append(result["accepted"])
        for kind in kinds:
            corrupted = score_relative_actions(corrupt(segment, kind), profile)
            corrupt_scores[kind].append(corrupted["action_score"])
            corrupt_accepts[kind].append(corrupted["accepted"])
        if max_segments and len(real_scores) >= max_segments:
            break
    all_corrupt_scores = [
        score for kind in kinds for score in corrupt_scores[kind]
    ]
    all_corrupt_accepts = [
        accepted for kind in kinds for accepted in corrupt_accepts[kind]
    ]
    return {
        "evaluation_scope": "stage1_visible_actions_with_synthetic_corruption",
        "reads_stage2_action": False,
        "real_segments": len(real_scores),
        "corrupted_segments": len(all_corrupt_scores),
        "roc_auc": binary_auc(real_scores, all_corrupt_scores),
        "false_reject_rate": float(1.0 - np.mean(real_accepts)),
        "false_accept_rate": float(np.mean(all_corrupt_accepts)),
        "mean_real_score": float(np.mean(real_scores)),
        "mean_corrupt_score": float(np.mean(all_corrupt_scores)),
        "per_negative_type": {
            kind: {
                "roc_auc": binary_auc(real_scores, corrupt_scores[kind]),
                "false_accept_rate": float(np.mean(corrupt_accepts[kind])),
                "mean_score": float(np.mean(corrupt_scores[kind])),
            }
            for kind in kinds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-segments", type=int, default=1000)
    args = parser.parse_args()
    result = evaluate(
        args.prepared_root,
        args.profile,
        max_segments=args.max_segments,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("ACTION_CRITIC_EVALUATION_OK")


if __name__ == "__main__":
    main()
