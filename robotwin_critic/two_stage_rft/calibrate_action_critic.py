"""Calibrate the kinematic Action Critic strictly from fixed Stage-1 actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotwin_critic.two_stage_rft.kinematic_action_critic import (
    calibrate_kinematic_profile,
)
from robotwin_critic.two_stage_rft.protocol import (
    iter_episode_refs,
    sha256_file,
)


def load_actions(path: Path):
    import numpy as np
    import pyarrow.parquet as pq

    return np.asarray(
        pq.read_table(path, columns=["action"])["action"].to_pylist(),
        dtype=np.float32,
    )


def find_parquet(repo: Path, episode_index: int) -> Path:
    matches = list(repo.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for {repo} episode {episode_index}, got {matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override action fps. Zero derives and verifies fps from info.json.",
    )
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--soft-quantile", type=float, default=0.99)
    parser.add_argument("--hard-quantile", type=float, default=0.999)
    parser.add_argument("--minimum-score", type=float, default=0.5)
    parser.add_argument("--workspace-quantile", type=float, default=0.001)
    parser.add_argument("--action-chunk-steps", type=int, default=16)
    parser.add_argument("--expected-per-domain-total", type=int, default=50)
    parser.add_argument("--expected-stage1-per-domain", type=int, default=30)
    args = parser.parse_args()

    refs = list(
        iter_episode_refs(
            args.prepared_root,
            stages=("stage1",),
            expected_per_domain_total=args.expected_per_domain_total,
            expected_stage1_per_domain=args.expected_stage1_per_domain,
        )
    )
    if args.max_trajectories:
        refs = refs[: args.max_trajectories]
    fps_values = set()
    trajectories = []
    episode_ids = []
    task_keys = []
    for ref in refs:
        with (ref.repo / "meta" / "info.json").open(encoding="utf-8") as handle:
            fps_values.add(float(json.load(handle)["fps"]))
        actions = load_actions(find_parquet(ref.repo, ref.output_episode_index))
        if len(actions) < args.action_chunk_steps:
            continue
        trajectories.append(actions)
        episode_ids.append(
            f"{ref.task}/{ref.domain}/source-{ref.source_episode_index}"
        )
        task_keys.append(ref.task)
    if args.fps > 0:
        fps = args.fps
    elif len(fps_values) == 1:
        fps = fps_values.pop()
    else:
        raise ValueError(f"Dataset repositories disagree on action fps: {fps_values}")
    profile = calibrate_kinematic_profile(
        trajectories,
        episode_ids=episode_ids,
        fps=fps,
        split_manifest_sha256=sha256_file(
            args.prepared_root / "split_manifest.json"
        ),
        soft_quantile=args.soft_quantile,
        hard_quantile=args.hard_quantile,
        minimum_score=args.minimum_score,
        workspace_quantile=args.workspace_quantile,
        action_chunk_steps=args.action_chunk_steps,
        task_keys=task_keys,
    )
    profile.to_json(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "calibration_scope": profile.calibration_scope,
                "calibration_trajectories": profile.calibration_trajectories,
                "calibration_chunks": profile.calibration_chunks,
                "calibration_frames": profile.calibration_frames,
                "split_manifest_sha256": profile.split_manifest_sha256,
            },
            indent=2,
        )
    )
    print("ACTION_CRITIC_CALIBRATION_OK")


if __name__ == "__main__":
    main()
