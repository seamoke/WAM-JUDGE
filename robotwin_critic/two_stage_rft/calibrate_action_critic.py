"""Calibrate the analytic Action Critic strictly from fixed Stage-1 actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotwin_critic.two_stage_rft.action_critic import calibrate
from robotwin_critic.two_stage_rft.protocol import iter_episode_refs


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


def episode_segments(repo: Path, episode_index: int) -> list[tuple[int, int]]:
    with (repo / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["episode_index"]) == episode_index:
                return [
                    (int(config["start_frame"]), int(config["end_frame"]))
                    for config in row.get("action_config", [])
                ]
    raise KeyError(f"{repo}: episode {episode_index} is absent from metadata")


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
    args = parser.parse_args()

    refs = list(iter_episode_refs(args.prepared_root, stages=("stage1",)))
    if args.max_trajectories:
        refs = refs[: args.max_trajectories]
    fps_values = set()
    trajectories = []
    for ref in refs:
        with (ref.repo / "meta" / "info.json").open(encoding="utf-8") as handle:
            fps_values.add(float(json.load(handle)["fps"]))
        actions = load_actions(find_parquet(ref.repo, ref.output_episode_index))
        for start, end in episode_segments(ref.repo, ref.output_episode_index):
            if end - start >= 4:
                trajectories.append(actions[start:end])
    if args.max_trajectories:
        trajectories = trajectories[: args.max_trajectories]
    if args.fps > 0:
        fps = args.fps
    elif len(fps_values) == 1:
        fps = fps_values.pop()
    else:
        raise ValueError(f"Dataset repositories disagree on action fps: {fps_values}")
    profile = calibrate(
        trajectories,
        fps=fps,
        soft_quantile=args.soft_quantile,
        hard_quantile=args.hard_quantile,
    )
    profile.to_json(args.output)
    print(json.dumps({"output": str(args.output), **profile.__dict__}, default=str))
    print("ACTION_CRITIC_CALIBRATION_OK")


if __name__ == "__main__":
    main()
