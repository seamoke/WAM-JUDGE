#!/usr/bin/env python3
"""Audit the clean RoboTwin LeRobot dataset used by LingBot-VA training."""

import argparse
import json
from pathlib import Path


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--expected-episodes", type=int, default=2500)
    parser.add_argument(
        "--allow-missing-latents",
        type=int,
        default=8,
        help="Known published snapshot tolerance. Use 0 for a strict mirror.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    args = parse_args()
    root = args.dataset_root.resolve()
    info_paths = sorted(root.glob("*/meta/info.json"))
    if len(info_paths) != args.expected_tasks:
        raise SystemExit(
            f"Expected {args.expected_tasks} clean task repositories, "
            f"found {len(info_paths)} under {root}"
        )

    total_episodes = 0
    total_segments = 0
    valid_segments = 0
    missing = []

    for info_path in info_paths:
        repo = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        chunks_size = int(info.get("chunks_size", 1000))
        episodes_path = repo / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            raise SystemExit(f"Missing {episodes_path}")
        if not any((repo / "data").rglob("*.parquet")):
            raise SystemExit(f"Missing LeRobot parquet data under {repo / 'data'}")
        episodes = read_jsonl(episodes_path)
        total_episodes += len(episodes)

        for episode in episodes:
            episode_index = int(episode["episode_index"])
            chunk = episode_index // chunks_size
            for action_config in episode.get("action_config", []):
                total_segments += 1
                start = int(action_config["start_frame"])
                end = int(action_config["end_frame"])
                absent = []
                for camera in CAMERA_KEYS:
                    path = (
                        repo
                        / "latents"
                        / f"chunk-{chunk:03d}"
                        / camera
                        / f"episode_{episode_index:06d}_{start}_{end}.pth"
                    )
                    if not path.is_file():
                        absent.append(str(path.relative_to(root)))
                if absent:
                    missing.append(
                        {
                            "task_repo": repo.name,
                            "episode": episode_index,
                            "start_frame": start,
                            "end_frame": end,
                            "missing": absent,
                        }
                    )
                else:
                    valid_segments += 1

    summary = {
        "dataset_root": str(root),
        "task_repositories": len(info_paths),
        "episodes": total_episodes,
        "segments": total_segments,
        "valid_segments": valid_segments,
        "segments_with_missing_latents": len(missing),
        "missing": missing,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps({key: value for key, value in summary.items() if key != "missing"}, indent=2))
    if total_episodes != args.expected_episodes:
        raise SystemExit(
            f"Expected {args.expected_episodes} episodes, found {total_episodes}"
        )
    if len(missing) > args.allow_missing_latents:
        raise SystemExit(
            f"{len(missing)} segments have missing camera latents; "
            f"allowed={args.allow_missing_latents}"
        )
    print("DATASET_AUDIT_OK")


if __name__ == "__main__":
    main()
