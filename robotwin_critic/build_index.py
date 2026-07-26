from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from robotwin_critic.common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    CAM_KEYS,
    build_task_file_maps,
    entry_ref,
    iter_task_dirs,
    read_jsonl,
    task_name_from_dir,
    write_jsonl,
)


def build_index(args: argparse.Namespace) -> list[dict]:
    include_splits = set(args.include_splits) if args.include_splits else None
    rows: list[dict] = []
    task_dirs = list(iter_task_dirs(args.dataset_root, include_splits))
    if args.max_tasks:
        task_dirs = task_dirs[: args.max_tasks]

    for dataset_split, task_dir in tqdm(task_dirs, desc="index tasks"):
        task_name = task_name_from_dir(task_dir)
        episodes = read_jsonl(task_dir / "meta" / "episodes.jsonl")
        if args.max_episodes_per_task:
            episodes = episodes[: args.max_episodes_per_task]
        with (task_dir / "meta" / "info.json").open() as f:
            info = json.load(f)
        latent_map, parquet_map = build_task_file_maps(task_dir)

        for ep in episodes:
            episode_index = int(ep["episode_index"])
            length = int(ep.get("length", 0))
            if length < args.min_episode_frames:
                continue
            latents = latent_map.get(episode_index)
            parquet_path = parquet_map.get(episode_index)
            if (
                latents is None
                or parquet_path is None
                or any(cam not in latents for cam in CAM_KEYS)
            ):
                continue
            text = ""
            if ep.get("tasks"):
                text = ep["tasks"][0]
            elif ep.get("action_config"):
                text = ep["action_config"][0].get("action_text", "")
            rows.append(
                {
                    "dataset_root": str(args.dataset_root),
                    "dataset_split": dataset_split,
                    "task_dir": str(task_dir),
                    "task_name": task_name,
                    "episode_index": episode_index,
                    "length": length,
                    "fps": info.get("fps"),
                    "text": text,
                    "parquet_path": parquet_path,
                    "latents": latents,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    summary = {
        "dataset_root": str(args.dataset_root),
        "output": str(args.output),
        "episodes": len(rows),
        "tasks": len({r["task_dir"] for r in rows}),
        "task_names": len({r["task_name"] for r in rows}),
    }
    with args.output.with_suffix(".summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a RoboTwin critic episode index.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "index.jsonl")
    parser.add_argument("--include-splits", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--min-episode-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    build_index(parse_args())


if __name__ == "__main__":
    main()
