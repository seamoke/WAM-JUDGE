"""Build a RoboTwin episode index for RGB-based VLAC training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from robotwin_critic.common import iter_task_dirs, read_jsonl, task_name_from_dir
from robotwin_critic.vlac_finetune.common import DEFAULT_CAMERAS, write_jsonl


EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-splits", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    parser.add_argument("--min-episode-frames", type=int, default=8)
    parser.add_argument("--cameras", nargs=3, default=list(DEFAULT_CAMERAS))
    return parser.parse_args()


def parquet_files(task_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(task_dir.glob("data/chunk-*/episode_*.parquet")):
        match = EPISODE_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def videos_exist(
    task_dir: Path, parquet_path: Path, episode_index: int, cameras: list[str]
) -> bool:
    chunk = parquet_path.parent.name
    return all(
        (
            task_dir
            / "videos"
            / chunk
            / camera
            / f"episode_{episode_index:06d}.mp4"
        ).is_file()
        for camera in cameras
    )


def main() -> None:
    args = parse_args()
    include_splits = set(args.include_splits) if args.include_splits else None
    task_dirs = list(iter_task_dirs(args.dataset_root, include_splits))
    if args.max_tasks:
        task_dirs = task_dirs[: args.max_tasks]

    rows: list[dict] = []
    skipped_missing_rgb = 0
    skipped_missing_parquet = 0
    for dataset_split, task_dir in tqdm(task_dirs, desc="index RGB tasks"):
        episodes = read_jsonl(task_dir / "meta" / "episodes.jsonl")
        if args.max_episodes_per_task:
            episodes = episodes[: args.max_episodes_per_task]
        with (task_dir / "meta" / "info.json").open() as handle:
            info = json.load(handle)
        parquet_map = parquet_files(task_dir)

        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode.get("length", 0))
            if length < args.min_episode_frames:
                continue
            parquet_path = parquet_map.get(episode_index)
            if parquet_path is None:
                skipped_missing_parquet += 1
                continue
            if not videos_exist(task_dir, parquet_path, episode_index, args.cameras):
                skipped_missing_rgb += 1
                continue
            text = ""
            if episode.get("tasks"):
                text = episode["tasks"][0]
            elif episode.get("action_config"):
                text = episode["action_config"][0].get("action_text", "")
            task_name = task_name_from_dir(task_dir)
            rows.append(
                {
                    "dataset_root": str(args.dataset_root.resolve()),
                    "dataset_split": dataset_split,
                    "task_dir": str(task_dir.resolve()),
                    "task_name": task_name,
                    "task": task_name,
                    "episode_index": episode_index,
                    "length": length,
                    "fps": info.get("fps"),
                    "text": text,
                    "parquet_path": str(parquet_path.resolve()),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "output": str(args.output.resolve()),
        "episodes": len(rows),
        "task_dirs": len({row["task_dir"] for row in rows}),
        "task_names": len({row["task_name"] for row in rows}),
        "skipped_missing_parquet": skipped_missing_parquet,
        "skipped_missing_rgb": skipped_missing_rgb,
    }
    with args.output.with_suffix(".summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
