"""Build VLAC-compatible RoboTwin RGB pair data with episode-level splits."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .common import (
    DEFAULT_CAMERAS,
    SYSTEM_PROMPT,
    VideoDecodeError,
    VideoFrameReader,
    format_score,
    make_tshape_state,
    normalized_pixel_difference,
    pair_prompt,
    read_jsonl,
    stable_fraction,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default="/workspace/lingbot-va/train_out/critic/robotwin/index.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/lingbot-va/train_out/critic/robotwin/vlac_finetune/smoke_2task",
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--groups-per-episode", type=int, default=8)
    parser.add_argument("--min-long-gap", type=int, default=8)
    parser.add_argument("--adjacent-stride", type=int, default=2)
    parser.add_argument("--eval-frames", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--pixel-static-threshold", type=float, default=0.01)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--trainer-val-samples", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cameras", nargs=3, default=list(DEFAULT_CAMERAS))
    return parser.parse_args()


def row_value(row: dict[str, Any], *keys: str, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def episode_identity(row: dict[str, Any]) -> tuple[str, int]:
    task = str(row_value(row, "task", "task_name"))
    episode = int(row_value(row, "episode_index", "episode", "episode_id"))
    return task, episode


def episode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    _, episode = episode_identity(row)
    dataset_split = str(row_value(row, "dataset_split", default="unknown_split"))
    task_dir = Path(str(row_value(row, "task_dir", default="unknown_task_dir"))).name
    return dataset_split, task_dir, episode


def task_root_and_chunk(row: dict[str, Any]) -> tuple[Path, str]:
    parquet_path = Path(str(row_value(row, "parquet_path", "data_path")))
    task_dir = row_value(row, "task_dir", "episode_root")
    if task_dir is not None:
        return Path(task_dir), parquet_path.parent.name
    parts = parquet_path.parts
    data_indices = [index for index, part in enumerate(parts) if part == "data"]
    if not data_indices:
        raise ValueError(f"Cannot derive task root from {parquet_path}")
    data_index = data_indices[-1]
    task_root = Path(*parts[:data_index])
    chunk = parquet_path.parent.name
    return task_root, chunk


def video_paths(row: dict[str, Any], cameras: Sequence[str]) -> dict[str, Path]:
    task_root, chunk = task_root_and_chunk(row)
    _, episode = episode_identity(row)
    paths = {
        camera: task_root / "videos" / chunk / camera / f"episode_{episode:06d}.mp4"
        for camera in cameras
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RoboTwin RGB videos: {missing}")
    return paths


def extract_state(
    reader: VideoFrameReader,
    row: dict[str, Any],
    frame_index: int,
    cameras: Sequence[str],
    cache_root: Path,
    width: int,
    jpeg_quality: int,
) -> Path:
    task, episode = episode_identity(row)
    dataset_split, source_task_dir, _ = episode_key(row)
    destination = (
        cache_root
        / dataset_split
        / source_task_dir
        / task
        / f"episode_{episode:06d}"
        / f"frame_{frame_index:06d}.jpg"
    )
    if destination.exists():
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = video_paths(row, cameras)
    images = [reader.read(paths[camera], frame_index) for camera in cameras]
    mosaic = make_tshape_state(images, output_width=width)
    ok = cv2.imwrite(
        str(destination),
        cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise OSError(f"Failed to write {destination}")
    return destination.resolve()


def load_cached_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def local_forward_score(start: int, end: int, length: int) -> float:
    denominator = max(1, length - 1 - start)
    return float(np.clip(100.0 * (end - start) / denominator, 0.0, 100.0))


def swift_sample(
    task: str,
    first_path: Path,
    second_path: Path,
    score: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair_prompt(task)},
            {"role": "assistant", "content": format_score(score)},
        ],
        "images": [str(first_path), str(second_path)],
        "metadata": metadata,
    }


def ordered_pair_samples(
    row: dict[str, Any],
    first_index: int,
    second_index: int,
    first_path: Path,
    second_path: Path,
    goal_path: Path,
    pair_kind: str,
    static_threshold: float,
) -> list[dict[str, Any]]:
    task, episode = episode_identity(row)
    length = int(row_value(row, "length", "episode_length", "num_frames"))
    forward_score = local_forward_score(first_index, second_index, length)
    pixel_difference = normalized_pixel_difference(
        load_cached_rgb(first_path), load_cached_rgb(second_path)
    )
    if pixel_difference < static_threshold:
        forward_score = 0.0

    common = {
        "task": task,
        "dataset_split": row_value(row, "dataset_split", default="unknown_split"),
        "task_dir": row_value(row, "task_dir", default=None),
        "episode_index": episode,
        "pair_kind": pair_kind,
        "pixel_difference": pixel_difference,
        "length": length,
        "goal_image": str(goal_path),
    }
    return [
        swift_sample(
            task,
            first_path,
            second_path,
            forward_score,
            {**common, "i": first_index, "j": second_index, "target": forward_score},
        ),
        swift_sample(
            task,
            second_path,
            first_path,
            -forward_score,
            {**common, "i": second_index, "j": first_index, "target": -forward_score},
        ),
    ]


def state_score_record(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample["metadata"]
    target = float(metadata["target"])
    if abs(target) < 0.05:
        label = 0
    else:
        label = 1 if target > 0 else -1
    return {
        "task": metadata["task"],
        "dataset_split": metadata["dataset_split"],
        "task_dir": metadata["task_dir"],
        "episode_index": metadata["episode_index"],
        "i": metadata["i"],
        "j": metadata["j"],
        "length": metadata["length"],
        "pair_kind": metadata["pair_kind"],
        "state_i": sample["images"][0],
        "state_j": sample["images"][1],
        "goal": metadata["goal_image"],
        "target_delta": target,
        "label": label,
    }


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task, _ = episode_identity(row)
        by_task[task].append(row)

    if args.tasks:
        selected_tasks = list(args.tasks)
        missing = sorted(set(selected_tasks).difference(by_task))
        if missing:
            raise KeyError(f"Tasks are absent from index: {missing}")
    else:
        selected_tasks = sorted(by_task)
        if args.max_tasks > 0:
            selected_tasks = selected_tasks[: args.max_tasks]

    selected: list[dict[str, Any]] = []
    for task in selected_tasks:
        task_rows = sorted(by_task[task], key=episode_key)
        if args.episodes_per_task > 0:
            task_rows = task_rows[: args.episodes_per_task]
        selected.extend(task_rows)
    return selected


def episode_splits(
    rows: Sequence[dict[str, Any]], seed: int, val_ratio: float
) -> dict[tuple[str, str, int], str]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[episode_identity(row)[0]].append(row)
    splits: dict[tuple[str, str, int], str] = {}
    for task, task_rows in by_task.items():
        ranked = sorted(
            task_rows,
            key=lambda row: stable_fraction(seed, *episode_key(row), "split"),
        )
        val_count = min(len(ranked) - 1, max(1, int(round(len(ranked) * val_ratio))))
        val_ids = {episode_key(row) for row in ranked[:val_count]}
        for row in task_rows:
            identity = episode_key(row)
            splits[identity] = "val" if identity in val_ids else "train"
    return splits


def trainer_validation_subset(
    rows: list[dict[str, Any]], max_samples: int
) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    if len(rows) % 2:
        raise RuntimeError("Validation rows must contain adjacent forward/reverse pairs")
    groups_by_task: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for offset in range(0, len(rows), 2):
        pair = rows[offset : offset + 2]
        task = str(pair[0]["metadata"]["task"])
        groups_by_task[task].append(pair)

    max_groups = max(1, max_samples // 2)
    tasks = sorted(groups_by_task)
    base, remainder = divmod(max_groups, len(tasks))
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        groups = groups_by_task[task]
        allocation = base + (1 if task_index < remainder else 0)
        allocation = min(len(groups), max(1, allocation))
        indices = np.unique(
            np.linspace(0, len(groups) - 1, allocation, dtype=int)
        ).tolist()
        for index in indices:
            selected.extend(groups[index])
    return selected


def build_episode(
    payload: tuple[dict[str, Any], bool, argparse.Namespace],
) -> dict[str, Any] | None:
    row, is_val, args = payload
    cv2.setNumThreads(1)
    output_dir = Path(args.output_dir)
    cache_root = output_dir / "rgb_cache"
    task, episode = episode_identity(row)
    length = int(row_value(row, "length", "episode_length", "num_frames"))
    if length < max(args.min_long_gap + 1, args.adjacent_stride + 1):
        return None
    source_key = episode_key(row)
    rng = np.random.default_rng(
        int(stable_fraction(args.seed, *source_key, "pairs") * (2**32 - 1))
    )
    episode_samples: list[dict[str, Any]] = []
    val_trajectory = None
    try:
        with VideoFrameReader() as reader:
            goal_path = extract_state(
                reader,
                row,
                length - 1,
                args.cameras,
                cache_root,
                args.image_width,
                args.jpeg_quality,
            )
            max_start = length - args.min_long_gap - 1
            for _ in range(args.groups_per_episode):
                start = int(rng.integers(0, max_start + 1))
                adjacent_end = min(length - 1, start + args.adjacent_stride)
                long_end = int(rng.integers(start + args.min_long_gap, length))
                required = sorted({start, adjacent_end, long_end})
                state_paths = {
                    frame: extract_state(
                        reader,
                        row,
                        frame,
                        args.cameras,
                        cache_root,
                        args.image_width,
                        args.jpeg_quality,
                    )
                    for frame in required
                }
                episode_samples.extend(
                    ordered_pair_samples(
                        row,
                        start,
                        adjacent_end,
                        state_paths[start],
                        state_paths[adjacent_end],
                        goal_path,
                        "adjacent",
                        args.pixel_static_threshold,
                    )
                )
                episode_samples.extend(
                    ordered_pair_samples(
                        row,
                        start,
                        long_end,
                        state_paths[start],
                        state_paths[long_end],
                        goal_path,
                        "long",
                        args.pixel_static_threshold,
                    )
                )

            if is_val:
                eval_indices = np.unique(
                    np.linspace(
                        0,
                        length - 1,
                        min(args.eval_frames, length),
                        dtype=int,
                    )
                ).tolist()
                eval_paths = [
                    str(
                        extract_state(
                            reader,
                            row,
                            frame,
                            args.cameras,
                            cache_root,
                            args.image_width,
                            args.jpeg_quality,
                        )
                    )
                    for frame in eval_indices
                ]
                val_trajectory = {
                    "task": task,
                    "dataset_split": source_key[0],
                    "task_dir": source_key[1],
                    "episode_index": episode,
                    "length": length,
                    "frame_indices": eval_indices,
                    "images": eval_paths,
                    "text": row_value(
                        row,
                        "text",
                        "instruction",
                        "language_instruction",
                        default=task,
                    ),
                }
    except (FileNotFoundError, IndexError, OSError, VideoDecodeError) as exc:
        return {
            "skipped": {
                "task": task,
                "dataset_split": source_key[0],
                "task_dir": source_key[1],
                "episode_index": episode,
                "length": length,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        }

    return {
        "split": "val" if is_val else "train",
        "samples": episode_samples,
        "val_trajectory": val_trajectory,
        "summary": {
            "task": task,
            "dataset_split": source_key[0],
            "task_dir": source_key[1],
            "episode_index": episode,
            "split": "val" if is_val else "train",
            "samples": len(episode_samples),
            "length": length,
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    rows = select_rows(read_jsonl(args.index), args)
    split_by_episode = episode_splits(rows, args.seed, args.val_ratio)
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    val_trajectories: list[dict[str, Any]] = []
    episode_summary: list[dict[str, Any]] = []
    skipped_episodes: list[dict[str, Any]] = []

    workers = max(1, int(getattr(args, "workers", 1)))
    progress_every = max(0, int(getattr(args, "progress_every", 100)))
    payloads = (
        (row, split_by_episode[episode_key(row)] == "val", args)
        for row in rows
    )
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = (
            executor.map(build_episode, payloads, chunksize=1)
            if executor is not None
            else map(build_episode, payloads)
        )
        for processed, result in enumerate(results, 1):
            if result is None:
                pass
            elif "skipped" in result:
                skipped_episodes.append(result["skipped"])
            else:
                destination = (
                    val_samples if result["split"] == "val" else train_samples
                )
                destination.extend(result["samples"])
                if result["val_trajectory"] is not None:
                    val_trajectories.append(result["val_trajectory"])
                episode_summary.append(result["summary"])
            if progress_every and (
                processed % progress_every == 0 or processed == len(rows)
            ):
                elapsed = max(time.perf_counter() - started, 1e-6)
                rate = processed / elapsed
                remaining = max(0, len(rows) - processed)
                print(
                    json.dumps(
                        {
                            "processed_episodes": processed,
                            "total_episodes": len(rows),
                            "successful_episodes": len(episode_summary),
                            "skipped_episodes": len(skipped_episodes),
                            "episodes_per_second": rate,
                            "eta_seconds": remaining / rate,
                        }
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    if not train_samples or not val_samples:
        raise RuntimeError(
            f"Episode-level split produced train={len(train_samples)} val={len(val_samples)}; "
            "increase episodes or adjust --val-ratio"
        )
    state_score_train = [state_score_record(sample) for sample in train_samples]
    state_score_val = [state_score_record(sample) for sample in val_samples]
    trainer_val_samples = trainer_validation_subset(
        val_samples, args.trainer_val_samples
    )
    counts = {
        "train_samples": write_jsonl(output_dir / "train.jsonl", train_samples),
        "val_samples": write_jsonl(output_dir / "val.jsonl", val_samples),
        "trainer_val_samples": write_jsonl(
            output_dir / "val_train.jsonl", trainer_val_samples
        ),
        "state_score_train_samples": write_jsonl(
            output_dir / "state_score_train.jsonl", state_score_train
        ),
        "state_score_val_samples": write_jsonl(
            output_dir / "state_score_val.jsonl", state_score_val
        ),
        "val_trajectories": write_jsonl(output_dir / "val_trajectories.jsonl", val_trajectories),
        "episodes": len(episode_summary),
        "skipped_episodes": len(skipped_episodes),
        "workers": workers,
        "build_seconds": time.perf_counter() - started,
        "tasks": sorted({item["task"] for item in episode_summary}),
        "static_pairs": sum(
            abs(float(item["metadata"]["target"])) < 0.05 for item in train_samples + val_samples
        ),
    }
    with (output_dir / "build_summary.json").open("w") as handle:
        json.dump(
            {
                **counts,
                "episode_summary": episode_summary,
                "skipped_episode_summary": skipped_episodes,
            },
            handle,
            indent=2,
        )
    print(json.dumps(counts, indent=2))
    return counts


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
