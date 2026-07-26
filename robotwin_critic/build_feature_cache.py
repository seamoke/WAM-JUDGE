from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, read_jsonl, stable_int, state_feature, text_feature


def _episode_key(row: dict) -> str:
    parts = [
        row.get("dataset_split", ""),
        row.get("task_name", ""),
        str(row.get("episode_index", "")),
        row.get("parquet_path", ""),
    ]
    return "|".join(parts)


def _cache_path(cache_root: Path, row: dict) -> Path:
    split = str(row.get("dataset_split", "unknown"))
    task = str(row.get("task_name", "unknown"))
    episode = int(row.get("episode_index", 0))
    suffix = stable_int(_episode_key(row))
    return cache_root / split / task / f"episode_{episode:06d}_{suffix:x}.pt"


def _needed_frames(rows: Iterable[dict]) -> set[int]:
    frames: set[int] = set()
    for row in rows:
        frames.add(int(row["frame_i"]))
        frames.add(int(row["frame_j"]))
        frames.add(int(row["final_frame"]))
    return frames


def build_cache(args: argparse.Namespace) -> dict:
    rows = read_jsonl(args.input_jsonl)
    if args.limit_rows:
        rows = rows[: args.limit_rows]

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_episode_key(row), []).append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_episodes = 0
    skipped_rows = 0

    with args.output_jsonl.open("w") as out:
        for episode_rows in tqdm(grouped.values(), desc="build RoboTwin process feature cache"):
            ref = episode_rows[0]
            path = _cache_path(args.cache_root, ref)
            try:
                frames = sorted(_needed_frames(episode_rows))
                dtype = torch.float16 if args.float16 else torch.float32
                if path.exists() and not args.overwrite:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    states = payload.get("states", {})
                    missing_frames = [int(frame) for frame in frames if int(frame) not in states]
                    for frame in missing_frames:
                        states[frame] = state_feature(ref["latents"], frame).to(dtype)
                    payload["states"] = states
                    payload["num_frames"] = len(states)
                    if missing_frames:
                        torch.save(payload, path)
                else:
                    states = {
                        int(frame): state_feature(ref["latents"], int(frame)).to(dtype)
                        for frame in frames
                    }
                    payload = {
                        "states": states,
                        "text_emb": text_feature(ref["latents"]).to(dtype),
                        "task_name": ref["task_name"],
                        "episode_index": int(ref["episode_index"]),
                        "dataset_split": ref.get("dataset_split", ""),
                        "num_frames": len(states),
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, path)
                for row in episode_rows:
                    cached = dict(row)
                    cached["feature_cache_path"] = str(path)
                    out.write(json.dumps(cached, ensure_ascii=False, separators=(",", ":")) + "\n")
                    kept += 1
            except Exception as exc:
                skipped_episodes += 1
                skipped_rows += len(episode_rows)
                if args.verbose:
                    print(
                        json.dumps(
                            {
                                "skipped_episode": _episode_key(ref),
                                "rows": len(episode_rows),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                    )

    summary = {
        "input_rows": len(rows),
        "episodes": len(grouped),
        "kept_rows": kept,
        "skipped_episodes": skipped_episodes,
        "skipped_rows": skipped_rows,
        "output_jsonl": str(args.output_jsonl),
        "cache_root": str(args.cache_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute compact RoboTwin process critic state features.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train_cached.jsonl")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "feature_cache" / "process")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_cache(parse_args())


if __name__ == "__main__":
    main()
