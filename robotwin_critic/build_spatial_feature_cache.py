from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from robotwin_critic.common import (
    CAM_KEYS,
    DEFAULT_OUTPUT_ROOT,
    load_latent_file,
    read_jsonl,
    stable_int,
    text_feature,
)


def _episode_key(row: dict) -> str:
    return "|".join(
        [
            row.get("dataset_split", ""),
            row.get("task_name", ""),
            str(row.get("episode_index", "")),
            row.get("parquet_path", ""),
        ]
    )


def _cache_path(cache_root: Path, row: dict, grid: int) -> Path:
    split = str(row.get("dataset_split", "unknown"))
    task = str(row.get("task_name", "unknown"))
    episode = int(row.get("episode_index", 0))
    suffix = stable_int(_episode_key(row))
    return cache_root / f"grid{grid}" / split / task / f"episode_{episode:06d}_{suffix:x}.pt"


def _needed_frames(rows: Iterable[dict]) -> set[int]:
    frames: set[int] = set()
    for row in rows:
        frames.add(int(row["frame_i"]))
        frames.add(int(row["frame_j"]))
        frames.add(int(row["final_frame"]))
    return frames


def _nearest_latent_index(latent_dict: dict, frame: int) -> int:
    frame_ids = np.asarray(latent_dict["frame_ids"], dtype=np.int64)
    if frame_ids.size == 0:
        return 0
    idx = int(np.abs(frame_ids - int(frame)).argmin())
    return max(0, min(idx, int(latent_dict["latent_num_frames"]) - 1))


def spatial_latent_frame_feature(latent_path: str, frame: int, grid: int) -> torch.Tensor:
    data = load_latent_file(latent_path)
    latent = data["latent"].float()
    num_frames = int(data["latent_num_frames"])
    height = int(data["latent_height"])
    width = int(data["latent_width"])
    latent = latent.reshape(num_frames, height, width, -1)
    idx = _nearest_latent_index(data, frame)
    x = latent[idx].permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, (grid, grid)).squeeze(0)
    return pooled.flatten()


def spatial_state_feature(latents: dict[str, str], frame: int, grid: int) -> torch.Tensor:
    return torch.cat([spatial_latent_frame_feature(latents[cam], frame, grid) for cam in CAM_KEYS], dim=0)


def spatial_state_features(latents: dict[str, str], frames: Iterable[int], grid: int) -> dict[int, torch.Tensor]:
    frame_list = [int(frame) for frame in frames]
    if not frame_list:
        return {}

    per_camera: list[dict[int, torch.Tensor]] = []
    for cam in CAM_KEYS:
        data = load_latent_file(latents[cam])
        latent = data["latent"].float()
        num_frames = int(data["latent_num_frames"])
        height = int(data["latent_height"])
        width = int(data["latent_width"])
        latent = latent.reshape(num_frames, height, width, -1)
        indices = [_nearest_latent_index(data, frame) for frame in frame_list]
        x = latent[indices].permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(x, (grid, grid)).flatten(1)
        per_camera.append({frame: pooled[pos] for pos, frame in enumerate(frame_list)})

    return {
        frame: torch.cat([camera_features[frame] for camera_features in per_camera], dim=0)
        for frame in frame_list
    }


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

    dtype = torch.float16 if args.float16 else torch.float32
    with args.output_jsonl.open("w") as out:
        for episode_rows in tqdm(grouped.values(), desc=f"build RoboTwin spatial grid{args.grid} cache"):
            ref = episode_rows[0]
            path = _cache_path(args.cache_root, ref, args.grid)
            try:
                frames = sorted(_needed_frames(episode_rows))
                if not path.exists() or args.overwrite:
                    states = {frame: feat.to(dtype) for frame, feat in spatial_state_features(ref["latents"], frames, args.grid).items()}
                    payload = {
                        "states": states,
                        "text_emb": text_feature(ref["latents"]).to(dtype),
                        "task_name": ref["task_name"],
                        "episode_index": int(ref["episode_index"]),
                        "dataset_split": ref.get("dataset_split", ""),
                        "grid": args.grid,
                        "state_dim": int(next(iter(states.values())).numel()) if states else 0,
                        "num_frames": len(states),
                    }
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, path)
                else:
                    try:
                        payload = torch.load(path, map_location="cpu", weights_only=False)
                    except Exception:
                        states = {frame: feat.to(dtype) for frame, feat in spatial_state_features(ref["latents"], frames, args.grid).items()}
                        payload = {
                            "states": states,
                            "text_emb": text_feature(ref["latents"]).to(dtype),
                            "task_name": ref["task_name"],
                            "episode_index": int(ref["episode_index"]),
                            "dataset_split": ref.get("dataset_split", ""),
                            "grid": args.grid,
                            "state_dim": int(next(iter(states.values())).numel()) if states else 0,
                            "num_frames": len(states),
                        }
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(payload, path)
                    states = payload.get("states", {})
                    missing_frames = [int(frame) for frame in frames if int(frame) not in states]
                    if missing_frames:
                        missing_states = spatial_state_features(ref["latents"], missing_frames, args.grid)
                        for frame, feat in missing_states.items():
                            states[frame] = feat.to(dtype)
                        payload["states"] = states
                        payload["num_frames"] = len(states)
                        payload["state_dim"] = int(next(iter(states.values())).numel()) if states else int(payload.get("state_dim", 0))
                        torch.save(payload, path)
                for row in episode_rows:
                    cached = dict(row)
                    cached["feature_cache_path"] = str(path)
                    cached["feature_cache_type"] = f"spatial_grid{args.grid}"
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
                        ),
                        flush=True,
                    )

    summary = {
        "input_rows": len(rows),
        "episodes": len(grouped),
        "kept_rows": kept,
        "skipped_episodes": skipped_episodes,
        "skipped_rows": skipped_rows,
        "output_jsonl": str(args.output_jsonl),
        "cache_root": str(args.cache_root),
        "grid": args.grid,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute spatial pooled RoboTwin process critic features.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train.jsonl")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train_spatial_cached.jsonl")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "feature_cache" / "process_spatial")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_cache(parse_args())


if __name__ == "__main__":
    main()
