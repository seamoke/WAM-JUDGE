#!/usr/bin/env python3
"""Sample RoboTwin LeRobot dataset (aug fraction + clean all, hardlink)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

# aug: keep this fraction per task; clean: keep all episodes
SPLIT_CONFIG = {
    "lerobot_robotwin_eef_aug_500": 0.1,
    "lerobot_robotwin_eef_clean_50": None,
}


def _task_seed(task_name: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{base_seed}:{task_name}".encode()).hexdigest()
    return int(digest[:8], 16)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.samefile(src):
            return
        dst.unlink()
    os.link(src, dst)


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _resolve_n_keep(episode_count: int, keep: float | int | None) -> int | None:
    if keep is None:
        return None
    if isinstance(keep, float):
        return max(1, int(round(episode_count * keep)))
    return int(keep)


def _select_episode_indices(
    episodes: list[dict],
    keep: float | int | None,
    task_name: str,
    seed: int,
) -> list[int]:
    indices = sorted(ep["episode_index"] for ep in episodes)
    n_keep = _resolve_n_keep(len(indices), keep)
    if n_keep is None or n_keep >= len(indices):
        return indices
    rng = np.random.default_rng(_task_seed(task_name, seed))
    chosen = rng.choice(indices, size=n_keep, replace=False)
    return sorted(int(x) for x in chosen)


def _link_episode_files(
    src_task: Path,
    dst_task: Path,
    src_episode_index: int,
    dst_episode_index: int,
    chunks_size: int,
) -> None:
    src_chunk = _episode_chunk(src_episode_index, chunks_size)
    dst_chunk = _episode_chunk(dst_episode_index, chunks_size)
    src_chunk_tag = f"chunk-{src_chunk:03d}"
    dst_chunk_tag = f"chunk-{dst_chunk:03d}"
    src_ep_tag = f"episode_{src_episode_index:06d}"
    dst_ep_tag = f"episode_{dst_episode_index:06d}"

    parquet = src_task / "data" / src_chunk_tag / f"{src_ep_tag}.parquet"
    if parquet.is_file():
        _hardlink(parquet, dst_task / "data" / dst_chunk_tag / f"{dst_ep_tag}.parquet")

    for cam in VIDEO_KEYS:
        video = src_task / "videos" / src_chunk_tag / cam / f"{src_ep_tag}.mp4"
        if video.is_file():
            _hardlink(video, dst_task / "videos" / dst_chunk_tag / cam / f"{dst_ep_tag}.mp4")

        latent_dir = src_task / "latents" / src_chunk_tag / cam
        if latent_dir.is_dir():
            for latent in latent_dir.glob(f"{src_ep_tag}_*.pth"):
                dst_name = latent.name.replace(src_ep_tag, dst_ep_tag, 1)
                _hardlink(
                    latent,
                    dst_task / "latents" / dst_chunk_tag / cam / dst_name,
                )


def _process_task(
    src_task: Path,
    dst_task: Path,
    keep: float | int | None,
    seed: int,
) -> dict:
    meta_dir = src_task / "meta"
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    chunks_size = int(info.get("chunks_size", 1000))

    episodes = _load_jsonl(meta_dir / "episodes.jsonl")
    selected = _select_episode_indices(episodes, keep, src_task.name, seed)

    ep_by_idx = {ep["episode_index"]: ep for ep in episodes}
    kept_episodes = []
    stats_rows = []
    ori_rows = []

    stats_by_idx = {}
    stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.is_file():
        for row in _load_jsonl(stats_path):
            stats_by_idx[row["episode_index"]] = row

    ori_by_idx = {}
    ori_path = meta_dir / "episodes_ori.jsonl"
    if ori_path.is_file():
        for row in _load_jsonl(ori_path):
            ori_by_idx[row["episode_index"]] = row

    # LeRobot expects contiguous episode files: episode_000000 .. episode_{N-1}
    for new_idx, old_idx in enumerate(selected):
        ep = dict(ep_by_idx[old_idx])
        ep["episode_index"] = new_idx
        kept_episodes.append(ep)

        if old_idx in stats_by_idx:
            row = dict(stats_by_idx[old_idx])
            row["episode_index"] = new_idx
            stats_rows.append(row)

        if old_idx in ori_by_idx:
            row = dict(ori_by_idx[old_idx])
            row["episode_index"] = new_idx
            ori_rows.append(row)

    dst_meta = dst_task / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    _write_jsonl(dst_meta / "episodes.jsonl", kept_episodes)
    if stats_rows:
        _write_jsonl(dst_meta / "episodes_stats.jsonl", stats_rows)
    if ori_rows:
        _write_jsonl(dst_meta / "episodes_ori.jsonl", ori_rows)

    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.is_file():
        _hardlink(tasks_path, dst_meta / "tasks.jsonl")

    total_frames = sum(ep.get("length", 0) for ep in kept_episodes)
    n_eps = len(kept_episodes)
    info["total_episodes"] = n_eps
    info["total_frames"] = total_frames
    info["total_videos"] = n_eps * len(VIDEO_KEYS)
    info["total_chunks"] = max(
        1, (_episode_chunk(len(kept_episodes) - 1, chunks_size) + 1) if kept_episodes else 1
    )
    info["splits"] = {"train": f"0:{n_eps}"}
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    for new_idx, old_idx in enumerate(selected):
        _link_episode_files(src_task, dst_task, old_idx, new_idx, chunks_size)

    return {
        "task": src_task.name,
        "source_episodes": len(episodes),
        "kept_episodes": n_eps,
        "kept_frames": total_frames,
        "selected_indices": selected,
        "index_map": {str(old): new for new, old in enumerate(selected)},
    }


def sample_dataset(
    src_root: Path,
    dst_root: Path,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    manifest: dict = {
        "scheme": "aug_fraction+clean_all",
        "aug_fraction": SPLIT_CONFIG["lerobot_robotwin_eef_aug_500"],
        "seed": seed,
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "splits": {},
        "tasks": [],
    }

    if dry_run:
        for split_name, keep in SPLIT_CONFIG.items():
            split_dir = src_root / split_name
            for task_dir in sorted(split_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                episodes = _load_jsonl(task_dir / "meta" / "episodes.jsonl")
                selected = _select_episode_indices(
                    episodes, keep, task_dir.name, seed
                )
                manifest["tasks"].append(
                    {
                        "split": split_name,
                        "task": task_dir.name,
                        "source_episodes": len(episodes),
                        "kept_episodes": len(selected),
                    }
                )
        manifest["total_episodes"] = sum(t["kept_episodes"] for t in manifest["tasks"])
        return manifest

    dst_root.mkdir(parents=True, exist_ok=True)

    empty_emb = src_root / "empty_emb.pt"
    if empty_emb.is_file():
        _hardlink(empty_emb, dst_root / "empty_emb.pt")

    readme = src_root.parent / "README.md"
    if readme.is_file():
        _hardlink(readme, dst_root / "README.md")

    for split_name, keep in SPLIT_CONFIG.items():
        src_split = src_root / split_name
        dst_split = dst_root / split_name
        dst_split.mkdir(parents=True, exist_ok=True)

        split_stats = {"tasks": 0, "episodes": 0, "frames": 0}
        task_dirs = sorted(p for p in src_split.iterdir() if p.is_dir())
        for task_dir in tqdm(task_dirs, desc=split_name):
            result = _process_task(task_dir, dst_split / task_dir.name, keep, seed)
            result["split"] = split_name
            manifest["tasks"].append(result)
            split_stats["tasks"] += 1
            split_stats["episodes"] += result["kept_episodes"]
            split_stats["frames"] += result["kept_frames"]
        manifest["splits"][split_name] = split_stats

    manifest["total_episodes"] = sum(
        t["kept_episodes"] for t in manifest["tasks"]
    )
    manifest["total_frames"] = sum(t["kept_frames"] for t in manifest["tasks"])

    manifest_path = dst_root / "sample_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/data/lingbot-va/models/datasets/robotwin-clean-and-aug-lerobot"
            "/robotwin-clean-and-aug-lerobot"
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/data/lingbot-va/models/datasets/robotwin-clean-and-aug-lerobot"
            "/robotwin-sample-b"
        ),
    )
    parser.add_argument("--aug-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    SPLIT_CONFIG["lerobot_robotwin_eef_aug_500"] = args.aug_fraction

    manifest = sample_dataset(args.src, args.dst, seed=args.seed, dry_run=args.dry_run)
    print(json.dumps(
        {
            "dst": str(args.dst),
            "total_episodes": manifest["total_episodes"],
            "total_frames": manifest.get("total_frames"),
            "splits": manifest.get("splits", {}),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
