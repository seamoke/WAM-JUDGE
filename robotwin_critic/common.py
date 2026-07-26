from __future__ import annotations

import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

CAM_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

DEFAULT_DATASET_ROOT = Path(
    "/data/lingbot-va/models/datasets/robotwin-clean-and-aug-lerobot/"
    "robotwin-clean-and-aug-lerobot"
)
DEFAULT_OUTPUT_ROOT = Path("/workspace/lingbot-va/train_out/critic/robotwin")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def stable_int(text: str, modulo: int | None = None) -> int:
    value = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)
    return value if modulo is None else value % modulo


def task_name_from_dir(task_dir: Path) -> str:
    return task_dir.name.split("-")[0]


def iter_task_dirs(dataset_root: Path, include_splits: set[str] | None = None) -> Iterable[tuple[str, Path]]:
    for split_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if include_splits and split_dir.name not in include_splits:
            continue
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if (task_dir / "meta" / "episodes.jsonl").is_file() and (
                task_dir / "meta" / "info.json"
            ).is_file():
                yield split_dir.name, task_dir


def find_episode_latents(task_dir: Path, episode_index: int) -> dict[str, str] | None:
    ep_tag = f"episode_{episode_index:06d}_"
    out: dict[str, str] = {}
    for cam in CAM_KEYS:
        matches = sorted((task_dir / "latents").glob(f"chunk-*/{cam}/{ep_tag}*.pth"))
        if not matches:
            return None
        out[cam] = str(matches[0])
    return out


def find_episode_parquet(task_dir: Path, episode_index: int) -> str | None:
    ep_tag = f"episode_{episode_index:06d}.parquet"
    matches = sorted((task_dir / "data").glob(f"chunk-*/{ep_tag}"))
    return str(matches[0]) if matches else None


_LATENT_RE = re.compile(r"episode_(\d{6})_.*\.pth$")
_PARQUET_RE = re.compile(r"episode_(\d{6})\.parquet$")


def build_task_file_maps(task_dir: Path) -> tuple[dict[int, dict[str, str]], dict[int, str]]:
    latent_maps: dict[int, dict[str, str]] = {}
    for cam in CAM_KEYS:
        for path in sorted((task_dir / "latents").glob(f"chunk-*/{cam}/episode_*.pth")):
            match = _LATENT_RE.match(path.name)
            if not match:
                continue
            episode_index = int(match.group(1))
            latent_maps.setdefault(episode_index, {}).setdefault(cam, str(path))

    parquet_map: dict[int, str] = {}
    for path in sorted((task_dir / "data").glob("chunk-*/episode_*.parquet")):
        match = _PARQUET_RE.match(path.name)
        if not match:
            continue
        parquet_map.setdefault(int(match.group(1)), str(path))
    return latent_maps, parquet_map


def split_train_val(rows: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


@lru_cache(maxsize=96)
def load_latent_file(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _nearest_latent_index(latent_dict: dict, frame: int) -> int:
    frame_ids = np.asarray(latent_dict["frame_ids"], dtype=np.int64)
    if frame_ids.size == 0:
        return 0
    idx = int(np.abs(frame_ids - int(frame)).argmin())
    return max(0, min(idx, int(latent_dict["latent_num_frames"]) - 1))


def latent_frame_feature(latent_path: str, frame: int) -> torch.Tensor:
    data = load_latent_file(latent_path)
    latent = data["latent"].float()
    num_frames = int(data["latent_num_frames"])
    height = int(data["latent_height"])
    width = int(data["latent_width"])
    latent = latent.reshape(num_frames, height, width, -1)
    idx = _nearest_latent_index(data, frame)
    x = latent[idx]
    mean = x.mean(dim=(0, 1))
    std = x.std(dim=(0, 1), unbiased=False)
    return torch.cat([mean, std], dim=0)


def state_feature(latents: dict[str, str], frame: int) -> torch.Tensor:
    features = [latent_frame_feature(latents[cam], frame) for cam in CAM_KEYS]
    return torch.cat(features, dim=0)


def text_feature(latents: dict[str, str]) -> torch.Tensor:
    data = load_latent_file(latents[CAM_KEYS[0]])
    text_emb = data.get("text_emb")
    if text_emb is None:
        return torch.zeros(4096, dtype=torch.float32)
    return text_emb.float().mean(dim=0)


@lru_cache(maxsize=128)
def load_action_array(parquet_path: str) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def action_feature(parquet_path: str, start_frame: int, horizon: int) -> torch.Tensor:
    actions = load_action_array(parquet_path)
    if actions.size == 0:
        return torch.zeros(48, dtype=torch.float32)
    start = max(0, min(int(start_frame), len(actions) - 1))
    end = max(start + 1, min(start + int(horizon), len(actions)))
    chunk = torch.from_numpy(actions[start:end]).float()
    mean = chunk.mean(dim=0)
    std = chunk.std(dim=0, unbiased=False)
    delta = chunk[-1] - chunk[0]
    return torch.cat([mean, std, delta], dim=0)


def entry_ref(row: dict) -> dict:
    return {
        "dataset_split": row["dataset_split"],
        "task_dir": row["task_dir"],
        "task_name": row["task_name"],
        "episode_index": row["episode_index"],
        "length": row["length"],
        "text": row["text"],
        "parquet_path": row["parquet_path"],
        "latents": row["latents"],
    }
