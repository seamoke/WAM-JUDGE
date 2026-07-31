"""Read RoboTwin metadata without exposing Stage-2 action labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
EPISODE_RE = re.compile(r"episode_(\d+)\.parquet$")
PROPRIO_COLUMNS = (
    "observation.state",
    "observation.proprio",
    "state",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_parquet(repo: Path, episode_index: int) -> Path:
    matches = list(repo.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one parquet for {repo} episode {episode_index}, got {matches}"
        )
    return matches[0]


def episode_metadata(repo: Path) -> dict[int, dict]:
    return {
        int(row["episode_index"]): row
        for row in read_jsonl(repo / "meta" / "episodes.jsonl")
    }


def episode_video_paths(
    repo: Path, parquet: Path, episode_index: int
) -> dict[str, str]:
    chunk = parquet.parent.name
    result = {
        camera: str(
            (
                repo
                / "videos"
                / chunk
                / camera
                / f"episode_{episode_index:06d}.mp4"
            ).resolve()
        )
        for camera in CAMERAS
    }
    missing = [path for path in result.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing camera videos: {missing}")
    return result


def select_proprio_column(parquet: Path) -> str | None:
    import pyarrow.parquet as pq

    columns = set(pq.read_schema(parquet).names)
    for column in PROPRIO_COLUMNS:
        if column in columns:
            return column
    return None


def read_non_action_rows(
    parquet: Path,
    frame_indices: Iterable[int],
    *,
    column: str | None,
) -> list[list[float] | None]:
    """Read only a declared proprio column; the hidden action column is forbidden."""
    indices = [int(index) for index in frame_indices]
    if column is None:
        return [None] * len(indices)
    if column == "action" or column.endswith(".action"):
        raise ValueError("Stage-2 context construction must never read action")
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=[column])
    values = table[column].to_pylist()
    rows = []
    for index in indices:
        if index < 0 or index >= len(values):
            raise IndexError(f"{parquet}: frame {index} outside [0,{len(values)})")
        value = np.asarray(values[index], dtype=np.float32).reshape(-1)
        rows.append(value.tolist())
    return rows


def instruction_from_episode(episode: dict, fallback: str) -> str:
    if episode.get("tasks"):
        return str(episode["tasks"][0])
    configs = episode.get("action_config", [])
    if configs and configs[0].get("action_text"):
        return str(configs[0]["action_text"])
    return fallback


def latent_segment_exists(
    repo: Path, episode_index: int, start: int, end: int
) -> bool:
    pattern = f"episode_{episode_index:06d}_{start}_{end}.pth"
    return all(
        any((repo / "latents").glob(f"chunk-*/{camera}/{pattern}"))
        for camera in CAMERAS
    )
