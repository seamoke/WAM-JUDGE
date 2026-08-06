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
EEF_STATE_NAMES = (
    "left_x",
    "left_y",
    "left_z",
    "left_q1",
    "left_q2",
    "left_q3",
    "left_q4",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_q1",
    "right_q2",
    "right_q3",
    "right_q4",
    "right_gripper",
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


def verified_eef_state_indices(repo: Path, column: str | None) -> tuple[int, ...]:
    """Return the EEF mapping only when LeRobot metadata names it exactly."""
    if column is None:
        return ()
    info = json.loads((repo / "meta" / "info.json").read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(column, {})
    names = feature.get("names")
    if len(names or []) == 1 and isinstance(names[0], list):
        names = names[0]
    if tuple(names or ()) == EEF_STATE_NAMES and list(feature.get("shape", [])) == [16]:
        return tuple(range(16))
    return ()


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


def latent_segment_num_frames(
    repo: Path, episode_index: int, start: int, end: int
) -> int:
    """Read one camera's latent metadata without touching the hidden action column."""
    pattern = f"episode_{episode_index:06d}_{start}_{end}.pth"
    matches = list((repo / "latents").glob(f"chunk-*/{CAMERAS[0]}/{pattern}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one latent segment for {repo} episode={episode_index} "
            f"range=[{start},{end}), got {matches}"
        )
    import torch

    try:
        payload = torch.load(
            matches[0], map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(matches[0], map_location="cpu", weights_only=False)
    frames = int(payload["latent_num_frames"])
    if frames <= 0:
        raise ValueError(f"Invalid latent_num_frames={frames} in {matches[0]}")
    return frames
