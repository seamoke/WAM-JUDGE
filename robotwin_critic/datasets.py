from __future__ import annotations

from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset

from robotwin_critic.common import action_feature, read_jsonl, stable_int, state_feature, text_feature


def _task_id(task_name: str, buckets: int) -> int:
    return stable_int(task_name, buckets)


class _RetryBadRowsMixin:
    max_retries = 64

    def _warn_bad_row(self, idx: int, row: dict, exc: Exception) -> None:
        warnings.warn(
            "Skipping bad RoboTwin critic row "
            f"idx={idx} task={row.get('task_name')} episode={row.get('episode_index')}: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _load_with_retry(self, idx: int, build_item):
        last_error: Exception | None = None
        if not self.rows:
            raise IndexError("empty RoboTwin critic dataset")
        for offset in range(min(self.max_retries, len(self.rows))):
            row_idx = (idx + offset) % len(self.rows)
            row = self.rows[row_idx]
            try:
                return build_item(row)
            except Exception as exc:
                last_error = exc
                if offset < 3:
                    self._warn_bad_row(row_idx, row, exc)
        raise RuntimeError(
            f"failed to load a valid RoboTwin critic row after {self.max_retries} retries"
        ) from last_error


class ProcessPairDataset(_RetryBadRowsMixin, Dataset):
    def __init__(self, jsonl_path: str | Path, task_buckets: int = 4096):
        self.rows = read_jsonl(Path(jsonl_path))
        self.task_buckets = task_buckets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self._load_with_retry(idx, self._build_item)

    def _build_item(self, row: dict) -> dict:
        latents = row["latents"]
        return {
            "state_i": state_feature(latents, row["frame_i"]),
            "state_j": state_feature(latents, row["frame_j"]),
            "state_final": state_feature(latents, row["final_frame"]),
            "text_emb": text_feature(latents),
            "task_id": torch.tensor(_task_id(row["task_name"], self.task_buckets), dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "frame_i": torch.tensor(int(row["frame_i"]), dtype=torch.float32),
            "frame_j": torch.tensor(int(row["frame_j"]), dtype=torch.float32),
            "task_name": row["task_name"],
        }


class ConsistencyPairDataset(_RetryBadRowsMixin, Dataset):
    def __init__(self, jsonl_path: str | Path, task_buckets: int = 4096):
        self.rows = read_jsonl(Path(jsonl_path))
        self.task_buckets = task_buckets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self._load_with_retry(idx, self._build_item)

    def _build_item(self, row: dict) -> dict:
        future_ref = row.get("future_ref", row)
        action_ref = row.get("action_ref", row)
        return {
            "state": state_feature(row["latents"], row["state_frame"]),
            "future": state_feature(future_ref["latents"], row["future_frame"]),
            "action": action_feature(action_ref["parquet_path"], row["action_frame"], row["horizon"]),
            "text_emb": text_feature(row["latents"]),
            "task_id": torch.tensor(_task_id(row["task_name"], self.task_buckets), dtype=torch.long),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "negative_type": row.get("negative_type", "unknown"),
            "task_name": row["task_name"],
        }
