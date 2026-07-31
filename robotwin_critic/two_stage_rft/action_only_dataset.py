"""Sidecar pseudo-chunk loader and deterministic 70/30 real/pseudo mixture."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generated_actions_to_tensor(
    actions: np.ndarray,
    *,
    latent_frames: int,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = np.asarray(actions, dtype=np.float32)
    expected = latent_frames * int(config.action_per_frame)
    executable = (latent_frames - 1) * int(config.action_per_frame)
    if actions.shape == (executable, 16):
        actions = np.pad(
            actions,
            ((int(config.action_per_frame), 0), (0, 0)),
            constant_values=0,
        )
    elif actions.shape != (expected, 16):
        raise ValueError(
            f"Expected executable [{executable},16] or packed [{expected},16], "
            f"got {actions.shape}"
        )
    padded = np.pad(actions, ((0, 0), (0, 1)), constant_values=0)
    aligned = padded[:, config.inverse_used_action_channel_ids]
    mask = np.ones_like(padded, dtype=bool)
    mask[:, -1] = False
    aligned_mask = mask[:, config.inverse_used_action_channel_ids]
    q01 = np.asarray(config.norm_stat["q01"], dtype=np.float32)[None]
    q99 = np.asarray(config.norm_stat["q99"], dtype=np.float32)[None]
    aligned = (aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    aligned = np.clip(aligned, -1.5, 1.5)
    aligned *= aligned_mask
    action_per_frame = int(config.action_per_frame)
    aligned = aligned.reshape(latent_frames, action_per_frame, -1)
    aligned_mask = aligned_mask.reshape(latent_frames, action_per_frame, -1)
    return (
        torch.from_numpy(aligned.transpose(2, 0, 1)[..., None]).float(),
        torch.from_numpy(aligned_mask.transpose(2, 0, 1)[..., None]).bool(),
    )


class GeneratedChunkDataset(torch.utils.data.Dataset):
    def __init__(self, selected_jsonl: str | Path, config):
        self.path = Path(selected_jsonl).expanduser().resolve()
        self.rows = read_jsonl(self.path)
        if not self.rows:
            raise ValueError(f"No selected pseudo chunks in {self.path}")
        self.config = config
        self.empty_emb = torch.load(
            config.empty_emb_path, map_location="cpu", weights_only=False
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index % len(self.rows)]
        latents = torch.load(
            row["latent_path"], map_location="cpu", weights_only=False
        )
        if isinstance(latents, dict):
            latents = latents["latents"]
        if latents.ndim == 5:
            if latents.shape[0] != 1:
                raise ValueError(f"Pseudo latent batch must be one: {latents.shape}")
            latents = latents[0]
        if latents.ndim != 4:
            raise ValueError(f"Pseudo latents must be [C,F,H,W], got {latents.shape}")
        text_emb = torch.load(
            row["text_emb_path"], map_location="cpu", weights_only=False
        )
        if text_emb.ndim == 3:
            if text_emb.shape[0] != 1:
                raise ValueError(
                    f"Pseudo text embedding batch must be one: {text_emb.shape}"
                )
            text_emb = text_emb[0]
        if torch.rand(1).item() < float(config.cfg_prob):
            text_emb = self.empty_emb
            if text_emb.ndim == 3 and text_emb.shape[0] == 1:
                text_emb = text_emb[0]
        actions, actions_mask = generated_actions_to_tensor(
            np.load(row["action_path"]),
            latent_frames=int(latents.shape[1]),
            config=config,
        )
        return {
            "latents": latents.float(),
            "actions": actions,
            "actions_mask": actions_mask,
            "text_emb": text_emb.float(),
        }


class RatioMixedDataset(torch.utils.data.Dataset):
    """Expose a fixed-ratio index view while preserving both source datasets."""

    def __init__(
        self,
        real_dataset,
        pseudo_dataset,
        *,
        real_fraction: float = 0.7,
        cycle_size: int = 10,
    ):
        if not 0.0 < real_fraction < 1.0:
            raise ValueError("real_fraction must be in (0,1)")
        if len(real_dataset) == 0 or len(pseudo_dataset) == 0:
            raise ValueError("Real and pseudo datasets must both be non-empty")
        real_slots = round(real_fraction * cycle_size)
        if not 0 < real_slots < cycle_size:
            raise ValueError("cycle_size cannot represent the requested ratio")
        self.real_dataset = real_dataset
        self.pseudo_dataset = pseudo_dataset
        self.real_slots = real_slots
        self.pseudo_slots = cycle_size - real_slots
        self.cycle_size = cycle_size
        cycles = max(
            math.ceil(len(real_dataset) / real_slots),
            math.ceil(len(pseudo_dataset) / self.pseudo_slots),
        )
        self.length = cycles * cycle_size

    def __len__(self) -> int:
        return self.length

    def source_for_index(self, index: int) -> str:
        return "real" if index % self.cycle_size < self.real_slots else "pseudo"

    def __getitem__(self, index: int) -> dict:
        cycle, offset = divmod(index % self.length, self.cycle_size)
        if offset < self.real_slots:
            source_index = cycle * self.real_slots + offset
            return self.real_dataset[source_index % len(self.real_dataset)]
        source_index = cycle * self.pseudo_slots + offset - self.real_slots
        return self.pseudo_dataset[source_index % len(self.pseudo_dataset)]
