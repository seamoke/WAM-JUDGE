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
    executable = (latent_frames - 1) * int(config.action_per_frame)
    if actions.shape == (executable, 16):
        actions = np.pad(
            actions,
            ((int(config.action_per_frame), 0), (0, 0)),
            constant_values=0,
        )
    else:
        raise ValueError(
            f"Expected only executable [{executable},16] actions; the generated "
            f"conditioning block must be removed, got {actions.shape}"
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
    # Frame zero is the current-state conditioning block. Inference clamps it
    # to zero in normalized model space, so it is context rather than a target.
    aligned[0] = 0.0
    aligned_mask[0] = False
    return (
        torch.from_numpy(aligned.transpose(2, 0, 1)[..., None]).float(),
        torch.from_numpy(aligned_mask.transpose(2, 0, 1)[..., None]).bool(),
    )


class GeneratedChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selected_jsonl: str | Path,
        config,
        *,
        expected_split_sha256: str,
        expected_selection_mode: str,
    ):
        self.path = Path(selected_jsonl).expanduser().resolve()
        self.rows = read_jsonl(self.path)
        if not self.rows:
            raise ValueError(f"No selected pseudo chunks in {self.path}")
        split_hashes = {
            str(row.get("split_manifest_sha256", "")) for row in self.rows
        }
        if split_hashes != {expected_split_sha256}:
            raise ValueError(
                f"Pseudo data split hashes {split_hashes} do not match "
                f"Stage-1 split {expected_split_sha256}"
            )
        selection_modes = {
            str(row.get("rft_selection", {}).get("mode", ""))
            for row in self.rows
        }
        if selection_modes != {expected_selection_mode}:
            raise ValueError(
                f"Pseudo data selection modes {selection_modes} do not match "
                f"expected mode {expected_selection_mode}"
            )
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
        if int(latents.shape[1]) != int(self.config.frame_chunk_size):
            raise ValueError(
                f"Pseudo sample must contain exactly one model chunk "
                f"F={self.config.frame_chunk_size}, got {latents.shape}"
            )
        text_emb = torch.load(
            row["text_emb_path"], map_location="cpu", weights_only=False
        )
        if text_emb.ndim == 3:
            if text_emb.shape[0] != 1:
                raise ValueError(
                    f"Pseudo text embedding batch must be one: {text_emb.shape}"
                )
            text_emb = text_emb[0]
        if torch.rand(1).item() < float(self.config.cfg_prob):
            text_emb = self.empty_emb
            if text_emb.ndim == 3 and text_emb.shape[0] == 1:
                text_emb = text_emb[0]
        actions, actions_mask = generated_actions_to_tensor(
            np.load(row["action_path"]),
            latent_frames=int(latents.shape[1]),
            config=self.config,
        )
        return {
            "latents": latents.float(),
            "actions": actions,
            "actions_mask": actions_mask,
            "text_emb": text_emb.to(dtype=self.config.param_dtype),
            "latents_mask": torch.ones(latents.shape[1], dtype=torch.bool),
        }


class FirstTransitionChunkDataset(torch.utils.data.Dataset):
    """Project real trajectories onto the same fixed F=2 chunk as pseudo data.

    The official real loader starts every segment with the conditioning frame.
    Keeping its first two latent frames therefore preserves one normal
    conditioning-to-target transition while making samples stackable without
    padding unrelated future frames into an online RFT update.
    """

    def __init__(self, dataset, *, frame_chunk_size: int = 2):
        if frame_chunk_size != 2:
            raise ValueError(
                "Online chunk RFT currently requires frame_chunk_size=2"
            )
        if len(dataset) == 0:
            raise ValueError("Real dataset must be non-empty")
        self.dataset = dataset
        self.frame_chunk_size = frame_chunk_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        frames = int(item["latents"].shape[1])
        if frames < self.frame_chunk_size:
            raise ValueError(
                f"Real sample has F={frames}, expected at least "
                f"{self.frame_chunk_size}"
            )
        for key in ("latents", "actions", "actions_mask"):
            item[key] = item[key][:, : self.frame_chunk_size].contiguous()
        item["latents_mask"] = torch.ones(
            self.frame_chunk_size, dtype=torch.bool
        )
        return item


def mixed_pad_latent_batch_collate(batch: list[dict], base_collate) -> dict:
    """Run the official padded collate while retaining RFT source labels."""
    sources = torch.stack(
        [item["_rft_source"].reshape(()) for item in batch], dim=0
    )
    model_items = [
        {key: value for key, value in item.items() if key != "_rft_source"}
        for item in batch
    ]
    result = base_collate(model_items)
    result["_rft_source"] = sources
    return result


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
            item = dict(self.real_dataset[source_index % len(self.real_dataset)])
            item["_rft_source"] = torch.tensor(0, dtype=torch.int64)
            return item
        source_index = cycle * self.pseudo_slots + offset - self.real_slots
        item = dict(self.pseudo_dataset[source_index % len(self.pseudo_dataset)])
        item["_rft_source"] = torch.tensor(1, dtype=torch.int64)
        return item
