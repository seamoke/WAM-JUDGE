"""Sidecar pseudo-chunk loader and deterministic 70/30 real/pseudo mixture."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from robotwin_critic.two_stage_rft.pseudo_provenance import (
    validate_pseudo_split_provenance,
)
from robotwin_critic.two_stage_rft.pseudo_action_contract import (
    validate_pseudo_action_contract,
    verify_legacy_pseudo_action_waiver,
)
from robotwin_critic.two_stage_rft.pseudo_artifact_contract import validate_pseudo_artifact_rows


def read_jsonl(path: Path) -> tuple[list[dict], list[int]]:
    import json

    rows, row_numbers = [], []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON in {path} at physical line {line_number}: {error.msg}"
                ) from error
            row_numbers.append(line_number)
    return rows, row_numbers


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
    action_mask = np.ones_like(actions, dtype=bool)
    padded_mask = np.pad(action_mask, ((0, 0), (0, 1)), constant_values=False)
    aligned_mask = padded_mask[:, config.inverse_used_action_channel_ids]
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
    def __init__(
        self,
        selected_jsonl: str | Path,
        config,
        *,
        expected_split_sha256: str,
        expected_selection_mode: str,
        split_manifest_path: str | Path | None = None,
        legacy_pseudo_action_waiver_sha256: str | None = None,
        legacy_pseudo_action_waiver_rows: int | None = None,
    ):
        self.path = Path(selected_jsonl).expanduser().resolve()
        legacy_waiver = verify_legacy_pseudo_action_waiver(
            self.path,
            expected_sha256=legacy_pseudo_action_waiver_sha256,
            expected_rows=legacy_pseudo_action_waiver_rows,
        )
        self.rows, row_numbers = read_jsonl(self.path)
        if not self.rows:
            raise ValueError(f"No selected pseudo chunks in {self.path}")
        validate_pseudo_action_contract(
            self.rows,
            expected_latent_frames=config.frame_chunk_size,
            action_per_frame=config.action_per_frame,
            row_numbers=row_numbers,
            allow_legacy_pseudo_action_metadata=legacy_waiver,
        )
        validate_pseudo_artifact_rows(
            self.rows, jsonl_parent=self.path.parent,
            row_numbers=row_numbers,
        )
        self.provenance_report = validate_pseudo_split_provenance(
            self.rows,
            expected_split_sha256=expected_split_sha256,
            split_manifest_path=split_manifest_path,
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
        for row_number, row in zip(row_numbers, self.rows):
            for key in ("latent_path", "text_emb_path", "action_path"):
                if key not in row:
                    raise ValueError(
                        f"Selected pseudo row {row_number} is missing {key!r}"
                    )
                path = Path(row[key]).expanduser()
                if not path.is_absolute():
                    path = self.path.parent / path
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Selected pseudo row {row_number} {key} does not exist: "
                        f"{path}"
                    )
                row[key] = str(path)
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
            "latents": latents.to(dtype=self.config.param_dtype),
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


class AllTransitionChunkDataset(torch.utils.data.Dataset):
    """Expose every adjacent real latent-frame transition as one F=2 sample.

    The official loader stores a complete action segment in one latent file.
    This view indexes every adjacent frame pair without copying those files. A
    small per-worker LRU keeps consecutive transitions from reloading the same
    source segment.
    """

    def __init__(
        self,
        dataset,
        *,
        frame_chunk_size: int = 2,
        cache_segments: int = 4,
        transition_index: list[tuple[int, int]] | None = None,
    ):
        if frame_chunk_size != 2:
            raise ValueError("Transition chunk training requires frame_chunk_size=2")
        if len(dataset) == 0:
            raise ValueError("Real dataset must be non-empty")
        self.dataset = dataset
        self.frame_chunk_size = frame_chunk_size
        self.cache_segments = max(int(cache_segments), 0)
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._can_rebase_actions = transition_index is None
        self.transition_index = (
            list(transition_index)
            if transition_index is not None
            else self._build_transition_index()
        )
        if not self.transition_index:
            raise ValueError("Real dataset contains no adjacent latent transitions")

    def _build_transition_index(self) -> list[tuple[int, int]]:
        result = []
        for dataset_id, source in enumerate(self.dataset._datasets):
            global_offset = int(self.dataset.acc_dset_num[dataset_id])
            camera = source.used_video_keys[0]
            for local_index, meta in enumerate(source.new_metas):
                episode_index = int(meta["episode_index"])
                start = int(meta["start_frame"])
                end = int(meta["end_frame"])
                episode_chunk = source.meta.get_episode_chunk(episode_index)
                latent_file = (
                    Path(source.latent_path)
                    / f"chunk-{episode_chunk:03d}"
                    / camera
                    / f"episode_{episode_index:06d}_{start}_{end}.pth"
                )
                try:
                    payload = torch.load(
                        latent_file,
                        map_location="cpu",
                        weights_only=False,
                        mmap=True,
                    )
                except TypeError:
                    payload = torch.load(
                        latent_file, map_location="cpu", weights_only=False
                    )
                latent_frames = int(payload["latent_num_frames"])
                result.extend(
                    (global_offset + local_index, offset)
                    for offset in range(max(latent_frames - 1, 0))
                )
        return result

    def __len__(self) -> int:
        return len(self.transition_index)

    def _source_record(self, source_index: int) -> dict:
        if source_index in self._cache:
            record = self._cache.pop(source_index)
            self._cache[source_index] = record
            return record
        record = {"item": self.dataset[source_index]}
        if self._can_rebase_actions:
            dataset_id = int(self.dataset.item_id_to_dataset_id[source_index])
            source = self.dataset._datasets[dataset_id]
            local_index = source_index - int(self.dataset.acc_dset_num[dataset_id])
            meta = source.new_metas[local_index]
            episode_index = int(meta["episode_index"])
            start = int(meta["start_frame"])
            end = int(meta["end_frame"])
            episode_chunk = source.meta.get_episode_chunk(episode_index)
            latent_file = (
                Path(source.latent_path)
                / f"chunk-{episode_chunk:03d}"
                / source.used_video_keys[0]
                / f"episode_{episode_index:06d}_{start}_{end}.pth"
            )
            latent_payload = torch.load(
                latent_file, map_location="cpu", weights_only=False
            )
            global_start = source._get_global_idx(episode_index, start)
            global_end = source._get_global_idx(episode_index, end)
            raw_actions = source._get_range_hf_data(global_start, global_end)["action"]
            record.update(
                {
                    "source": source,
                    "meta": meta,
                    "frame_ids": latent_payload["frame_ids"],
                    "latent_frames": int(latent_payload["latent_num_frames"]),
                    "raw_actions": raw_actions,
                }
            )
        if self.cache_segments:
            self._cache[source_index] = record
            while len(self._cache) > self.cache_segments:
                self._cache.popitem(last=False)
        return record

    @staticmethod
    def _rebased_actions(record: dict, offset: int):
        source = record["source"]
        meta = record["meta"]
        frame_ids = record["frame_ids"]
        latent_frames = int(record["latent_frames"])
        if latent_frames < 2:
            raise ValueError("Cannot extract a transition from fewer than two frames")
        compressed_steps = (len(frame_ids) - 1) // (latent_frames - 1)
        if compressed_steps <= 0 or (
            (len(frame_ids) - 1) % (latent_frames - 1)
        ):
            raise ValueError(
                f"Unexpected frame-id geometry: ids={len(frame_ids)} F={latent_frames}"
            )
        begin = offset * compressed_steps
        stop = (offset + 1) * compressed_steps + 1
        pair_frame_ids = frame_ids[begin:stop]
        actions, mask = source._action_post_process(
            int(meta["start_frame"]),
            int(meta["end_frame"]),
            pair_frame_ids,
            record["raw_actions"],
        )
        return actions, mask

    def __getitem__(self, index: int) -> dict:
        source_index, offset = self.transition_index[index % len(self.transition_index)]
        record = self._source_record(source_index)
        source = record["item"]
        stop = offset + self.frame_chunk_size
        frames = int(source["latents"].shape[1])
        if stop > frames:
            raise IndexError(
                f"Transition [{offset},{stop}) exceeds source F={frames}"
            )
        item = dict(source)
        item["latents"] = source["latents"][:, offset:stop].clone()
        if self._can_rebase_actions:
            item["actions"], item["actions_mask"] = self._rebased_actions(
                record, offset
            )
        else:
            for key in ("actions", "actions_mask"):
                item[key] = source[key][:, offset:stop].clone()
        # Every extracted pair starts with a fresh conditioning frame.
        item["actions"][:, 0] = 0.0
        item["actions_mask"][:, 0] = False
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


class UnionRFTDataset(torch.utils.data.Dataset):
    """Concatenate real and pseudo samples so each appears once per epoch."""

    def __init__(self, real_dataset, pseudo_dataset):
        if len(real_dataset) == 0 or len(pseudo_dataset) == 0:
            raise ValueError("Real and pseudo datasets must both be non-empty")
        self.real_dataset = real_dataset
        self.pseudo_dataset = pseudo_dataset

    def __len__(self) -> int:
        return len(self.real_dataset) + len(self.pseudo_dataset)

    def source_for_index(self, index: int) -> str:
        return "real" if index < len(self.real_dataset) else "pseudo"

    def __getitem__(self, index: int) -> dict:
        index %= len(self)
        if index < len(self.real_dataset):
            item = dict(self.real_dataset[index])
            item["_rft_source"] = torch.tensor(0, dtype=torch.int64)
            return item
        item = dict(self.pseudo_dataset[index - len(self.real_dataset)])
        item["_rft_source"] = torch.tensor(1, dtype=torch.int64)
        return item


class DeterministicFractionDataset(torch.utils.data.Dataset):
    """Select a reproducible fraction without changing the source dataset."""

    def __init__(self, dataset, *, fraction: float, seed: int = 42):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0,1]")
        if len(dataset) == 0:
            raise ValueError("Dataset must be non-empty")
        self.dataset = dataset
        self.fraction = float(fraction)
        self.seed = int(seed)
        selected = max(1, int(math.floor(len(dataset) * fraction + 0.5)))
        if selected >= len(dataset):
            self.indices = list(range(len(dataset)))
        else:
            generator = np.random.default_rng(self.seed)
            self.indices = sorted(
                int(index)
                for index in generator.choice(
                    len(dataset), size=selected, replace=False
                ).tolist()
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index % len(self.indices)]]
