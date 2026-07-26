"""Validate VLAC pair manifests and render a small RGB contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .common import parse_score, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args()


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def episode_identity(row: dict) -> tuple[str, str, int]:
    metadata = row.get("metadata") or {}
    return (
        str(metadata.get("dataset_split", "unknown_split")),
        str(metadata.get("task_dir", metadata.get("task", "unknown_task"))),
        int(metadata["episode_index"]),
    )


def validate_ordered_pairs(rows: list[dict], split: str) -> int:
    if len(rows) % 2:
        raise RuntimeError(f"{split} manifest contains an unpaired final row")
    pair_groups = 0
    for offset in range(0, len(rows), 2):
        forward, reverse = rows[offset : offset + 2]
        forward_meta = forward.get("metadata") or {}
        reverse_meta = reverse.get("metadata") or {}
        if episode_identity(forward) != episode_identity(reverse):
            raise RuntimeError(f"{split} row {offset} reverse pair crosses episodes")
        if forward_meta.get("pair_kind") != reverse_meta.get("pair_kind"):
            raise RuntimeError(f"{split} row {offset} reverse pair kind differs")
        if (
            int(forward_meta["i"]) != int(reverse_meta["j"])
            or int(forward_meta["j"]) != int(reverse_meta["i"])
        ):
            raise RuntimeError(f"{split} row {offset} frame indices are not reversed")
        if forward["images"] != list(reversed(reverse["images"])):
            raise RuntimeError(f"{split} row {offset} image order is not reversed")
        forward_target = float(forward_meta["target"])
        reverse_target = float(reverse_meta["target"])
        if not np.isclose(forward_target, -reverse_target, atol=1e-6):
            raise RuntimeError(f"{split} row {offset} targets are not antisymmetric")
        pair_groups += 1
    return pair_groups


def validate(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    train = read_jsonl(data_dir / "train.jsonl")
    val = read_jsonl(data_dir / "val.jsonl")
    trainer_val_path = data_dir / "val_train.jsonl"
    trainer_val = read_jsonl(trainer_val_path) if trainer_val_path.exists() else val
    trajectories = read_jsonl(data_dir / "val_trajectories.jsonl")
    state_score_train_path = data_dir / "state_score_train.jsonl"
    state_score_val_path = data_dir / "state_score_val.jsonl"
    state_score_train = read_jsonl(state_score_train_path) if state_score_train_path.exists() else []
    state_score_val = read_jsonl(state_score_val_path) if state_score_val_path.exists() else []
    if not train or not val or not trajectories:
        raise RuntimeError("Train, val, and trajectory manifests must all be non-empty")
    train_episode_ids = {episode_identity(row) for row in train}
    val_episode_ids = {episode_identity(row) for row in val}
    overlap = train_episode_ids.intersection(val_episode_ids)
    if overlap:
        raise RuntimeError(
            f"Episode-level leakage between train and val: {sorted(overlap)[:5]}"
        )
    train_pair_groups = validate_ordered_pairs(train, "train")
    val_pair_groups = validate_ordered_pairs(val, "val")
    trainer_val_pair_groups = validate_ordered_pairs(trainer_val, "trainer val")
    trainer_val_episode_ids = {episode_identity(row) for row in trainer_val}
    if not trainer_val_episode_ids.issubset(val_episode_ids):
        raise RuntimeError("Trainer validation subset contains a non-validation episode")

    selected = (train + val)[: args.samples]
    panels = []
    labels = []
    image_shapes = set()
    channel_stds = []
    for row in selected:
        if len(row.get("images", [])) != 2:
            raise ValueError("Every pair must contain exactly two state images")
        target = parse_score(row["messages"][-1]["content"])
        if not np.isfinite(target):
            raise ValueError("Target is not finite")
        first, second = [load_rgb(path) for path in row["images"]]
        if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
            raise ValueError(f"Invalid RGB pair shapes: {first.shape}, {second.shape}")
        image_shapes.add(first.shape)
        channel_stds.extend(np.std(image.reshape(-1, 3), axis=0).tolist() for image in (first, second))
        panel = np.concatenate([first, second], axis=1)
        cv2.putText(
            panel,
            f"target {target:+.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        panels.append(panel)
        labels.append(target)

    if min(np.asarray(channel_stds).reshape(-1)) < 1.0:
        raise RuntimeError("At least one RGB channel appears constant; inspect camera decoding")
    if len(image_shapes) != 1:
        raise RuntimeError(f"Inconsistent image shapes: {sorted(image_shapes)}")

    output = Path(args.output or data_dir / "rgb_validation.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    width = max(panel.shape[1] for panel in panels)
    resized = [
        cv2.resize(panel, (width, int(panel.shape[0] * width / panel.shape[1])))
        for panel in panels
    ]
    sheet = np.concatenate(resized, axis=0)
    if not cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {output}")

    summary = {
        "train_samples": len(train),
        "val_samples": len(val),
        "train_episodes": len(train_episode_ids),
        "val_episodes": len(val_episode_ids),
        "episode_overlap": 0,
        "train_pair_groups": train_pair_groups,
        "val_pair_groups": val_pair_groups,
        "trainer_val_samples": len(trainer_val),
        "trainer_val_episodes": len(trainer_val_episode_ids),
        "trainer_val_pair_groups": trainer_val_pair_groups,
        "trainer_val_tasks": len(
            {str((row.get("metadata") or {}).get("task")) for row in trainer_val}
        ),
        "all_pairs_antisymmetric": True,
        "val_trajectories": len(trajectories),
        "image_shape": list(next(iter(image_shapes))),
        "targets_min": float(min(labels)),
        "targets_max": float(max(labels)),
        "targets_have_positive": any(value > 0 for value in labels),
        "targets_have_negative": any(value < 0 for value in labels),
        "contact_sheet": str(output),
    }
    if not summary["targets_have_positive"] or not summary["targets_have_negative"]:
        raise RuntimeError("Validation subset must contain both forward and reverse labels")

    if bool(state_score_train) != bool(state_score_val):
        raise RuntimeError("State-score train and val manifests must be created together")
    if state_score_train:
        state_labels = set()
        for row in state_score_train + state_score_val:
            required = {"state_i", "state_j", "goal", "target_delta", "label"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"State-score row is missing fields: {sorted(missing)}")
            if int(row["label"]) not in {-1, 0, 1}:
                raise ValueError(f"Invalid state-score label: {row['label']}")
            if not np.isfinite(float(row["target_delta"])):
                raise ValueError("State-score target_delta is not finite")
            state_labels.add(int(row["label"]))
        state_image_samples = (state_score_train + state_score_val)[: args.samples]
        for row in state_image_samples:
            images = [load_rgb(row[key]) for key in ("state_i", "state_j", "goal")]
            if len({image.shape for image in images}) != 1:
                raise ValueError("State-score current, next, and goal image shapes differ")
        if not {-1, 1}.issubset(state_labels):
            raise RuntimeError("State-score manifests must include forward and reverse pairs")
        summary["state_score_train_samples"] = len(state_score_train)
        summary["state_score_val_samples"] = len(state_score_val)
        summary["state_score_labels"] = sorted(state_labels)
        summary["state_score_images_checked"] = len(state_image_samples)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    validate(parse_args())


if __name__ == "__main__":
    main()
