"""Materialize selected WAM chunks as a sidecar LeRobot training dataset.

Each accepted record becomes one short trajectory segment. The output follows
the same metadata, latent, parquet, and action_config contract consumed by
``LatentLeRobotDataset``. No original RoboTwin parquet or WAM source file is
modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
SCHEMA_VERSION = 1


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_actions(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        value = np.load(path)
    elif path.suffix == ".pt":
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            value = value["actions"]
        if torch.is_tensor(value):
            value = value.cpu().numpy()
    else:
        raise ValueError(f"Action file must be .npy or .pt: {path}")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 16:
        raise ValueError(f"Expected generated actions [T,16], got {value.shape}")
    return value


def load_and_validate_latents(
    paths: dict[str, str], start_frame: int, end_frame: int
) -> tuple[dict[str, dict], np.ndarray]:
    loaded = {}
    reference_ids = None
    for camera in CAMERAS:
        if camera not in paths:
            raise KeyError(f"Missing generated latent for {camera}")
        path = Path(paths[camera])
        value = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "latent",
            "latent_num_frames",
            "latent_height",
            "latent_width",
            "frame_ids",
            "text_emb",
        }
        missing = required.difference(value)
        if missing:
            raise KeyError(f"{path}: missing latent keys {sorted(missing)}")
        frame_ids = np.asarray(value["frame_ids"], dtype=np.int64)
        if len(frame_ids) < 2 or np.any(np.diff(frame_ids) <= 0):
            raise ValueError(f"{path}: frame_ids must be strictly increasing")
        packed_frames = (len(frame_ids) - 1) // 4 + 1
        if int(value["latent_num_frames"]) != packed_frames:
            raise ValueError(
                f"{path}: latent_num_frames={value['latent_num_frames']} does "
                f"not match temporal packing result {packed_frames}"
            )
        if frame_ids[0] < start_frame or frame_ids[-1] >= end_frame:
            raise ValueError(
                f"{path}: frame_ids [{frame_ids[0]}, {frame_ids[-1]}] fall "
                f"outside candidate range [{start_frame}, {end_frame})"
            )
        if reference_ids is not None and not np.array_equal(reference_ids, frame_ids):
            raise ValueError("The three generated camera latents use different frame_ids")
        reference_ids = frame_ids
        loaded[camera] = value
    assert reference_ids is not None
    return loaded, reference_ids


def validate_loader_alignment(
    actions: np.ndarray, frame_ids: np.ndarray, start_frame: int
) -> dict:
    act_shift = int(frame_ids[0] - start_frame)
    stride = int(frame_ids[1] - frame_ids[0])
    latent_frames = (len(frame_ids) - 1) // 4 + 1
    required_actions = latent_frames * stride * 4
    available_after_loader_padding = len(actions) - act_shift + stride * 4
    if available_after_loader_padding < required_actions:
        raise ValueError(
            "Generated chunk is too short for LatentLeRobotDataset: "
            f"need {required_actions}, have {available_after_loader_padding} "
            f"after shift/padding"
        )
    return {
        "act_shift": act_shift,
        "frame_stride": stride,
        "latent_frames_after_temporal_pack": latent_frames,
        "required_action_frames": required_actions,
    }


def replace_column(table: pa.Table, name: str, values) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def table_stats(table: pa.Table) -> dict:
    result = {}
    for name in table.column_names:
        values = np.asarray(table[name].to_pylist())
        if values.dtype.kind not in "biuf" or values.size == 0:
            continue
        result[name] = {
            "min": np.min(values, axis=0, keepdims=False).reshape(-1).tolist(),
            "max": np.max(values, axis=0, keepdims=False).reshape(-1).tolist(),
            "mean": np.mean(values, axis=0, keepdims=False).reshape(-1).tolist(),
            "std": np.std(values, axis=0, keepdims=False).reshape(-1).tolist(),
            "count": [table.num_rows],
        }
    return result


def build_episode_table(
    source_parquet: Path,
    *,
    source_start: int,
    source_end: int,
    actions: np.ndarray,
    episode_index: int,
    global_start: int,
    fps: float,
) -> pa.Table:
    source_schema = pq.read_schema(source_parquet)
    action_index = source_schema.get_field_index("action")
    if action_index < 0:
        raise KeyError(f"{source_parquet}: action field is missing")
    non_action_columns = [
        name for name in source_schema.names if name != "action"
    ]
    source = pq.read_table(source_parquet, columns=non_action_columns)
    if source_end > source.num_rows:
        raise ValueError(
            f"{source_parquet}: range ends at {source_end}, rows={source.num_rows}"
        )
    table = source.slice(source_start, source_end - source_start)
    if len(actions) != table.num_rows:
        raise ValueError(
            f"Action length {len(actions)} != selected parquet rows {table.num_rows}"
        )
    action_field = source_schema.field(action_index)
    table = table.add_column(
        action_index,
        action_field,
        pa.array(actions.tolist(), type=action_field.type),
    )
    table = replace_column(table, "episode_index", [episode_index] * table.num_rows)
    table = replace_column(table, "frame_index", range(table.num_rows))
    table = replace_column(
        table, "index", range(global_start, global_start + table.num_rows)
    )
    table = replace_column(
        table, "timestamp", np.arange(table.num_rows, dtype=np.float32) / fps
    )
    return table


def copy_rebased_latents(
    loaded: dict[str, dict],
    destination_repo: Path,
    *,
    episode_index: int,
    start_frame: int,
    length: int,
) -> None:
    episode_tag = f"episode_{episode_index:06d}_0_{length}.pth"
    for camera, value in loaded.items():
        copied = dict(value)
        source_ids = np.asarray(value["frame_ids"], dtype=np.int64)
        rebased = source_ids - start_frame
        if torch.is_tensor(value["frame_ids"]):
            copied["frame_ids"] = torch.as_tensor(
                rebased, dtype=value["frame_ids"].dtype
            )
        else:
            copied["frame_ids"] = rebased.tolist()
        copied["start_frame"] = 0
        copied["end_frame"] = length
        destination = (
            destination_repo / "latents" / "chunk-000" / camera / episode_tag
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(copied, destination)


def accepted(row: dict, min_process: float, min_action: float) -> bool:
    consistency = row.get("consistency", row.get("consistency_filter", {}))
    consistency_ok = bool(
        consistency.get("accepted", row.get("consistency_accepted", False))
    )
    process_score = float(row.get("process_score", float("-inf")))
    action = row.get("action_critic", {})
    action_ok = bool(action.get("accepted", False))
    action_score = float(action.get("action_score", float("-inf")))
    return (
        consistency_ok
        and process_score >= min_process
        and action_ok
        and action_score >= min_action
    )


def build_dataset(
    selected_jsonl: Path,
    output_root: Path,
    *,
    empty_embedding: Path,
    min_process_score: float = 0.0,
    min_action_score: float = 0.5,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite RFT dataset: {output_root}")
    rows = [
        row
        for row in read_jsonl(selected_jsonl)
        if accepted(row, min_process_score, min_action_score)
    ]
    if not rows:
        raise ValueError("No candidate passed consistency, process, and action filters")
    preparing = output_root.with_name(f"{output_root.name}.preparing")
    if preparing.exists():
        raise FileExistsError(f"Incomplete output already exists: {preparing}")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)

    preparing.mkdir(parents=True)
    try:
        try:
            os.link(empty_embedding, preparing / "empty_emb.pt")
        except OSError:
            shutil.copy2(empty_embedding, preparing / "empty_emb.pt")
        total_frames = 0
        compatibility = []
        for task, task_rows in sorted(by_task.items()):
            repo = preparing / "rft_selected_chunks" / task
            episode_rows = []
            episode_stats_rows = []
            source_info = None
            source_tasks = None
            global_index = 0
            for episode_index, row in enumerate(task_rows):
                start = int(row["start_frame"])
                end = int(row["end_frame"])
                if end <= start:
                    raise ValueError(f"Invalid candidate range: {start}:{end}")
                actions = load_actions(Path(row["action_path"]))
                if len(actions) != end - start:
                    raise ValueError(
                        f"{row['action_path']}: expected {end-start} actions, "
                        f"found {len(actions)}"
                    )
                loaded, frame_ids = load_and_validate_latents(
                    row["latent_paths"], start, end
                )
                contract = validate_loader_alignment(actions, frame_ids, start)
                compatibility.append({"task": task, "episode": episode_index, **contract})

                source_repo = Path(row["source_repo"])
                if source_info is None:
                    with (source_repo / "meta" / "info.json").open(
                        encoding="utf-8"
                    ) as handle:
                        source_info = json.load(handle)
                    source_tasks_path = source_repo / "meta" / "tasks.jsonl"
                    source_tasks = (
                        source_tasks_path
                        if source_tasks_path.is_file()
                        else None
                    )
                table = build_episode_table(
                    Path(row["source_parquet"]),
                    source_start=start,
                    source_end=end,
                    actions=actions,
                    episode_index=episode_index,
                    global_start=global_index,
                    fps=float(row.get("fps", source_info.get("fps", 30))),
                )
                stats = table_stats(table)
                episode_stats_rows.append(
                    {"episode_index": episode_index, "stats": stats}
                )
                parquet_path = (
                    repo
                    / "data"
                    / "chunk-000"
                    / f"episode_{episode_index:06d}.parquet"
                )
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, parquet_path)
                copy_rebased_latents(
                    loaded,
                    repo,
                    episode_index=episode_index,
                    start_frame=start,
                    length=len(actions),
                )
                instruction = str(row.get("text", row.get("instruction", task)))
                episode_rows.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [instruction],
                        "length": len(actions),
                        "action_config": [
                            {
                                "start_frame": 0,
                                "end_frame": len(actions),
                                "action_text": instruction,
                            }
                        ],
                    }
                )
                global_index += len(actions)
                total_frames += len(actions)

            assert source_info is not None
            info = dict(source_info)
            # RFT training consumes generated latents directly. Keeping source
            # video features would make LeRobot require unrelated source MP4s
            # even though LatentLeRobotDataset never decodes them.
            info["features"] = {
                key: value
                for key, value in source_info["features"].items()
                if value.get("dtype") != "video"
            }
            info.update(
                {
                    "total_episodes": len(episode_rows),
                    "total_frames": global_index,
                    "total_chunks": 1,
                    "total_videos": 0,
                    "splits": {"train": f"0:{len(episode_rows)}"},
                }
            )
            (repo / "meta").mkdir(parents=True, exist_ok=True)
            with (repo / "meta" / "info.json").open("w", encoding="utf-8") as handle:
                json.dump(info, handle, indent=2)
                handle.write("\n")
            write_jsonl(repo / "meta" / "episodes.jsonl", episode_rows)
            write_jsonl(
                repo / "meta" / "episodes_stats.jsonl", episode_stats_rows
            )
            if source_tasks is not None:
                shutil.copy2(source_tasks, repo / "meta" / "tasks.jsonl")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_candidates": str(selected_jsonl.resolve()),
            "filters": {
                "consistency_required": True,
                "min_process_score": min_process_score,
                "min_action_score": min_action_score,
            },
            "tasks": len(by_task),
            "episodes": len(rows),
            "frames": total_frames,
            "training_semantics": (
                "one selected generated chunk becomes one action_config trajectory "
                "segment; the unchanged WAM loader performs its normal temporal packing"
            ),
            "loader_compatibility": compatibility,
        }
        with (preparing / "rft_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        preparing.rename(output_root)
        return manifest
    except Exception:
        # Keep the .preparing tree for diagnosis; never expose it as training-ready.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--empty-embedding", type=Path, required=True)
    parser.add_argument("--min-process-score", type=float, default=0.0)
    parser.add_argument("--min-action-score", type=float, default=0.5)
    args = parser.parse_args()
    manifest = build_dataset(
        args.selected_jsonl,
        args.output_root,
        empty_embedding=args.empty_embedding,
        min_process_score=args.min_process_score,
        min_action_score=args.min_action_score,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("RFT_SIDECAR_DATASET_OK")


if __name__ == "__main__":
    main()
