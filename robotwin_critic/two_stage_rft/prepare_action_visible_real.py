#!/usr/bin/env python3
"""Build the selected Stage-1 + Stage-2 action-visible RoboTwin replay set.

Stage 2 deliberately contains no action column. This tool uses split_manifest.json
to recover only the selected Stage-2 episodes from the original RoboTwin dataset,
then combines them with the selected Stage-1 episodes into a self-contained view.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from script.prepare_robotwin_two_stage_dataset import (
    CAMERA_KEYS,
    dataset_name,
    episode_chunk,
    find_task_repositories,
    link_file,
    materialize_task,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = 1
COMPLETE_NAME = "ACTION_VISIBLE_COMPLETE.json"
MANIFEST_NAME = "action_visible_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Prepared root containing stage1/, stage2/, and split_manifest.json.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Original action-visible RoboTwin root. If omitted, the source_root "
            "recorded in split_manifest.json is used when it still exists."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Destination (default: PREPARED_ROOT/action_visible_real).",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="Use copy for a self-contained dataset that will move to another host.",
    )
    parser.add_argument(
        "--allow-missing-latent-segments",
        type=int,
        default=8,
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def resolve_source_root(manifest: dict, requested: Path | None) -> Path:
    candidates = []
    if requested is not None:
        candidates.append(requested.expanduser())
    recorded = manifest.get("source_root")
    if recorded:
        candidates.append(Path(recorded).expanduser())
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(
        "The Stage-2 action labels are redacted. Pass --source-root pointing to "
        "the original action-visible RoboTwin dataset."
    )


def selected_indices(domain_row: dict, *, domain: str, task: str) -> list[int]:
    selected = [
        int(value)
        for value in domain_row["selected_source_episode_indices_ranked"]
    ]
    stage1 = {int(value) for value in domain_row["stage1_source_episode_indices"]}
    stage2 = {int(value) for value in domain_row["stage2_source_episode_indices"]}
    if stage1 & stage2 or set(selected) != stage1 | stage2:
        raise ValueError(f"{domain}/{task}: invalid Stage-1/Stage-2 partition")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{domain}/{task}: duplicate selected source episodes")
    return selected


def audit_action_visible_root(output_root: Path, *, require_complete: bool) -> dict:
    manifest_path = output_root / MANIFEST_NAME
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported action-visible manifest schema")

    episodes_total = 0
    segments_total = 0
    missing_latents = 0
    for row in manifest["repositories"]:
        repo = output_root / row["output_repo"]
        info = read_json(repo / "meta" / "info.json")
        if info.get("action_visibility") != "visible":
            raise ValueError(f"{repo}: action_visibility is not visible")
        if "action" not in info.get("features", {}):
            raise ValueError(f"{repo}: action feature is absent from info.json")
        episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
        expected = int(row["episodes"])
        if len(episodes) != expected:
            raise ValueError(f"{repo}: expected {expected} episodes, got {len(episodes)}")
        indices = [int(item["episode_index"]) for item in episodes]
        if indices != list(range(expected)):
            raise ValueError(f"{repo}: episode indices are not contiguous")

        import pyarrow.parquet as pq

        chunks_size = int(info.get("chunks_size", 1000))
        for episode in episodes:
            index = int(episode["episode_index"])
            chunk = episode_chunk(index, chunks_size)
            tag = f"episode_{index:06d}"
            parquet = repo / "data" / f"chunk-{chunk:03d}" / f"{tag}.parquet"
            if "action" not in pq.read_schema(parquet).names:
                raise ValueError(f"{parquet}: action column is absent")
            for action_config in episode.get("action_config", []):
                segments_total += 1
                start = int(action_config["start_frame"])
                end = int(action_config["end_frame"])
                if any(
                    not (
                        repo
                        / "latents"
                        / f"chunk-{chunk:03d}"
                        / camera
                        / f"{tag}_{start}_{end}.pth"
                    ).is_file()
                    for camera in CAMERA_KEYS
                ):
                    missing_latents += 1
        episodes_total += len(episodes)

    if require_complete:
        complete = read_json(output_root / COMPLETE_NAME)
        if complete.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError(f"{COMPLETE_NAME}: manifest SHA256 mismatch")
    return {
        "repositories": len(manifest["repositories"]),
        "episodes": episodes_total,
        "segments": segments_total,
        "missing_latent_segments": missing_latents,
        "action_visible": True,
    }


def prepare(args: argparse.Namespace) -> dict:
    prepared_root = args.prepared_root.expanduser().resolve()
    for stage in ("stage1", "stage2"):
        if not (prepared_root / stage).is_dir():
            raise FileNotFoundError(prepared_root / stage)
    split_manifest_path = prepared_root / "split_manifest.json"
    split_manifest = read_json(split_manifest_path)
    complete_path = prepared_root / "PREPARATION_COMPLETE.json"
    if complete_path.is_file():
        complete = read_json(complete_path)
        if complete.get("manifest_sha256") != sha256_file(split_manifest_path):
            raise ValueError("PREPARATION_COMPLETE manifest SHA256 mismatch")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else prepared_root / "action_visible_real"
    )
    if args.verify_only:
        return audit_action_visible_root(output_root, require_complete=True)
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_root}; use --verify-only to audit it"
        )

    source_root = resolve_source_root(split_manifest, args.source_root)
    domain_dirs = split_manifest["domains"]
    repositories = {
        domain: find_task_repositories(source_root / domain_dirs[domain], domain)
        for domain in ("clean", "randomized")
    }
    preparing_root = output_root.with_name(f"{output_root.name}.preparing")
    if preparing_root.exists():
        raise FileExistsError(
            f"Incomplete preparation exists at {preparing_root}; inspect it first"
        )

    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(source_root),
        "prepared_root": str(prepared_root),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "link_mode": args.link_mode,
        "action_visibility": "visible",
        "selection": "exact union of manifest Stage-1 and Stage-2 episodes",
        "repositories": [],
    }
    preparing_root.mkdir(parents=True)
    try:
        empty_embedding = source_root / "empty_emb.pt"
        if not empty_embedding.is_file():
            empty_embedding = prepared_root / "stage1" / "empty_emb.pt"
        link_file(empty_embedding, preparing_root / "empty_emb.pt", args.link_mode)

        for task_row in split_manifest["tasks"]:
            task = task_row["task"]
            for domain in ("clean", "randomized"):
                if task not in repositories[domain]:
                    raise KeyError(f"Original dataset is missing {domain}/{task}")
                indices = selected_indices(
                    task_row["domains"][domain], domain=domain, task=task
                )
                relative_repo = Path(dataset_name(domain, "real", len(indices))) / task
                result = materialize_task(
                    src_repo=repositories[domain][task],
                    dst_repo=preparing_root / relative_repo,
                    selected_indices=indices,
                    link_mode=args.link_mode,
                    redact_action=False,
                )
                output_manifest["repositories"].append(
                    {
                        "domain": domain,
                        "task": task,
                        "output_repo": str(relative_repo),
                        "episodes": len(indices),
                        "selected_source_episode_indices": indices,
                        "source_to_destination_index": result[
                            "source_to_destination_index"
                        ],
                    }
                )

        write_json(preparing_root / MANIFEST_NAME, output_manifest)
        summary = audit_action_visible_root(preparing_root, require_complete=False)
        if summary["missing_latent_segments"] > args.allow_missing_latent_segments:
            raise ValueError(
                f"Found {summary['missing_latent_segments']} missing latent segments; "
                f"allowed={args.allow_missing_latent_segments}"
            )
        complete = {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "manifest_sha256": sha256_file(preparing_root / MANIFEST_NAME),
            "summary": summary,
        }
        write_json(preparing_root / COMPLETE_NAME, complete)
        os.replace(preparing_root, output_root)
        return complete
    except Exception:
        print(f"Partial output preserved at {preparing_root}", flush=True)
        raise


def main() -> None:
    args = parse_args()
    result = prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
