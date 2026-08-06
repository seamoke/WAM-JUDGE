#!/usr/bin/env python3
"""Prepare an immutable two-stage RoboTwin clean/randomized training split.

The output is a training-only LeRobot view. Video and latent payloads are linked
from the official dataset. Stage-2 parquet files are rewritten without action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

DOMAIN_DIRS = {
    "clean": "lerobot_robotwin_eef_clean_50",
    "randomized": "lerobot_robotwin_eef_aug_500",
}

SCHEMA_VERSION = 2

DOMAIN_REPOSITORY_SUFFIXES = {
    "clean": (
        "-demo_clean_collect_200-50",
        "-piper_clean_50-50",
    ),
    "randomized": (
        "-aloha-agilex_randomized_500-1000",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help=(
            "Official robotwin-clean-and-aug-lerobot root containing "
            "lerobot_robotwin_eef_clean_50 and lerobot_robotwin_eef_aug_500"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--per-domain-total", type=int, default=50)
    parser.add_argument("--stage1-per-domain", type=int, default=30)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help=(
            "Hardlinks use no extra payload space but require one filesystem; "
            "copy creates a self-contained dataset suitable for migration."
        ),
    )
    parser.add_argument(
        "--allow-missing-latent-segments",
        type=int,
        default=8,
        help=(
            "Maximum selected action segments missing at least one camera latent. "
            "The published clean snapshot is known to have up to 8."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Audit an existing prepared output without changing it.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_rank(seed: int, domain: str, task: str, episode_index: int) -> str:
    value = f"{seed}:{domain}:{task}:{episode_index}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_episode_indices(
    episodes: list[dict],
    *,
    seed: int,
    domain: str,
    task: str,
    total: int,
    stage1_count: int,
) -> tuple[list[int], list[int], list[int]]:
    indices = [int(row["episode_index"]) for row in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{domain}/{task}: duplicate episode_index values")
    if len(indices) < total:
        raise ValueError(
            f"{domain}/{task}: needs {total} episodes, found {len(indices)}"
        )

    ranked = sorted(
        indices,
        key=lambda index: (episode_rank(seed, domain, task, index), index),
    )
    selected = ranked[:total]
    stage1 = selected[:stage1_count]
    stage2 = selected[stage1_count:]
    return selected, sorted(stage1), sorted(stage2)


def link_file(src: Path, dst: Path, mode: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.samefile(src):
            return
        raise FileExistsError(dst)
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(os.path.relpath(src, dst.parent))
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def materialize_parquet(src: Path, dst: Path, *, redact_action: bool) -> None:
    if not redact_action:
        link_file(src, dst, "hardlink")
        return
    import pyarrow.parquet as pq

    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    table = pq.read_table(src)
    if "action" not in table.column_names:
        raise ValueError(f"Stage-2 source parquet has no action column: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.drop(["action"]), dst, compression="zstd")


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def reindexed_rows(
    rows: list[dict],
    selected_indices: list[int],
) -> list[dict]:
    rows_by_index = {int(row["episode_index"]): row for row in rows}
    result = []
    for new_index, source_index in enumerate(selected_indices):
        if source_index not in rows_by_index:
            raise KeyError(f"Missing metadata row for episode {source_index}")
        row = dict(rows_by_index[source_index])
        row["episode_index"] = new_index
        result.append(row)
    return result


def redact_episode_action_stats(rows: list[dict]) -> list[dict]:
    result = []
    for source in rows:
        row = dict(source)
        if isinstance(row.get("stats"), dict):
            row["stats"] = dict(row["stats"])
            row["stats"].pop("action", None)
        result.append(row)
    return result


def link_episode_payload(
    *,
    src_repo: Path,
    dst_repo: Path,
    source_episode: dict,
    source_index: int,
    destination_index: int,
    chunks_size: int,
    link_mode: str,
    redact_action: bool,
) -> dict:
    source_chunk = episode_chunk(source_index, chunks_size)
    destination_chunk = episode_chunk(destination_index, chunks_size)
    source_chunk_tag = f"chunk-{source_chunk:03d}"
    destination_chunk_tag = f"chunk-{destination_chunk:03d}"
    source_episode_tag = f"episode_{source_index:06d}"
    destination_episode_tag = f"episode_{destination_index:06d}"

    source_parquet = (
        src_repo / "data" / source_chunk_tag / f"{source_episode_tag}.parquet"
    )
    destination_parquet = (
        dst_repo
        / "data"
        / destination_chunk_tag
        / f"{destination_episode_tag}.parquet"
    )
    if redact_action:
        materialize_parquet(source_parquet, destination_parquet, redact_action=True)
    else:
        link_file(source_parquet, destination_parquet, link_mode)

    linked_videos = 0
    missing_videos = []
    for camera in CAMERA_KEYS:
        source_video = (
            src_repo
            / "videos"
            / source_chunk_tag
            / camera
            / f"{source_episode_tag}.mp4"
        )
        if source_video.is_file():
            link_file(
                source_video,
                dst_repo
                / "videos"
                / destination_chunk_tag
                / camera
                / f"{destination_episode_tag}.mp4",
                link_mode,
            )
            linked_videos += 1
        else:
            missing_videos.append(camera)

    missing_latent_segments = []
    for action_config in source_episode.get("action_config", []):
        start = int(action_config["start_frame"])
        end = int(action_config["end_frame"])
        missing_cameras = []
        for camera in CAMERA_KEYS:
            source_latent = (
                src_repo
                / "latents"
                / source_chunk_tag
                / camera
                / f"{source_episode_tag}_{start}_{end}.pth"
            )
            if not source_latent.is_file():
                missing_cameras.append(camera)
                continue
            link_file(
                source_latent,
                dst_repo
                / "latents"
                / destination_chunk_tag
                / camera
                / f"{destination_episode_tag}_{start}_{end}.pth",
                link_mode,
            )
        if missing_cameras:
            missing_latent_segments.append(
                {
                    "source_episode_index": source_index,
                    "start_frame": start,
                    "end_frame": end,
                    "missing_cameras": missing_cameras,
                }
            )

    return {
        "linked_videos": linked_videos,
        "missing_videos": missing_videos,
        "missing_latent_segments": missing_latent_segments,
    }


def materialize_task(
    *,
    src_repo: Path,
    dst_repo: Path,
    selected_indices: list[int],
    link_mode: str,
    redact_action: bool,
) -> dict:
    meta_dir = src_repo / "meta"
    info = read_json(meta_dir / "info.json")
    chunks_size = int(info.get("chunks_size", 1000))
    source_episodes = read_jsonl(meta_dir / "episodes.jsonl")
    episodes_by_index = {
        int(episode["episode_index"]): episode for episode in source_episodes
    }
    selected_episodes = reindexed_rows(source_episodes, selected_indices)

    destination_meta = dst_repo / "meta"
    write_jsonl(destination_meta / "episodes.jsonl", selected_episodes)

    for optional_name in ("episodes_stats.jsonl", "episodes_ori.jsonl"):
        source_path = meta_dir / optional_name
        if source_path.is_file():
            optional_rows = reindexed_rows(
                read_jsonl(source_path), selected_indices
            )
            if redact_action and optional_name == "episodes_stats.jsonl":
                optional_rows = redact_episode_action_stats(optional_rows)
            write_jsonl(
                destination_meta / optional_name,
                optional_rows,
            )

    tasks_path = meta_dir / "tasks.jsonl"
    if tasks_path.is_file():
        link_file(tasks_path, destination_meta / "tasks.jsonl", link_mode)

    linked_videos = 0
    missing_videos = []
    missing_latent_segments = []
    for destination_index, source_index in enumerate(selected_indices):
        payload = link_episode_payload(
            src_repo=src_repo,
            dst_repo=dst_repo,
            source_episode=episodes_by_index[source_index],
            source_index=source_index,
            destination_index=destination_index,
            chunks_size=chunks_size,
            link_mode=link_mode,
            redact_action=redact_action,
        )
        linked_videos += payload["linked_videos"]
        if payload["missing_videos"]:
            missing_videos.append(
                {
                    "source_episode_index": source_index,
                    "missing_cameras": payload["missing_videos"],
                }
            )
        missing_latent_segments.extend(payload["missing_latent_segments"])

    total_frames = sum(int(episode.get("length", 0)) for episode in selected_episodes)
    total_episodes = len(selected_episodes)
    total_chunks = (
        episode_chunk(total_episodes - 1, chunks_size) + 1
        if total_episodes
        else 0
    )
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = linked_videos
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    if redact_action:
        info.get("features", {}).pop("action", None)
        info["action_visibility"] = "redacted"
    else:
        info["action_visibility"] = "visible"
    write_json(destination_meta / "info.json", info)

    return {
        "episodes": total_episodes,
        "frames": total_frames,
        "linked_videos": linked_videos,
        "missing_video_episodes": len(missing_videos),
        "missing_latent_segments": missing_latent_segments,
        "action_visibility": "redacted" if redact_action else "visible",
        "source_to_destination_index": {
            str(source_index): destination_index
            for destination_index, source_index in enumerate(selected_indices)
        },
    }


def canonical_task_name(domain: str, repository_name: str) -> str:
    for suffix in DOMAIN_REPOSITORY_SUFFIXES[domain]:
        if repository_name.endswith(suffix):
            return repository_name[: -len(suffix)]
    return repository_name


def find_task_repositories(domain_root: Path, domain: str) -> dict[str, Path]:
    repositories = {}
    for info_path in sorted(domain_root.glob("*/meta/info.json")):
        repository = info_path.parent.parent
        task = canonical_task_name(domain, repository.name)
        if task in repositories:
            raise ValueError(
                f"{domain}: repository name collision for canonical task {task!r}: "
                f"{repositories[task].name!r} and {repository.name!r}"
            )
        repositories[task] = repository
    return repositories


def dataset_name(domain: str, stage: str, count: int) -> str:
    source_prefix = "clean" if domain == "clean" else "aug"
    return f"lerobot_robotwin_eef_{source_prefix}_{stage}_{count}"


def audit_prepared_root(
    output_root: Path,
    manifest: dict,
    *,
    allow_missing_latent_segments: int,
    require_complete_marker: bool,
) -> dict:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported or missing split manifest schema_version")
    if manifest.get("stage2_action_statistics_redacted") is not True:
        raise ValueError("Manifest does not guarantee Stage-2 action-stat redaction")

    per_domain_total = int(manifest["split"]["per_domain_total"])
    stage1_count = int(manifest["split"]["stage1_per_domain"])
    stage2_count = per_domain_total - stage1_count
    expected_tasks = int(manifest["expected_tasks"])
    task_rows = manifest["tasks"]
    if len(task_rows) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} manifest tasks, found {len(task_rows)}"
        )

    total_output_episodes = 0
    stage_segments = {"stage1": 0, "stage2": 0}
    stage_valid_segments = {"stage1": 0, "stage2": 0}
    missing_latent_segments = []
    seen_task_names = set()
    for task_row in task_rows:
        task = task_row["task"]
        if task in seen_task_names:
            raise ValueError(f"Duplicate task in manifest: {task}")
        seen_task_names.add(task)

        for domain in ("clean", "randomized"):
            domain_row = task_row["domains"][domain]
            selected = domain_row["selected_source_episode_indices_ranked"]
            stage1 = domain_row["stage1_source_episode_indices"]
            stage2 = domain_row["stage2_source_episode_indices"]
            if len(selected) != per_domain_total or len(set(selected)) != per_domain_total:
                raise ValueError(f"{domain}/{task}: invalid selected episode list")
            if len(stage1) != stage1_count or len(stage2) != stage2_count:
                raise ValueError(f"{domain}/{task}: invalid stage sizes")
            if set(stage1) & set(stage2):
                raise ValueError(f"{domain}/{task}: Stage 1 and Stage 2 overlap")
            if set(stage1) | set(stage2) != set(selected):
                raise ValueError(f"{domain}/{task}: stages do not cover selected set")

            for stage, expected_count in (
                ("stage1", stage1_count),
                ("stage2", stage2_count),
            ):
                relative_repo = Path(domain_row[f"{stage}_output_repo"])
                repo = output_root / relative_repo
                episodes = read_jsonl(repo / "meta" / "episodes.jsonl")
                if len(episodes) != expected_count:
                    raise ValueError(
                        f"{relative_repo}: expected {expected_count} episodes, "
                        f"found {len(episodes)}"
                    )
                if [int(row["episode_index"]) for row in episodes] != list(
                    range(expected_count)
                ):
                    raise ValueError(f"{relative_repo}: non-contiguous episode indices")

                info = read_json(repo / "meta" / "info.json")
                expected_visibility = "visible" if stage == "stage1" else "redacted"
                if info.get("action_visibility") != expected_visibility:
                    raise ValueError(
                        f"{relative_repo}: expected action_visibility="
                        f"{expected_visibility}, got {info.get('action_visibility')}"
                    )
                info_has_action = "action" in info.get("features", {})
                if info_has_action != (stage == "stage1"):
                    raise ValueError(
                        f"{relative_repo}: info.json action feature violates {stage} policy"
                    )
                stats_path = repo / "meta" / "episodes_stats.jsonl"
                if stats_path.is_file():
                    for stats_row in read_jsonl(stats_path):
                        stats_has_action = "action" in stats_row.get("stats", {})
                        if stats_has_action != (stage == "stage1"):
                            raise ValueError(
                                f"{stats_path}: action statistics violate {stage} "
                                "visibility policy"
                            )
                chunks_size = int(info.get("chunks_size", 1000))
                for episode in episodes:
                    episode_index = int(episode["episode_index"])
                    chunk = episode_chunk(episode_index, chunks_size)
                    episode_tag = f"episode_{episode_index:06d}"
                    parquet = (
                        repo
                        / "data"
                        / f"chunk-{chunk:03d}"
                        / f"{episode_tag}.parquet"
                    )
                    if not parquet.is_file():
                        raise FileNotFoundError(parquet)
                    import pyarrow.parquet as pq

                    parquet_has_action = "action" in pq.read_schema(parquet).names
                    if parquet_has_action != (stage == "stage1"):
                        raise ValueError(
                            f"{parquet}: action column violates {stage} visibility policy"
                        )

                    for action_config in episode.get("action_config", []):
                        stage_segments[stage] += 1
                        start = int(action_config["start_frame"])
                        end = int(action_config["end_frame"])
                        absent = []
                        for camera in CAMERA_KEYS:
                            latent = (
                                repo
                                / "latents"
                                / f"chunk-{chunk:03d}"
                                / camera
                                / f"{episode_tag}_{start}_{end}.pth"
                            )
                            if not latent.is_file():
                                absent.append(camera)
                        if absent:
                            missing_latent_segments.append(
                                {
                                    "stage": stage,
                                    "domain": domain,
                                    "task": task,
                                    "episode_index": episode_index,
                                    "start_frame": start,
                                    "end_frame": end,
                                    "missing_cameras": absent,
                                }
                            )
                        else:
                            stage_valid_segments[stage] += 1
                total_output_episodes += len(episodes)

    if len(missing_latent_segments) > allow_missing_latent_segments:
        raise ValueError(
            f"Found {len(missing_latent_segments)} selected segments with missing "
            f"latents; allowed={allow_missing_latent_segments}"
        )

    manifest_path = output_root / "split_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if require_complete_marker:
        complete_path = output_root / "PREPARATION_COMPLETE.json"
        complete = read_json(complete_path)
        if complete.get("manifest_sha256") != manifest_sha256:
            raise ValueError("PREPARATION_COMPLETE manifest SHA256 mismatch")

    return {
        "tasks": len(task_rows),
        "domains": 2,
        "stage1_episodes": expected_tasks * stage1_count * 2,
        "stage2_episodes": expected_tasks * stage2_count * 2,
        "total_output_episodes": total_output_episodes,
        "stage1_segments": stage_segments["stage1"],
        "stage1_valid_segments": stage_valid_segments["stage1"],
        "stage2_segments": stage_segments["stage2"],
        "stage2_valid_segments": stage_valid_segments["stage2"],
        "missing_latent_segments": len(missing_latent_segments),
        "manifest_sha256": manifest_sha256,
        "stage1_action_visible": True,
        "stage2_action_redacted": True,
        "stage2_action_statistics_redacted": True,
    }


def prepare_dataset(args: argparse.Namespace) -> dict:
    if args.source_root is None:
        raise ValueError("--source-root is required unless --verify-only is used")
    if args.per_domain_total <= 0:
        raise ValueError("--per-domain-total must be positive")
    if not 0 < args.stage1_per_domain < args.per_domain_total:
        raise ValueError(
            "--stage1-per-domain must be positive and smaller than --per-domain-total"
        )

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    preparing_root = output_root.with_name(f"{output_root.name}.preparing")
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared dataset: {output_root}. "
            "Use --verify-only to audit it."
        )
    if preparing_root.exists():
        raise FileExistsError(
            f"Found an incomplete preparation directory: {preparing_root}. "
            "Inspect it before removing it and retrying."
        )

    domain_repositories = {
        domain: find_task_repositories(source_root / directory, domain)
        for domain, directory in DOMAIN_DIRS.items()
    }
    clean_tasks = set(domain_repositories["clean"])
    randomized_tasks = set(domain_repositories["randomized"])
    if clean_tasks != randomized_tasks:
        raise ValueError(
            "Clean/randomized task sets differ: "
            f"clean_only={sorted(clean_tasks - randomized_tasks)}, "
            f"randomized_only={sorted(randomized_tasks - clean_tasks)}"
        )
    if len(clean_tasks) != args.expected_tasks:
        raise ValueError(
            f"Expected {args.expected_tasks} matched tasks, found {len(clean_tasks)}"
        )

    empty_embedding = source_root / "empty_emb.pt"
    if not empty_embedding.is_file():
        raise FileNotFoundError(empty_embedding)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "expected_tasks": args.expected_tasks,
        "link_mode": args.link_mode,
        "stage1_action_visible": True,
        "stage2_action_redacted": True,
        "stage2_action_statistics_redacted": True,
        "split": {
            "seed": args.seed,
            "ranking": "sha256(seed:domain:task:episode_index)",
            "per_domain_total": args.per_domain_total,
            "stage1_per_domain": args.stage1_per_domain,
            "stage2_per_domain": args.per_domain_total - args.stage1_per_domain,
        },
        "domains": DOMAIN_DIRS,
        "tasks": [],
    }

    preparing_root.mkdir(parents=True)
    try:
        for stage in ("stage1", "stage2"):
            link_file(
                empty_embedding,
                preparing_root / stage / "empty_emb.pt",
                args.link_mode,
            )

        for task in sorted(clean_tasks):
            task_row = {"task": task, "domains": {}}
            for domain in ("clean", "randomized"):
                source_repo = domain_repositories[domain][task]
                source_episodes = read_jsonl(
                    source_repo / "meta" / "episodes.jsonl"
                )
                selected, stage1_indices, stage2_indices = select_episode_indices(
                    source_episodes,
                    seed=args.seed,
                    domain=domain,
                    task=task,
                    total=args.per_domain_total,
                    stage1_count=args.stage1_per_domain,
                )

                domain_row = {
                    "source_repo": str(source_repo),
                    "available_episodes": len(source_episodes),
                    "selected_source_episode_indices_ranked": selected,
                    "stage1_source_episode_indices": stage1_indices,
                    "stage2_source_episode_indices": stage2_indices,
                }
                for stage, selected_indices in (
                    ("stage1", stage1_indices),
                    ("stage2", stage2_indices),
                ):
                    count = len(selected_indices)
                    relative_repo = (
                        Path(stage)
                        / dataset_name(domain, stage, count)
                        / task
                    )
                    result = materialize_task(
                        src_repo=source_repo,
                        dst_repo=preparing_root / relative_repo,
                        selected_indices=selected_indices,
                        link_mode=args.link_mode,
                        redact_action=(stage == "stage2"),
                    )
                    domain_row[f"{stage}_output_repo"] = str(relative_repo)
                    domain_row[f"{stage}_output"] = result
                task_row["domains"][domain] = domain_row
            manifest["tasks"].append(task_row)

        manifest_path = preparing_root / "split_manifest.json"
        write_json(manifest_path, manifest)
        summary = audit_prepared_root(
            preparing_root,
            manifest,
            allow_missing_latent_segments=args.allow_missing_latent_segments,
            require_complete_marker=False,
        )
        complete = {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "manifest_sha256": summary["manifest_sha256"],
            "summary": summary,
        }
        write_json(preparing_root / "PREPARATION_COMPLETE.json", complete)
        os.replace(preparing_root, output_root)
        return complete
    except Exception:
        print(
            f"Preparation failed; partial files were preserved at {preparing_root}",
            flush=True,
        )
        raise


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.verify_only:
        manifest = read_json(output_root / "split_manifest.json")
        summary = audit_prepared_root(
            output_root,
            manifest,
            allow_missing_latent_segments=args.allow_missing_latent_segments,
            require_complete_marker=True,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("TWO_STAGE_DATASET_AUDIT_OK")
        return

    complete = prepare_dataset(args)
    print(json.dumps(complete, indent=2, sort_keys=True))
    print("TWO_STAGE_DATASET_PREPARATION_OK")


if __name__ == "__main__":
    main()
