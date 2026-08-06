"""Read and validate the immutable 50+50 RoboTwin split manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


DOMAINS = ("clean", "randomized")
STAGES = ("stage1", "stage2")


@dataclass(frozen=True)
class EpisodeRef:
    task: str
    domain: str
    stage: str
    source_episode_index: int
    output_episode_index: int
    repo: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(
    prepared_root: Path,
    require_complete: bool = True,
    *,
    expected_per_domain_total: int = 50,
    expected_stage1_per_domain: int = 30,
) -> dict:
    prepared_root = prepared_root.expanduser().resolve()
    manifest_path = prepared_root / "split_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError(
            "Expected redacted split_manifest schema_version=2; rebuild old prepared "
            "data so Stage-2 parquet files do not physically expose action"
        )
    if manifest.get("stage1_action_visible") is not True:
        raise ValueError("Stage-1 action visibility contract is missing")
    if manifest.get("stage2_action_redacted") is not True:
        raise ValueError("Stage-2 action redaction contract is missing")
    if manifest.get("stage2_action_statistics_redacted") is not True:
        raise ValueError("Stage-2 action-stat redaction contract is missing")
    split = manifest["split"]
    expected_stage2_per_domain = (
        expected_per_domain_total - expected_stage1_per_domain
    )
    observed = (
        int(split["per_domain_total"]),
        int(split["stage1_per_domain"]),
        int(split["stage2_per_domain"]),
    )
    expected = (
        expected_per_domain_total,
        expected_stage1_per_domain,
        expected_stage2_per_domain,
    )
    if observed != expected:
        raise ValueError(
            f"Protocol split must be total/stage1/stage2={expected}, got {observed}"
        )
    if require_complete:
        with (prepared_root / "PREPARATION_COMPLETE.json").open(
            encoding="utf-8"
        ) as handle:
            complete = json.load(handle)
        actual = sha256_file(manifest_path)
        if complete.get("manifest_sha256") != actual:
            raise ValueError("Prepared-data completion marker does not match manifest")
    return manifest


def iter_episode_refs(
    prepared_root: Path,
    *,
    stages: tuple[str, ...] = STAGES,
    domains: tuple[str, ...] = DOMAINS,
    expected_per_domain_total: int = 50,
    expected_stage1_per_domain: int = 30,
) -> Iterator[EpisodeRef]:
    manifest = read_manifest(
        prepared_root,
        expected_per_domain_total=expected_per_domain_total,
        expected_stage1_per_domain=expected_stage1_per_domain,
    )
    for task_row in manifest["tasks"]:
        task = task_row["task"]
        for domain in domains:
            domain_row = task_row["domains"][domain]
            for stage in stages:
                source_indices = domain_row[f"{stage}_source_episode_indices"]
                mapping = domain_row[f"{stage}_output"][
                    "source_to_destination_index"
                ]
                repo = prepared_root / domain_row[f"{stage}_output_repo"]
                for source_index in source_indices:
                    yield EpisodeRef(
                        task=task,
                        domain=domain,
                        stage=stage,
                        source_episode_index=int(source_index),
                        output_episode_index=int(mapping[str(source_index)]),
                        repo=repo,
                    )


def audit_protocol(prepared_root: Path, expected_tasks: int = 50) -> dict:
    manifest = read_manifest(prepared_root)
    if len(manifest["tasks"]) != expected_tasks:
        raise ValueError(
            f"Expected {expected_tasks} tasks, found {len(manifest['tasks'])}"
        )
    counts = {
        stage: {domain: 0 for domain in DOMAINS}
        for stage in STAGES
    }
    seen: set[tuple[str, str, int]] = set()
    for ref in iter_episode_refs(prepared_root):
        key = (ref.task, ref.domain, ref.source_episode_index)
        if key in seen:
            raise ValueError(f"Episode leakage across stages: {key}")
        seen.add(key)
        counts[ref.stage][ref.domain] += 1
        if not (ref.repo / "meta" / "episodes.jsonl").is_file():
            raise FileNotFoundError(ref.repo / "meta" / "episodes.jsonl")
    expected = {
        "stage1": expected_tasks * 30,
        "stage2": expected_tasks * 20,
    }
    for stage in STAGES:
        for domain in DOMAINS:
            if counts[stage][domain] != expected[stage]:
                raise ValueError(
                    f"{stage}/{domain}: expected {expected[stage]}, "
                    f"found {counts[stage][domain]}"
                )
    return {
        "tasks": expected_tasks,
        "counts": counts,
        "episodes_total": sum(sum(row.values()) for row in counts.values()),
        "manifest_sha256": sha256_file(
            prepared_root / "split_manifest.json"
        ),
    }
