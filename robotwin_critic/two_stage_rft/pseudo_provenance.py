"""Validate pseudo-package split provenance without importing Torch."""

from __future__ import annotations

import json
from pathlib import Path

from robotwin_critic.two_stage_rft.protocol import sha256_file


def validate_pseudo_split_provenance(
    rows: list[dict],
    *,
    expected_split_sha256: str,
    split_manifest_path: str | Path | None,
) -> dict[str, object]:
    package_hashes = sorted(
        {str(row.get("split_manifest_sha256", "")) for row in rows}
    )
    if package_hashes == [expected_split_sha256]:
        return {
            "validation_mode": "raw_hash",
            "package_split_sha256": expected_split_sha256,
            "current_split_sha256": expected_split_sha256,
            "validated_rows": len(rows),
        }
    if split_manifest_path is None:
        raise ValueError(
            f"Pseudo data split hashes {set(package_hashes)} do not match Stage-1 "
            f"split {expected_split_sha256}; split_manifest_path is required for "
            "semantic validation"
        )
    manifest_path = Path(split_manifest_path).expanduser().resolve()
    actual_hash = sha256_file(manifest_path)
    if actual_hash != expected_split_sha256:
        raise ValueError(
            f"Current split manifest hash {actual_hash} does not match expected "
            f"hash {expected_split_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    membership: dict[tuple[str, str], dict[str, object]] = {}
    for task_row in manifest.get("tasks", []):
        task = str(task_row.get("task", ""))
        for domain, domain_row in task_row.get("domains", {}).items():
            repo_basenames = {
                Path(str(repo)).name
                for repo in (
                    domain_row.get("source_repo"),
                    domain_row.get("stage2_output_repo"),
                )
                if repo
            }
            membership[(task, str(domain))] = {
                "episodes": {
                    int(value)
                    for value in domain_row.get(
                        "stage2_source_episode_indices", []
                    )
                },
                "repo_basenames": repo_basenames,
                "source_to_output": {
                    int(source): int(output)
                    for source, output in domain_row.get("stage2_output", {})
                    .get("source_to_destination_index", {})
                    .items()
                },
            }

    for row_number, row in enumerate(rows, start=1):
        required = (
            "source_stage", "task", "domain", "source_episode_index",
            "output_episode_index", "frame_index",
        )
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise ValueError(
                f"Pseudo row {row_number} is missing provenance required for "
                f"semantic split validation: {missing}"
            )
        if str(row["source_stage"]) != "stage2":
            raise ValueError(
                f"Pseudo row {row_number} source_stage must be stage2, got "
                f"{row['source_stage']!r}"
            )
        task = str(row["task"])
        source_task = row.get("source_task")
        if source_task is not None and str(source_task) != task:
            raise ValueError(
                f"Pseudo row {row_number} task/source_task disagree: "
                f"{task!r} != {source_task!r}"
            )
        domain = str(row["domain"])
        key = (task, domain)
        if key not in membership:
            raise ValueError(
                f"Pseudo row {row_number} task/domain is absent from current "
                f"split: {task}/{domain}"
            )
        split_entry = membership[key]
        episodes = split_entry["episodes"]
        episode = int(row["source_episode_index"])
        if episode not in episodes:
            raise ValueError(
                f"Pseudo row {row_number} episode {episode} is not in current "
                f"stage2 membership for {task}/{domain}"
            )
        expected_output = split_entry["source_to_output"].get(episode)
        output_episode = int(row["output_episode_index"])
        if expected_output is None or output_episode != expected_output:
            raise ValueError(
                f"Pseudo row {row_number} source/output episode mapping "
                f"{episode}->{output_episode} does not match current split "
                f"{episode}->{expected_output} for {task}/{domain}"
            )
        frame = int(row["frame_index"])
        source_context_id = row.get("source_context_id")
        expected_context_id = f"{task}/{domain}/{episode}/{frame}"
        if source_context_id and str(source_context_id) != expected_context_id:
            raise ValueError(
                f"Pseudo row {row_number} source_context_id "
                f"{source_context_id!r} does not match {expected_context_id!r}"
            )
        source_parquet = row.get("source_parquet")
        expected_parquet = f"episode_{output_episode:06d}.parquet"
        if source_parquet and Path(str(source_parquet)).name != expected_parquet:
            raise ValueError(
                f"Pseudo row {row_number} source_parquet basename "
                f"{Path(str(source_parquet)).name!r} does not match "
                f"{expected_parquet!r}"
            )
        source_repo = row.get("source_repo")
        accepted_repo_basenames = split_entry["repo_basenames"]
        if source_repo and accepted_repo_basenames:
            observed_repo_basename = Path(str(source_repo)).name
            if observed_repo_basename not in accepted_repo_basenames:
                raise ValueError(
                    f"Pseudo row {row_number} source_repo basename "
                    f"{observed_repo_basename!r} does not match current split "
                    f"repos {sorted(accepted_repo_basenames)!r}"
                )

    return {
        "validation_mode": "semantic_membership",
        "package_split_sha256": ",".join(package_hashes),
        "current_split_sha256": actual_hash,
        "validated_rows": len(rows),
    }
