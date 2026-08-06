"""Count the exact Stage-2 WAM-loader chunk budget without reading hidden actions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from robotwin_critic.two_stage_rft.data_access import (
    episode_metadata,
    latent_segment_exists,
    latent_segment_num_frames,
)
from robotwin_critic.two_stage_rft.protocol import iter_episode_refs, sha256_file


def count_budget(
    prepared_root: Path,
    output: Path,
    *,
    max_episode_frames: int = 500,
    max_tasks: int = 0,
    max_episodes_per_group: int = 0,
    expected_per_domain_total: int = 50,
    expected_stage1_per_domain: int = 30,
) -> dict:
    metadata_cache: dict[Path, dict[int, dict]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    segment_counts: Counter[tuple[str, str]] = Counter()
    latent_frame_counts: Counter[tuple[str, str]] = Counter()
    skipped_long = skipped_missing_latent = 0
    refs = list(
        iter_episode_refs(
            prepared_root,
            stages=("stage2",),
            expected_per_domain_total=expected_per_domain_total,
            expected_stage1_per_domain=expected_stage1_per_domain,
        )
    )
    tasks = sorted({ref.task for ref in refs})
    if max_tasks:
        tasks = tasks[:max_tasks]
    task_set = set(tasks)
    group_seen: Counter[tuple[str, str]] = Counter()
    for ref in refs:
        if ref.task not in task_set:
            continue
        group = (ref.task, ref.domain)
        if max_episodes_per_group and group_seen[group] >= max_episodes_per_group:
            continue
        group_seen[group] += 1
        if ref.repo not in metadata_cache:
            metadata_cache[ref.repo] = episode_metadata(ref.repo)
        episode = metadata_cache[ref.repo][ref.output_episode_index]
        for config in episode.get("action_config", []):
            start = int(config["start_frame"])
            end = int(config["end_frame"])
            if end <= start:
                continue
            if max_episode_frames and end - start > max_episode_frames:
                skipped_long += 1
                continue
            if not latent_segment_exists(
                ref.repo, ref.output_episode_index, start, end
            ):
                skipped_missing_latent += 1
                continue
            latent_frames = latent_segment_num_frames(
                ref.repo, ref.output_episode_index, start, end
            )
            transition_chunks = max(latent_frames - 1, 0)
            counts[group] += transition_chunks
            segment_counts[group] += 1
            latent_frame_counts[group] += latent_frames
    groups = {
        f"{task}/{domain}": int(value)
        for (task, domain), value in sorted(counts.items())
    }
    result = {
        "schema_version": 1,
        "definition": "executable_state_transition_chunks_in_original_WAM_loader_for_stage2",
        "prepared_root": str(prepared_root.resolve()),
        "split_manifest_sha256": sha256_file(
            prepared_root / "split_manifest.json"
        ),
        "reads_action_column": False,
        "max_episode_frames": max_episode_frames,
        "pseudo_chunk_latent_frames": 2,
        "max_tasks": max_tasks,
        "max_episodes_per_group": max_episodes_per_group,
        "protocol_expected_per_domain_total": expected_per_domain_total,
        "protocol_expected_stage1_per_domain": expected_stage1_per_domain,
        "groups": groups,
        "segments_by_group": {
            f"{task}/{domain}": int(value)
            for (task, domain), value in sorted(segment_counts.items())
        },
        "latent_frames_by_group": {
            f"{task}/{domain}": int(value)
            for (task, domain), value in sorted(latent_frame_counts.items())
        },
        "total": sum(groups.values()),
        "skipped_long": skipped_long,
        "skipped_missing_latent": skipped_missing_latent,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episode-frames", type=int, default=500)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-group", type=int, default=0)
    parser.add_argument("--expected-per-domain-total", type=int, default=50)
    parser.add_argument("--expected-stage1-per-domain", type=int, default=30)
    args = parser.parse_args()
    result = count_budget(
        args.prepared_root,
        args.output,
        max_episode_frames=args.max_episode_frames,
        max_tasks=args.max_tasks,
        max_episodes_per_group=args.max_episodes_per_group,
        expected_per_domain_total=args.expected_per_domain_total,
        expected_stage1_per_domain=args.expected_stage1_per_domain,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("ROBOTWIN_PSEUDO_BUDGET_OK")


if __name__ == "__main__":
    main()
