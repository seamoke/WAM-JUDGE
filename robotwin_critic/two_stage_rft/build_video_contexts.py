"""Build action-hidden middle-state contexts from fixed Stage-2 trajectories."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from robotwin_critic.two_stage_rft.data_access import (
    episode_metadata,
    episode_video_paths,
    find_parquet,
    instruction_from_episode,
    latent_segment_exists,
    latent_segment_num_frames,
    read_non_action_rows,
    select_proprio_column,
    verified_eef_state_indices,
)
from robotwin_critic.two_stage_rft.protocol import iter_episode_refs, sha256_file


DEFAULT_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def history_indices(frame: int, length: int, history_frames: int) -> list[int]:
    start = max(0, frame - history_frames + 1)
    indices = list(range(start, frame + 1))
    return [indices[0]] * (history_frames - len(indices)) + indices


def episode_transition_chunks(
    episode: dict,
    repo: Path,
    max_episode_frames: int,
) -> int:
    chunks = 0
    episode_index = int(episode["episode_index"])
    for config in episode.get("action_config", []):
        start = int(config["start_frame"])
        end = int(config["end_frame"])
        if end <= start:
            continue
        if max_episode_frames and end - start > max_episode_frames:
            continue
        if not latent_segment_exists(repo, episode_index, start, end):
            continue
        latent_frames = latent_segment_num_frames(repo, episode_index, start, end)
        chunks += max(latent_frames - 1, 0)
    return chunks


def coverage_frames(
    length: int, fractions: tuple[float, ...], minimum_count: int
) -> list[int]:
    """Keep requested progress anchors and add uniform in-between contexts."""
    maximum_frame = length - 2
    anchors = {
        min(maximum_frame, max(0, round(fraction * (length - 1))))
        for fraction in fractions
    }
    target = min(max(len(anchors), minimum_count), maximum_frame + 1)
    frames = set(anchors)
    candidates = [
        round((index + 0.5) * maximum_frame / max(target, 1))
        for index in range(target)
    ]
    candidates.extend(range(maximum_frame + 1))
    for frame in candidates:
        frames.add(min(maximum_frame, max(0, frame)))
        if len(frames) >= target:
            break
    return sorted(frames)


def build_contexts(
    prepared_root: Path,
    output: Path,
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    history_frames: int = 4,
    eef_proprio_indices: tuple[int, ...] = (),
    max_episode_frames: int = 500,
    context_pool_multiplier: float = 2.0,
    require_eef_state: bool = True,
    max_tasks: int = 0,
    max_episodes_per_group: int = 0,
    expected_per_domain_total: int = 50,
    expected_stage1_per_domain: int = 30,
) -> dict:
    if history_frames < 2 or history_frames > 4:
        raise ValueError("history_frames must be in [2,4]")
    if any(not 0.0 < fraction < 1.0 for fraction in fractions):
        raise ValueError("All progress fractions must be in (0,1)")
    if eef_proprio_indices and len(eef_proprio_indices) != 16:
        raise ValueError("eef_proprio_indices must contain exactly 16 indices")
    if context_pool_multiplier < 1.0:
        raise ValueError("context_pool_multiplier must be at least 1")
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
    metadata_cache: dict[Path, dict[int, dict]] = {}
    split_manifest_sha256 = sha256_file(prepared_root / "split_manifest.json")
    rows = []
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
        length = int(episode["length"])
        if length < history_frames + 2:
            continue
        parquet = find_parquet(ref.repo, ref.output_episode_index)
        video_paths = episode_video_paths(
            ref.repo, parquet, ref.output_episode_index
        )
        proprio_column = select_proprio_column(parquet)
        required_contexts = episode_transition_chunks(
            episode, ref.repo, max_episode_frames
        )
        context_pool_target = math.ceil(
            required_contexts * context_pool_multiplier
        )
        sampled_frames = coverage_frames(length, fractions, context_pool_target)
        proprios = read_non_action_rows(
            parquet, sampled_frames, column=proprio_column
        )
        verified_indices = verified_eef_state_indices(ref.repo, proprio_column)
        selected_eef_indices = eef_proprio_indices or verified_indices
        if require_eef_state and not selected_eef_indices:
            raise ValueError(
                f"{ref.repo}: Stage-2 requires an explicit or metadata-verified "
                "16-D EEF observation.state mapping"
            )
        mapping_mode = (
            "explicit_indices"
            if eef_proprio_indices
            else "metadata_verified"
            if verified_indices
            else None
        )
        for frame, proprio in zip(sampled_frames, proprios):
            history = history_indices(frame, length, history_frames)
            start_state = None
            if selected_eef_indices:
                if proprio is None or max(selected_eef_indices) >= len(proprio):
                    raise ValueError(
                        f"{parquet}: proprio cannot satisfy EEF index mapping"
                    )
                start_state = [proprio[index] for index in selected_eef_indices]
            rows.append(
                {
                    "schema_version": 1,
                    "context_id": (
                        f"{ref.task}/{ref.domain}/"
                        f"{ref.source_episode_index}/{frame}"
                    ),
                    "task": ref.task,
                    "source_task": ref.task,
                    "domain": ref.domain,
                    "stage": "stage2_video_only",
                    "source_stage": "stage2",
                    "source_episode_index": ref.source_episode_index,
                    "output_episode_index": ref.output_episode_index,
                    "length": length,
                    "progress_fraction": frame / max(length - 1, 1),
                    "frame_index": frame,
                    "history_frame_indices": history,
                    "history_frames": history_frames,
                    "video_paths": video_paths,
                    "proprio_column": proprio_column,
                    "proprio": proprio,
                    "start_state": start_state,
                    "start_state_semantics": (
                        f"{mapping_mode}_eef_proprio_mapping"
                        if start_state is not None
                        else None
                    ),
                    "episode_transition_chunks": required_contexts,
                    "episode_context_pool_target": context_pool_target,
                    "split_manifest_sha256": split_manifest_sha256,
                    "text": instruction_from_episode(episode, ref.task),
                    "source_repo": str(ref.repo.resolve()),
                    "source_parquet": str(parquet.resolve()),
                    "action_label_present": False,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter((row["task"], row["domain"]) for row in rows)
    summary = {
        "schema_version": 1,
        "prepared_root": str(prepared_root.resolve()),
        "split_manifest_sha256": split_manifest_sha256,
        "stage": "stage2_video_only",
        "reads_action_column": False,
        "history_frames": history_frames,
        "eef_proprio_indices": list(eef_proprio_indices),
        "automatic_eef_mapping": "requires exact 16-D named metadata",
        "requires_eef_state": require_eef_state,
        "pseudo_chunk_latent_frames": 2,
        "context_pool_multiplier": context_pool_multiplier,
        "max_episode_frames": max_episode_frames,
        "protocol_expected_per_domain_total": expected_per_domain_total,
        "protocol_expected_stage1_per_domain": expected_stage1_per_domain,
        "fractions": list(fractions),
        "contexts": len(rows),
        "groups": {
            f"{task}/{domain}": count
            for (task, domain), count in sorted(counts.items())
        },
        "output": str(output.resolve()),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument(
        "--eef-proprio-indices",
        nargs=16,
        type=int,
        default=(),
        help="Optional explicit 16-column EEF mapping; never inferred automatically.",
    )
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-group", type=int, default=0)
    parser.add_argument("--max-episode-frames", type=int, default=500)
    parser.add_argument("--context-pool-multiplier", type=float, default=2.0)
    parser.add_argument("--allow-missing-eef-state", action="store_true")
    parser.add_argument("--expected-per-domain-total", type=int, default=50)
    parser.add_argument("--expected-stage1-per-domain", type=int, default=30)
    args = parser.parse_args()
    summary = build_contexts(
        args.prepared_root,
        args.output,
        fractions=tuple(args.fractions),
        history_frames=args.history_frames,
        eef_proprio_indices=tuple(args.eef_proprio_indices),
        max_episode_frames=args.max_episode_frames,
        context_pool_multiplier=args.context_pool_multiplier,
        require_eef_state=not args.allow_missing_eef_state,
        max_tasks=args.max_tasks,
        max_episodes_per_group=args.max_episodes_per_group,
        expected_per_domain_total=args.expected_per_domain_total,
        expected_stage1_per_domain=args.expected_stage1_per_domain,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ROBOTWIN_VIDEO_CONTEXTS_OK")


if __name__ == "__main__":
    main()
