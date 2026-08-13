"""Build a balanced, quality-gated one-shot pseudo buffer from collect rounds."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

from robotwin_critic.two_stage_rft.rollout_provenance import with_rollout_provenance


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def group_key(row: dict) -> str:
    return f"{row['task']}/{row['domain']}"


def source_context_key(row: dict) -> str:
    value = str(row.get("source_context_id") or row["context_id"])
    return value.split("@e", 1)[0]


def progress_bin(row: dict, bins: int) -> int:
    value = min(max(float(row.get("progress_fraction", 0.0)), 0.0), 1.0)
    return min(int(value * bins), bins - 1)


def proportional_quotas(weights: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0:
        raise ValueError("target must be positive")
    total = sum(int(value) for value in weights.values())
    if total <= 0:
        raise ValueError("group weights must have positive total")
    exact = {key: target * int(value) / total for key, value in weights.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise AssertionError("quota allocation did not preserve target")
    return quotas


def kinematic_vector(row: dict) -> list[float] | None:
    diagnostics = row.get("action_critic", {}).get("diagnostics", {})
    values = []
    for name in sorted(diagnostics):
        item = diagnostics[name]
        maximum = float(item.get("maximum", 0.0))
        hard = float(item.get("hard", 0.0))
        values.append(min(max(maximum / hard, 0.0), 4.0) if hard > 0 else 0.0)
    return values or None


def rms_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def load_visual_cache(path: Path) -> dict[str, dict]:
    return {
        str(row["path"]): row
        for row in read_jsonl(path)
        if isinstance(row.get("path"), str)
    }


def inspect_image(path: str) -> dict:
    from PIL import Image, ImageStat

    image_path = Path(path)
    result = {"path": str(image_path), "exists": image_path.is_file()}
    if not result["exists"]:
        return result
    with Image.open(image_path) as image:
        gray = image.convert("L")
        stat = ImageStat.Stat(gray)
        histogram = gray.histogram()
        pixels = max(sum(histogram), 1)
        result.update(
            {
                "mean_luma": float(stat.mean[0]),
                "std_luma": float(stat.stddev[0]),
                "dark_fraction": float(sum(histogram[:8]) / pixels),
                "width": int(gray.width),
                "height": int(gray.height),
            }
        )
    return result


def image_is_valid(
    info: dict,
    *,
    min_mean_luma: float,
    min_std_luma: float,
    max_dark_fraction: float,
) -> bool:
    return bool(info.get("exists")) and (
        float(info.get("mean_luma", 0.0)) >= min_mean_luma
        and float(info.get("std_luma", 0.0)) >= min_std_luma
        and float(info.get("dark_fraction", 1.0)) <= max_dark_fraction
    )


def collect_rows(collect_root: Path) -> list[dict]:
    rows = []
    for path in sorted(collect_root.glob("collect_*/selected_winners.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def finalize_buffer(
    rows: list[dict],
    group_weights: dict[str, int],
    *,
    target: int,
    visual_cache_path: Path,
    max_per_context: int = 4,
    max_per_episode: int = 16,
    progress_bins: int = 5,
    min_action_distance: float = 0.03,
    min_mean_luma: float = 8.0,
    min_std_luma: float = 4.0,
    max_dark_fraction: float = 0.98,
) -> tuple[list[dict], dict]:
    if max_per_context <= 0 or max_per_episode <= 0 or progress_bins <= 0:
        raise ValueError("diversity limits must be positive")
    quotas = proportional_quotas(group_weights, target)
    cache = load_visual_cache(visual_cache_path)
    deduplicated = {}
    for raw_row in rows:
        row = with_rollout_provenance(raw_row)
        deduplicated[str(row["candidate_id"])] = row

    visual_rejected = 0
    valid_rows = []
    for row in deduplicated.values():
        image_path = str(row.get("generated_image", ""))
        if image_path not in cache:
            try:
                cache[image_path] = inspect_image(image_path)
            except Exception as error:
                cache[image_path] = {
                    "path": image_path,
                    "exists": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        if not image_is_valid(
            cache[image_path],
            min_mean_luma=min_mean_luma,
            min_std_luma=min_std_luma,
            max_dark_fraction=max_dark_fraction,
        ):
            visual_rejected += 1
            continue
        valid_rows.append(row)
    atomic_write_jsonl(visual_cache_path, list(cache.values()))

    by_context: dict[str, list[dict]] = defaultdict(list)
    for row in valid_rows:
        if group_key(row) in quotas:
            by_context[source_context_key(row)].append(row)

    action_duplicate_rejected = 0
    context_limited = []
    for context in sorted(by_context):
        ranked = sorted(
            by_context[context],
            key=lambda row: (
                float(row.get("process_score", float("-inf"))),
                float(row.get("action_critic", {}).get("action_score", 0.0)),
                str(row["candidate_id"]),
            ),
            reverse=True,
        )
        kept_vectors: list[list[float]] = []
        kept = 0
        for row in ranked:
            vector = kinematic_vector(row)
            if vector is not None and any(
                rms_distance(vector, prior) < min_action_distance
                for prior in kept_vectors
            ):
                action_duplicate_rejected += 1
                continue
            context_limited.append(row)
            if vector is not None:
                kept_vectors.append(vector)
            kept += 1
            if kept >= max_per_context:
                break

    by_group_stratum: dict[str, dict[tuple[int, int], deque]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for row in context_limited:
        stratum = (
            progress_bin(row, progress_bins),
            int(row.get("source_episode_index", -1)),
        )
        by_group_stratum[group_key(row)][stratum].append(row)
    for strata in by_group_stratum.values():
        for key, values in list(strata.items()):
            strata[key] = deque(
                sorted(
                    values,
                    key=lambda row: (
                        float(row.get("process_score", float("-inf"))),
                        float(row.get("action_critic", {}).get("action_score", 0.0)),
                        str(row["candidate_id"]),
                    ),
                    reverse=True,
                )
            )

    selected = []
    quota_shortfalls = {}
    for group in sorted(quotas):
        quota = quotas[group]
        strata = by_group_stratum.get(group, {})
        episode_counts: Counter[int] = Counter()
        group_selected = []
        keys_by_bin: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key in sorted(strata):
            keys_by_bin[key[0]].append(key)
        active = []
        for offset in range(max((len(keys) for keys in keys_by_bin.values()), default=0)):
            for bin_index in sorted(keys_by_bin):
                keys = keys_by_bin[bin_index]
                if offset < len(keys):
                    active.append(keys[offset])
        while active and len(group_selected) < quota:
            next_active = []
            progressed = False
            for key in active:
                queue = strata[key]
                while queue and episode_counts[key[1]] >= max_per_episode:
                    queue.popleft()
                if queue and len(group_selected) < quota:
                    row = queue.popleft()
                    group_selected.append(row)
                    episode_counts[key[1]] += 1
                    progressed = True
                if queue:
                    next_active.append(key)
            if not progressed:
                break
            active = next_active
        if len(group_selected) < quota:
            quota_shortfalls[group] = {
                "quota": quota,
                "selected": len(group_selected),
                "missing": quota - len(group_selected),
            }
        selected.extend(group_selected)

    # A hard quota can make collection non-terminating when one difficult task
    # rarely clears the critics. Preserve proportional quotas as the first pass,
    # then round-robin surplus from other groups while retaining all per-context
    # and per-episode limits.
    selected_ids = {str(row["candidate_id"]) for row in selected}
    episode_counts = Counter(
        (
            group_key(row),
            int(row.get("source_episode_index", -1)),
        )
        for row in selected
    )
    surplus_by_group: dict[str, deque] = {}
    for group in sorted(quotas):
        surplus_by_group[group] = deque(
            sorted(
                (
                    row
                    for row in context_limited
                    if group_key(row) == group
                    and str(row["candidate_id"]) not in selected_ids
                ),
                key=lambda row: (
                    float(row.get("process_score", float("-inf"))),
                    float(row.get("action_critic", {}).get("action_score", 0.0)),
                    str(row["candidate_id"]),
                ),
                reverse=True,
            )
        )
    while len(selected) < target:
        progressed = False
        for group in sorted(surplus_by_group):
            queue = surplus_by_group[group]
            while queue:
                row = queue.popleft()
                episode_key = (
                    group,
                    int(row.get("source_episode_index", -1)),
                )
                if episode_counts[episode_key] >= max_per_episode:
                    continue
                selected.append(row)
                selected_ids.add(str(row["candidate_id"]))
                episode_counts[episode_key] += 1
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break

    selected = selected[:target]
    selected = [
        {
            **row,
            "rft_selection": {
                **row.get("rft_selection", {}),
                "one_shot": True,
                "diversity_balanced": True,
                "source_context_id": source_context_key(row),
                "group_quota": quotas[group_key(row)],
                "max_per_context": max_per_context,
                "max_per_episode": max_per_episode,
                "progress_bins": progress_bins,
                "min_action_distance": min_action_distance,
                "visual_gate": {
                    "min_mean_luma": min_mean_luma,
                    "min_std_luma": min_std_luma,
                    "max_dark_fraction": max_dark_fraction,
                },
            },
        }
        for row in selected
    ]
    selected_groups = Counter(group_key(row) for row in selected)
    selected_progress = Counter(progress_bin(row, progress_bins) for row in selected)
    selected_tasks = Counter(str(row["source_task"]) for row in selected)
    selected_stages = Counter(str(row["source_stage"]) for row in selected)
    quota_overfill = {
        group: selected_groups[group] - quota
        for group, quota in quotas.items()
        if selected_groups[group] > quota
    }
    summary = {
        "schema_version": 1,
        "target": target,
        "ready": len(selected) == target,
        "selected": len(selected),
        "unique_candidates_seen": len(deduplicated),
        "visual_valid": len(valid_rows),
        "visual_rejected": visual_rejected,
        "action_duplicate_rejected": action_duplicate_rejected,
        "unique_contexts": len({source_context_key(row) for row in selected}),
        "unique_episodes": len(
            {
                (row["task"], row["domain"], row.get("source_episode_index"))
                for row in selected
            }
        ),
        "group_quotas": quotas,
        "selected_by_group": dict(sorted(selected_groups.items())),
        "selected_by_task": dict(sorted(selected_tasks.items())),
        "selected_by_source_stage": dict(sorted(selected_stages.items())),
        "selected_by_progress_bin": {
            str(key): value for key, value in sorted(selected_progress.items())
        },
        "quota_shortfalls_before_backfill": quota_shortfalls,
        "quota_overfill_after_backfill": quota_overfill,
        "constraints": {
            "max_per_context": max_per_context,
            "max_per_episode": max_per_episode,
            "progress_bins": progress_bins,
            "min_action_distance": min_action_distance,
            "min_mean_luma": min_mean_luma,
            "min_std_luma": min_std_luma,
            "max_dark_fraction": max_dark_fraction,
        },
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--target", type=int, default=25_000)
    parser.add_argument("--max-per-context", type=int, default=4)
    parser.add_argument("--max-per-episode", type=int, default=16)
    parser.add_argument("--progress-bins", type=int, default=5)
    parser.add_argument("--min-action-distance", type=float, default=0.03)
    parser.add_argument("--min-mean-luma", type=float, default=8.0)
    parser.add_argument("--min-std-luma", type=float, default=4.0)
    parser.add_argument("--max-dark-fraction", type=float, default=0.98)
    args = parser.parse_args()
    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    selected, summary = finalize_buffer(
        collect_rows(args.collect_root),
        {str(key): int(value) for key, value in budget["groups"].items()},
        target=args.target,
        visual_cache_path=args.visual_cache,
        max_per_context=args.max_per_context,
        max_per_episode=args.max_per_episode,
        progress_bins=args.progress_bins,
        min_action_distance=args.min_action_distance,
        min_mean_luma=args.min_mean_luma,
        min_std_luma=args.min_std_luma,
        max_dark_fraction=args.max_dark_fraction,
    )
    atomic_write_jsonl(args.output, selected)
    atomic_write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ROBOTWIN_ONE_SHOT_BUFFER_READY" if summary["ready"] else "ROBOTWIN_ONE_SHOT_BUFFER_GROWING")


if __name__ == "__main__":
    main()
