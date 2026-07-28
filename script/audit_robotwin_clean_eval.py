#!/usr/bin/env python3
"""Strictly audit a 50-task RoboTwin Easy/Clean evaluation."""

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed-cache", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--tasks",
        help="Comma-separated task list. Defaults to the ordered seed-cache tasks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-sr", type=float)
    return parser.parse_args()


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main():
    args = parse_args()
    cache_payload = json.loads(args.seed_cache.read_text(encoding="utf-8"))
    if cache_payload.get("task_config") != "demo_clean":
        raise SystemExit("Seed cache is not demo_clean")
    cache = cache_payload["tasks"]
    tasks = (
        [task.strip() for task in args.tasks.split(",") if task.strip()]
        if args.tasks
        else list(cache)
    )
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise SystemExit(f"Expected 50 unique tasks, got {len(tasks)}")
    if set(tasks) != set(cache):
        raise SystemExit("Task list does not exactly match the 50-task seed cache")

    seed_root = (
        args.results_root
        / "demo_clean"
        / args.label
        / "stseed-10000"
    )
    per_task = []
    timings = []

    for task in tasks:
        res_path = seed_root / "metrics" / task / "res.json"
        timing_path = seed_root / "eval_timing" / f"{task}.jsonl"
        if not res_path.is_file() or not timing_path.is_file():
            raise SystemExit(f"Missing result files for {task}")
        res = json.loads(res_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in timing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if int(res["total_num"]) != args.episodes:
            raise SystemExit(f"{task}: res total={res['total_num']}")
        if len(records) != args.episodes:
            raise SystemExit(f"{task}: timing rows={len(records)}")
        episode_ids = [int(row["episode"]) for row in records]
        if episode_ids != list(range(1, args.episodes + 1)):
            raise SystemExit(f"{task}: non-contiguous episodes {episode_ids}")

        observed = [int(row["seed"]) for row in records]
        if len(observed) != len(set(observed)):
            raise SystemExit(f"{task}: duplicate seeds {observed}")
        cached = [int(row["seed"]) for row in cache[task]]
        cursor = 0
        for seed in observed:
            try:
                cursor = cached.index(seed, cursor) + 1
            except ValueError as error:
                raise SystemExit(
                    f"{task}: seed {seed} is not an ordered cache subsequence"
                ) from error

        successes = sum(bool(row["success"]) for row in records)
        if int(res["succ_num"]) != successes:
            raise SystemExit(
                f"{task}: res successes={res['succ_num']} timing={successes}"
            )
        task_timings = [float(row["total_sec"]) for row in records]
        timings.extend(task_timings)
        per_task.append(
            {
                "task": task,
                "successes": successes,
                "total": args.episodes,
                "sr": successes / args.episodes,
                "timing_mean_s": statistics.mean(task_timings),
            }
        )

    successes = sum(row["successes"] for row in per_task)
    total = len(tasks) * args.episodes
    summary = {
        "task_config": "demo_clean",
        "tasks": len(tasks),
        "episodes_per_task": args.episodes,
        "episodes": total,
        "successes": successes,
        "sr": successes / total,
        "timing_mean_s": statistics.mean(timings),
        "timing_median_s": statistics.median(timings),
        "timing_p95_s": percentile(timings, 0.95),
        "serial_episode_hours": sum(timings) / 3600,
        "per_task": per_task,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"AUDIT_OK tasks={len(tasks)} episodes={total} successes={successes} "
        f"sr={summary['sr']:.4f}"
    )
    if args.min_sr is not None and summary["sr"] < args.min_sr:
        raise SystemExit(
            f"SR {summary['sr']:.4f} is below required {args.min_sr:.4f}"
        )


if __name__ == "__main__":
    main()
