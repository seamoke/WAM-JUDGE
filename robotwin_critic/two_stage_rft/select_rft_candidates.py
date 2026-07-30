"""Filter and rank WAM candidates with consistency, process, and action rewards."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def context_key(row: dict) -> tuple:
    return (
        row["task"],
        row.get("source_repo"),
        row.get("source_episode_index", row.get("episode_index")),
        row["start_frame"],
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--process-temperature",
        type=float,
        default=1.0,
        help="Temperature for sigmoid-normalizing the raw process delta.",
    )
    parser.add_argument("--min-process-score", type=float, default=0.0)
    parser.add_argument("--min-action-score", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1]")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.process_temperature <= 0:
        raise ValueError("--process-temperature must be positive")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    total = consistency_rejected = reward_rejected = 0
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            consistency = row.get("consistency", row.get("consistency_filter", {}))
            if not bool(
                consistency.get("accepted", row.get("consistency_accepted", False))
            ):
                consistency_rejected += 1
                continue
            process = float(row["process_score"])
            action = row["action_critic"]
            action_score = float(action["action_score"])
            if (
                process < args.min_process_score
                or not bool(action["accepted"])
                or action_score < args.min_action_score
            ):
                reward_rejected += 1
                continue
            process_reward = sigmoid(process / args.process_temperature)
            row["process_reward"] = process_reward
            row["combined_reward"] = (
                args.alpha * process_reward
                + (1.0 - args.alpha) * action_score
            )
            grouped[context_key(row)].append(row)

    selected = []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: row["combined_reward"], reverse=True)
        selected.extend(candidates[: args.top_k])
    selected.sort(key=lambda row: (context_key(row), -row["combined_reward"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "total": total,
                "consistency_rejected": consistency_rejected,
                "reward_rejected": reward_rejected,
                "contexts": len(grouped),
                "selected": len(selected),
                "alpha": args.alpha,
                "process_temperature": args.process_temperature,
                "top_k": args.top_k,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    print("RFT_CANDIDATE_SELECTION_OK")


if __name__ == "__main__":
    main()
