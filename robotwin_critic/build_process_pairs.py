from __future__ import annotations

import argparse
import random
from pathlib import Path

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, entry_ref, read_jsonl, split_train_val, write_jsonl


def _sample_pair(length: int, delta: int, label: int, rng: random.Random) -> tuple[int, int]:
    if label == 0:
        i = rng.randrange(0, length)
        lo = max(0, i - delta)
        hi = min(length - 1, i + delta)
        j = rng.randrange(lo, hi + 1)
        return i, j

    min_gap = delta + 1
    if length <= min_gap:
        return 0, length - 1
    i = rng.randrange(0, length - min_gap)
    j = rng.randrange(i + min_gap, length)
    if label > 0:
        return i, j
    return j, i


def build_pairs(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(args.index)
    if args.max_tasks:
        task_dirs = sorted({r["task_dir"] for r in rows})[: args.max_tasks]
        rows = [r for r in rows if r["task_dir"] in set(task_dirs)]
    if args.max_episodes_per_task:
        by_task: dict[str, int] = {}
        kept = []
        for row in rows:
            n = by_task.get(row["task_dir"], 0)
            if n < args.max_episodes_per_task:
                kept.append(row)
                by_task[row["task_dir"]] = n + 1
        rows = kept

    rng = random.Random(args.seed)
    pairs: list[dict] = []
    labels = [1, -1, 0]
    for row in rows:
        length = int(row["length"])
        if length <= args.delta + 2:
            continue
        for n in range(args.pairs_per_episode):
            label = labels[n % len(labels)]
            frame_i, frame_j = _sample_pair(length, args.delta, label, rng)
            pair = {
                "id": f'{row["task_name"]}:{row["dataset_split"]}:{row["episode_index"]}:{n}',
                "label": label,
                "frame_i": frame_i,
                "frame_j": frame_j,
                "final_frame": length - 1,
            }
            pair.update(entry_ref(row))
            pairs.append(pair)

    train, val = split_train_val(pairs, args.val_fraction, args.seed)
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.train_output, train)
    write_jsonl(args.val_output, val)
    print(
        {
            "train": len(train),
            "val": len(val),
            "delta": args.delta,
            "pairs_per_episode": args.pairs_per_episode,
        }
    )
    return train, val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build process-value pair data.")
    parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT_ROOT / "index.jsonl")
    parser.add_argument("--train-output", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train.jsonl")
    parser.add_argument("--val-output", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_val.jsonl")
    parser.add_argument("--pairs-per-episode", type=int, default=12)
    parser.add_argument("--delta", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    build_pairs(parse_args())


if __name__ == "__main__":
    main()

