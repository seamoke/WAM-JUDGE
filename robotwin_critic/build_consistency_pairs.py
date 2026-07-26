from __future__ import annotations

import argparse
import random
from pathlib import Path

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, entry_ref, read_jsonl, split_train_val, write_jsonl

NEGATIVE_TYPES = (
    "action_mismatch",
    "future_mismatch",
    "time_reversal",
    "cross_task_mismatch",
    "offset_mismatch",
)


def _choose_other(rows: list[dict], rng: random.Random, task_name: str | None = None, different_task: bool = False) -> dict:
    candidates = rows
    if task_name is not None:
        if different_task:
            candidates = [r for r in rows if r["task_name"] != task_name]
        else:
            candidates = [r for r in rows if r["task_name"] == task_name]
    if not candidates:
        candidates = rows
    return rng.choice(candidates)


def _frame_pair(length: int, horizon: int, rng: random.Random) -> tuple[int, int]:
    if length <= horizon + 1:
        return 0, max(0, length - 1)
    start = rng.randrange(0, length - horizon)
    return start, start + horizon


def build_consistency_pairs(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
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
    for row in rows:
        length = int(row["length"])
        if length <= args.horizon + 1:
            continue
        for n in range(args.samples_per_episode):
            state_frame, future_frame = _frame_pair(length, args.horizon, rng)
            base = {
                "id": f'{row["task_name"]}:{row["dataset_split"]}:{row["episode_index"]}:{n}',
                "state_frame": state_frame,
                "future_frame": future_frame,
                "action_frame": state_frame,
                "horizon": args.horizon,
                "label": 1,
                "negative_type": "positive",
            }
            base.update(entry_ref(row))
            base["future_ref"] = entry_ref(row)
            base["action_ref"] = entry_ref(row)
            pairs.append(base)

            for neg_type in NEGATIVE_TYPES:
                neg = dict(base)
                neg["id"] = base["id"] + f":{neg_type}"
                neg["label"] = 0
                neg["negative_type"] = neg_type
                if neg_type == "action_mismatch":
                    other = _choose_other(rows, rng, row["task_name"])
                    neg["action_ref"] = entry_ref(other)
                    neg["action_frame"] = min(state_frame, int(other["length"]) - 1)
                elif neg_type == "future_mismatch":
                    other = _choose_other(rows, rng, row["task_name"])
                    neg["future_ref"] = entry_ref(other)
                    neg["future_frame"] = min(future_frame, int(other["length"]) - 1)
                elif neg_type == "time_reversal":
                    neg["state_frame"], neg["future_frame"] = future_frame, state_frame
                elif neg_type == "cross_task_mismatch":
                    other = _choose_other(rows, rng, row["task_name"], different_task=True)
                    neg["future_ref"] = entry_ref(other)
                    neg["action_ref"] = entry_ref(other)
                    neg["future_frame"] = min(future_frame, int(other["length"]) - 1)
                    neg["action_frame"] = min(state_frame, int(other["length"]) - 1)
                elif neg_type == "offset_mismatch":
                    offset = args.horizon * 2
                    neg["action_frame"] = min(max(0, state_frame + offset), length - 1)
                pairs.append(neg)

    train, val = split_train_val(pairs, args.val_fraction, args.seed)
    write_jsonl(args.train_output, train)
    write_jsonl(args.val_output, val)
    print({"train": len(train), "val": len(val), "horizon": args.horizon})
    return train, val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build action-video consistency data.")
    parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT_ROOT / "index.jsonl")
    parser.add_argument("--train-output", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_pairs_train.jsonl")
    parser.add_argument("--val-output", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_pairs_val.jsonl")
    parser.add_argument(
        "--samples-per-episode",
        type=int,
        default=1,
        help="Number of start frames per episode. Each start frame emits one positive and all negative types.",
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--max-episodes-per-task", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    build_consistency_pairs(parse_args())


if __name__ == "__main__":
    main()
