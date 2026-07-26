from __future__ import annotations

import argparse
from pathlib import Path

import torch

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, entry_ref, read_jsonl


def build_goal_bank(args: argparse.Namespace) -> dict:
    rows = read_jsonl(args.index)
    bank: dict[str, list[dict]] = {}
    for row in rows:
        refs = bank.setdefault(row["task_name"], [])
        if len(refs) >= args.max_goals_per_task:
            continue
        ref = entry_ref(row)
        ref["final_frame"] = int(row["length"]) - 1
        refs.append(ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "robotwin_critic_goal_bank_v1",
        "index": str(args.index),
        "goals": bank,
    }
    torch.save(payload, args.output)
    print({"tasks": len(bank), "goals": sum(len(v) for v in bank.values()), "output": str(args.output)})
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-level RoboTwin final-state goal bank.")
    parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT_ROOT / "index.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "goal_bank.pt")
    parser.add_argument("--max-goals-per-task", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    build_goal_bank(parse_args())


if __name__ == "__main__":
    main()

