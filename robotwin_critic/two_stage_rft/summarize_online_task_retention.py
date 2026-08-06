"""Summarize generated and retained online RFT Q-A pairs by RoboTwin task."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_name(row: dict) -> str:
    return str(
        row.get("task")
        or row.get("task_name")
        or row.get("source_task")
        or "__unknown__"
    )


def summarize_rows(generated: list[dict], retained: list[dict]) -> dict:
    counters = defaultdict(
        lambda: {
            "generated_q_ids": set(),
            "generated_qa_ids": set(),
            "retained_q_ids": set(),
            "retained_qa_ids": set(),
        }
    )
    for label, rows in (("generated", generated), ("retained", retained)):
        for row in rows:
            task = task_name(row)
            context_id = str(row["context_id"])
            candidate_id = str(row["candidate_id"])
            counters[task][f"{label}_q_ids"].add(context_id)
            counters[task][f"{label}_qa_ids"].add(candidate_id)

    output = {}
    for task, values in sorted(counters.items()):
        generated_q = len(values["generated_q_ids"])
        generated_qa = len(values["generated_qa_ids"])
        retained_q = len(values["retained_q_ids"])
        retained_qa = len(values["retained_qa_ids"])
        output[task] = {
            "generated_q": generated_q,
            "generated_qa_pairs": generated_qa,
            "retained_q": retained_q,
            "retained_qa_pairs": retained_qa,
            "q_retention_rate": retained_q / generated_q if generated_q else 0.0,
            "qa_retention_rate": retained_qa / generated_qa if generated_qa else 0.0,
        }
    return output


def summarize_collect_root(collect_root: Path) -> dict:
    generated = []
    retained = []
    history = []
    completed_collects = 0
    for collect_dir in sorted(collect_root.glob("collect_*")):
        scored_path = collect_dir / "dual_scored.jsonl"
        selected_path = collect_dir / "selected_winners.jsonl"
        if not scored_path.is_file() or not selected_path.is_file():
            continue
        collect_generated = read_jsonl(scored_path)
        collect_retained = read_jsonl(selected_path)
        generated.extend(collect_generated)
        retained.extend(collect_retained)
        history.append(
            {
                "collect": collect_dir.name,
                "tasks": summarize_rows(collect_generated, collect_retained),
            }
        )
        completed_collects += 1

    tasks = summarize_rows(generated, retained)
    return {
        "completed_collects": completed_collects,
        "generated_q": sum(row["generated_q"] for row in tasks.values()),
        "generated_qa_pairs": sum(
            row["generated_qa_pairs"] for row in tasks.values()
        ),
        "retained_q": sum(row["retained_q"] for row in tasks.values()),
        "retained_qa_pairs": sum(
            row["retained_qa_pairs"] for row in tasks.values()
        ),
        "tasks": tasks,
        "history": history,
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize_collect_root(args.collect_root)
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "completed_collects": payload["completed_collects"],
                "generated_qa_pairs": payload["generated_qa_pairs"],
                "retained_qa_pairs": payload["retained_qa_pairs"],
                "tasks": len(payload["tasks"]),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
