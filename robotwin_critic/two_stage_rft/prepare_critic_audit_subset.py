"""Materialize a reproducible candidate subset for corrected critic rescoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.shards <= 0:
        raise ValueError("--shards must be positive")

    requested = [
        str(row["candidate_id"])
        for row in json.loads(args.samples.read_text(encoding="utf-8"))
    ]
    requested = list(dict.fromkeys(requested))
    requested_set = set(requested)
    action_by_id = {}
    dual_by_id = {}
    for collect_dir in sorted(args.collect_root.glob("collect_*")):
        action_path = collect_dir / "action_scored_audit.jsonl"
        dual_path = collect_dir / "dual_scored.jsonl"
        if not action_path.is_file() or not dual_path.is_file():
            continue
        for row in read_jsonl(action_path):
            if row["candidate_id"] in requested_set:
                action_by_id[row["candidate_id"]] = row["action_critic"]
        for row in read_jsonl(dual_path):
            if row["candidate_id"] in requested_set:
                dual_by_id[row["candidate_id"]] = row

    missing = [value for value in requested if value not in dual_by_id or value not in action_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested candidates: {missing[:3]}")
    rows = []
    for candidate_id in requested:
        row = dual_by_id[candidate_id]
        row["original_process_score"] = row.get("process_score")
        row["original_process_critic"] = row.get("process_critic")
        row["action_critic"] = action_by_id[candidate_id]
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "candidates.jsonl", rows)
    for index in range(args.shards):
        write_jsonl(args.output_dir / f"shard_{index:02d}.jsonl", rows[index:: args.shards])
    print(
        json.dumps(
            {
                "requested_records": len(requested),
                "unique_candidates": len(rows),
                "shards": args.shards,
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
