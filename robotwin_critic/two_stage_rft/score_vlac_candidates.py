"""Append VLAC process rewards to generated WAM candidate records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from robotwin_critic.vlac_finetune.evaluate_vlac import (
    infer_scores,
    load_critic,
)
from robotwin_critic.vlac_finetune.common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--action-accepted-only",
        action="store_true",
        help="Skip candidates rejected by the kinematic gate before VLAC inference.",
    )
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    if args.action_accepted_only:
        rows = [
            row
            for row in rows
            if bool(row.get("action_critic", {}).get("accepted", False))
        ]
    critic = load_critic(
        SimpleNamespace(
            model=str(args.model),
            adapter=None if args.adapter is None else str(args.adapter),
            device=args.device,
        )
    )
    queries = [
        (row["text"], row["current_image"], row["generated_image"])
        for row in rows
    ]
    scores, answers, parsed, seconds = infer_scores(
        critic, queries, args.batch_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, score, answer, ok in zip(rows, scores, answers, parsed):
            row["process_score"] = float(score)
            row["process_critic"] = {
                "model": str(args.model.resolve()),
                "adapter": None if args.adapter is None else str(args.adapter.resolve()),
                "numeric_parsed": bool(ok),
                "raw_answer": answer,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "candidates": len(rows),
        "numeric_parse_rate": (
            sum(bool(value) for value in parsed) / len(parsed) if parsed else 0.0
        ),
        "inference_seconds": seconds,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ROBOTWIN_VLAC_CANDIDATE_SCORING_OK")


if __name__ == "__main__":
    main()
