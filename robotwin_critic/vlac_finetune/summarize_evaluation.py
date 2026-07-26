"""Rebuild VLAC evaluation summaries from saved predictions without GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl
from .evaluate_vlac import summarize_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neutral-threshold", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    with summary_path.open() as handle:
        summary = json.load(handle)

    record_summary, per_task_metrics = summarize_prediction_records(
        read_jsonl(output_dir / "pair_predictions.jsonl"),
        read_jsonl(output_dir / "trajectory_predictions.jsonl"),
        args.neutral_threshold,
    )
    summary.update(record_summary)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "per_task_metrics.json").open("w") as handle:
        json.dump(per_task_metrics, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
