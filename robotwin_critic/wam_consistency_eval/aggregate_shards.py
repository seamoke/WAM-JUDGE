"""Aggregate independently decoded WAM consistency metric shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate_debug_run import PairRecord, _median_gap, _rank_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_records(input_dirs: list[str]) -> list[PairRecord]:
    records: dict[tuple[Any, ...], PairRecord] = {}
    for input_dir in input_dirs:
        for path in sorted(Path(input_dir).glob("**/frame_metrics.csv")):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    feature = row.get("feature_cosine")
                    record = PairRecord(
                        run=row["run"],
                        chunk_start=int(row["chunk_start"]),
                        pred_index=int(row["pred_index"]),
                        real_index=int(row["real_index"]),
                        pair_type=row["pair_type"],
                        is_match=int(row["is_match"]),
                        mae=float(row["mae"]),
                        psnr=float(row["psnr"]),
                        ssim=float(row["ssim"]),
                        feature_cosine=float(feature) if feature not in (None, "") else None,
                    )
                    key = (
                        record.run,
                        record.chunk_start,
                        record.pred_index,
                        record.real_index,
                        record.pair_type,
                    )
                    records[key] = record
    return list(records.values())


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(args.input_dir)
    if not records:
        raise FileNotFoundError("No frame_metrics.csv shards found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].as_dict()))
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)

    labels = [record.is_match for record in records]
    metric_specs = {"mae": -1.0, "psnr": 1.0, "ssim": 1.0}
    if any(record.feature_cosine is not None for record in records):
        metric_specs["feature_cosine"] = 1.0
    metrics: dict[str, Any] = {}
    for field, direction in metric_specs.items():
        values = [getattr(record, field) for record in records]
        valid = [(label, value) for label, value in zip(labels, values) if value is not None]
        metrics[field] = {
            "auc_matched_vs_mismatch": _rank_auc(
                [label for label, _ in valid],
                [direction * float(value) for _, value in valid],
            ),
            "matched_minus_mismatch_median": _median_gap(records, field),
            "matched_median": float(
                np.median([float(value) for label, value in valid if label])
            ),
            "mismatch_median": float(
                np.median([float(value) for label, value in valid if not label])
            ),
        }

    matched = int(sum(labels))
    semantic = metrics.get("feature_cosine")
    if matched < 30:
        recommendation = "collect_more_aligned_chunks"
    elif semantic and semantic["auc_matched_vs_mismatch"] >= 0.75:
        recommendation = "consistency_filter_has_discriminative_signal"
    elif metrics["ssim"]["auc_matched_vs_mismatch"] >= 0.75:
        recommendation = "structural_signal_only_add_semantic_encoder_before_filtering"
    else:
        recommendation = "current_distances_do_not_support_a_reliable_filter"
    summary = {
        "input_dirs": args.input_dir,
        "chunk_count": len({(record.run, record.chunk_start) for record in records}),
        "matched_frames": matched,
        "mismatch_pairs": len(records) - matched,
        "metrics": metrics,
        "recommendation": recommendation,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    aggregate(parse_args())


if __name__ == "__main__":
    main()
