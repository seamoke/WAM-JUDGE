#!/usr/bin/env python3
"""Aggregate LIBERO eval JSON outputs into CSV / Markdown tables."""
import argparse
import csv
import json
from pathlib import Path

BENCHMARKS = [
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long"),
]


def load_benchmark_rate(result_dir: Path, benchmark: str) -> tuple[float | None, int, int]:
    task_files = sorted(result_dir.glob(f"{benchmark}_*.json"))
    if not task_files:
        return None, 0, 0

    rates = []
    total_succ = 0
    total_num = 0
    for task_file in task_files:
        with open(task_file) as f:
            data = json.load(f)
        rates.append(float(data["succ_rate"]))
        total_succ += int(data["succ_num"])
        total_num += int(data["total_num"])

    avg_rate = sum(rates) / len(rates) if rates else None
    return avg_rate, total_succ, total_num


def collect_checkpoint(result_dir: Path) -> dict:
    row = {"checkpoint": result_dir.name}
    rates = []
    for benchmark, label in BENCHMARKS:
        rate, succ, total = load_benchmark_rate(result_dir, benchmark)
        row[f"{label}_sr"] = rate
        row[f"{label}_succ"] = succ
        row[f"{label}_total"] = total
        if rate is not None:
            rates.append(rate)
    row["avg_sr"] = sum(rates) / len(rates) if rates else None
    return row


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}"


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fieldnames = ["checkpoint"]
    for _, label in BENCHMARKS:
        fieldnames.append(f"{label}_sr")
    fieldnames.append("avg_sr")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "checkpoint": row["checkpoint"],
                    **{f"{label}_sr": format_pct(row.get(f"{label}_sr")) for _, label in BENCHMARKS},
                    "avg_sr": format_pct(row.get("avg_sr")),
                }
            )


def write_markdown(rows: list[dict], out_md: Path) -> None:
    headers = ["Checkpoint"] + [label for _, label in BENCHMARKS] + ["Avg"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [row["checkpoint"]]
        for _, label in BENCHMARKS:
            cells.append(format_pct(row.get(f"{label}_sr")))
        cells.append(format_pct(row.get("avg_sr")))
        lines.append("| " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="/workspace/lingbot-va/train_out/libero/eval_results",
        help="Root directory containing per-checkpoint result folders",
    )
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_csv = Path(args.out_csv or results_root / "results.csv")
    out_md = Path(args.out_md or results_root / "results.md")

    ckpt_dirs = sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint_step_")
    )
    if not ckpt_dirs:
        raise SystemExit(f"No checkpoint result dirs under {results_root}")

    rows = [collect_checkpoint(p) for p in ckpt_dirs]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)
    write_markdown(rows, out_md)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print()
    print(out_md.read_text())


if __name__ == "__main__":
    main()
