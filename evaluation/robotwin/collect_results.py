#!/usr/bin/env python3
"""Aggregate RoboTwin eval metrics (res.json) into CSV / Markdown."""
import argparse
import csv
import json
from pathlib import Path

from evaluation.robotwin.calc_stat import TASK_CLASS, mean_rate_of

ALL_TASKS = [
    "stack_bowls_three", "handover_block", "hanging_mug", "scan_object", "lift_pot",
    "put_object_cabinet", "stack_blocks_three", "place_shoe",
    "adjust_bottle", "place_mouse_pad", "dump_bin_bigbin", "move_pillbottle_pad",
    "pick_dual_bottles", "shake_bottle", "place_fan", "turn_switch",
    "shake_bottle_horizontally", "place_container_plate", "rotate_qrcode",
    "place_object_stand", "put_bottles_dustbin", "move_stapler_pad",
    "place_burger_fries", "place_bread_basket",
    "pick_diverse_bottles", "open_microwave", "beat_block_hammer", "press_stapler",
    "click_bell", "move_playingcard_away", "open_laptop", "move_can_pot",
    "stack_bowls_two", "place_a2b_right", "stamp_seal", "place_object_basket",
    "handover_mic", "place_bread_skillet", "stack_blocks_two", "place_cans_plasticbox",
    "click_alarmclock", "blocks_ranking_size", "place_phone_stand", "place_can_basket",
    "place_object_scale", "place_a2b_left", "grab_roller", "place_dual_shoes",
    "place_empty_cup", "blocks_ranking_rgb",
]


def find_metrics_file(result_dir: Path, task_name: str, st_seed: int) -> Path | None:
    candidates = [
        result_dir / f"stseed-{st_seed}" / "metrics" / task_name / "res.json",
        result_dir / "metrics" / task_name / "res.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_task_rate(result_dir: Path, task_name: str, st_seed: int) -> float | None:
    metrics_file = find_metrics_file(result_dir, task_name, st_seed)
    if metrics_file is None:
        return None
    with open(metrics_file) as f:
        data = json.load(f)
    return float(data["succ_rate"])


def collect_checkpoint(result_dir: Path, st_seed: int) -> dict:
    row = {"checkpoint": result_dir.name}
    rates = []
    for task_name in ALL_TASKS:
        rate = load_task_rate(result_dir, task_name, st_seed)
        row[f"task_{task_name}"] = rate
        if rate is not None:
            rates.append(rate)

    row["avg_sr"] = sum(rates) / len(rates) if rates else None
    row["num_tasks"] = len(rates)

    for cls in (1, 2, 3):
        cls_rates = [
            load_task_rate(result_dir, task, st_seed)
            for task in ALL_TASKS
            if TASK_CLASS.get(task) == cls
        ]
        cls_rates = [r for r in cls_rates if r is not None]
        row[f"class{cls}_sr"] = sum(cls_rates) / len(cls_rates) if cls_rates else None

    return row


def is_model_result_dir(path: Path) -> bool:
    """Return true for both checkpoint and direct full-model result layouts."""
    if not path.is_dir():
        return False
    if (path / "metrics").is_dir():
        return True
    return any(metrics_dir.is_dir() for metrics_dir in path.glob("stseed-*/metrics"))


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}"


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fieldnames = ["checkpoint", "num_tasks", "avg_sr", "class1_sr", "class2_sr", "class3_sr"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "checkpoint": row["checkpoint"],
                    "num_tasks": row.get("num_tasks", 0),
                    "avg_sr": format_pct(row.get("avg_sr")),
                    "class1_sr": format_pct(row.get("class1_sr")),
                    "class2_sr": format_pct(row.get("class2_sr")),
                    "class3_sr": format_pct(row.get("class3_sr")),
                }
            )


def write_markdown(rows: list[dict], out_md: Path, task_config: str) -> None:
    headers = ["Checkpoint", "Tasks", "Avg SR", "Class1", "Class2", "Class3"]
    lines = [
        f"# RoboTwin eval ({task_config})",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["checkpoint"],
                    str(row.get("num_tasks", 0)),
                    format_pct(row.get("avg_sr")),
                    format_pct(row.get("class1_sr")),
                    format_pct(row.get("class2_sr")),
                    format_pct(row.get("class3_sr")),
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="/workspace/lingbot-va/train_out/robotwin/eval_results",
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--st-seed", type=int, default=10000)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root) / args.task_config
    out_csv = Path(args.out_csv or results_root / "results.csv")
    out_md = Path(args.out_md or results_root / "results.md")

    model_result_dirs = sorted(
        (p for p in results_root.iterdir() if is_model_result_dir(p)),
        key=lambda path: path.name,
    ) if results_root.is_dir() else []

    if not model_result_dirs:
        raise SystemExit(f"No model result dirs under {results_root}")

    rows = [collect_checkpoint(p, args.st_seed) for p in model_result_dirs]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)
    write_markdown(rows, out_md, args.task_config)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print()
    print(out_md.read_text())


if __name__ == "__main__":
    main()
