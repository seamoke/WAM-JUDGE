"""Synchronously upload completed online RFT collection records to SwanLab."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from robotwin_critic.two_stage_rft.summarize_online_task_retention import (
    read_jsonl,
    summarize_rows,
    task_name,
)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_metrics(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {f"{prefix}/count": 0.0}
    return {
        f"{prefix}/count": float(len(values)),
        f"{prefix}/mean": statistics.fmean(values),
        f"{prefix}/std": statistics.pstdev(values),
        f"{prefix}/min": min(values),
        f"{prefix}/p05": quantile(values, 0.05),
        f"{prefix}/p25": quantile(values, 0.25),
        f"{prefix}/median": quantile(values, 0.50),
        f"{prefix}/p75": quantile(values, 0.75),
        f"{prefix}/p95": quantile(values, 0.95),
        f"{prefix}/max": max(values),
    }


def build_collect_metrics(
    generated: list[dict],
    retained: list[dict],
    selection_summary: dict,
    cumulative_generated: list[dict],
    cumulative_retained: list[dict],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    per_task = summarize_rows(generated, retained)
    cumulative_tasks = summarize_rows(cumulative_generated, cumulative_retained)
    generated_q = len({str(row["context_id"]) for row in generated})
    retained_q = len({str(row["context_id"]) for row in retained})
    metrics.update(
        {
            "collect/generated_q": float(generated_q),
            "collect/generated_qa_pairs": float(len(generated)),
            "collect/retained_q": float(retained_q),
            "collect/retained_qa_pairs": float(len(retained)),
            "collect/q_retention_rate": retained_q / generated_q
            if generated_q
            else 0.0,
            "collect/qa_retention_rate": len(retained) / len(generated)
            if generated
            else 0.0,
            "collect/tasks_generated": float(len(per_task)),
            "collect/tasks_retained": float(
                sum(row["retained_qa_pairs"] > 0 for row in per_task.values())
            ),
            "buffer/pending": float(selection_summary.get("pending_after_commit", 0)),
            "buffer/capacity": float(selection_summary.get("buffer_capacity", 0)),
            "reject/action": float(selection_summary.get("action_rejected", 0)),
            "reject/process": float(selection_summary.get("process_rejected", 0)),
            "reject/numeric_parse": float(
                selection_summary.get("numeric_parse_rejected", 0)
            ),
        }
    )
    capacity = metrics["buffer/capacity"]
    metrics["buffer/fill_fraction"] = (
        metrics["buffer/pending"] / capacity if capacity else 0.0
    )

    process_scores = [float(row["process_score"]) for row in generated]
    action_scores = [
        float(row.get("action_critic", {}).get("action_score", 0.0))
        for row in generated
    ]
    metrics.update(numeric_metrics("critic/process_score", process_scores))
    metrics.update(numeric_metrics("critic/action_score", action_scores))
    metrics["critic/process_positive_rate"] = (
        sum(value > 5.0 for value in process_scores) / len(process_scores)
        if process_scores
        else 0.0
    )
    metrics["critic/process_neutral_rate"] = (
        sum(abs(value) <= 5.0 for value in process_scores) / len(process_scores)
        if process_scores
        else 0.0
    )
    metrics["critic/process_negative_rate"] = (
        sum(value < -5.0 for value in process_scores) / len(process_scores)
        if process_scores
        else 0.0
    )
    metrics["critic/process_saturation_rate"] = (
        sum(abs(value) >= 99.9 for value in process_scores) / len(process_scores)
        if process_scores
        else 0.0
    )
    metrics["critic/action_accepted_rate"] = (
        sum(bool(row.get("action_critic", {}).get("accepted", False)) for row in generated)
        / len(generated)
        if generated
        else 0.0
    )
    metrics["critic/numeric_parse_rate"] = (
        sum(
            bool(row.get("process_critic", {}).get("numeric_parsed", True))
            for row in generated
        )
        / len(generated)
        if generated
        else 0.0
    )

    for task, values in per_task.items():
        prefix = f"task/{slug(task)}"
        metrics[f"{prefix}/generated_q"] = float(values["generated_q"])
        metrics[f"{prefix}/generated_qa_pairs"] = float(
            values["generated_qa_pairs"]
        )
        metrics[f"{prefix}/retained_q"] = float(values["retained_q"])
        metrics[f"{prefix}/retained_qa_pairs"] = float(
            values["retained_qa_pairs"]
        )
        metrics[f"{prefix}/qa_retention_rate"] = float(
            values["qa_retention_rate"]
        )
    for task, values in cumulative_tasks.items():
        prefix = f"task_cumulative/{slug(task)}"
        metrics[f"{prefix}/generated_qa_pairs"] = float(
            values["generated_qa_pairs"]
        )
        metrics[f"{prefix}/retained_qa_pairs"] = float(
            values["retained_qa_pairs"]
        )
        metrics[f"{prefix}/qa_retention_rate"] = float(
            values["qa_retention_rate"]
        )

    violations = Counter()
    for row in generated:
        action = row.get("action_critic", {})
        for violation in action.get("gate_violations", []):
            violations[str(violation)] += 1
        for violation in action.get("hard_violations", []):
            violations[f"hard.{violation}"] += 1
    for name, count in violations.items():
        metrics[f"action_violation/{slug(name)}"] = float(count)

    for domain in sorted({str(row.get("domain", "unknown")) for row in generated}):
        domain_generated = [row for row in generated if str(row.get("domain", "unknown")) == domain]
        domain_retained = [row for row in retained if str(row.get("domain", "unknown")) == domain]
        prefix = f"domain/{slug(domain)}"
        metrics[f"{prefix}/generated_qa_pairs"] = float(len(domain_generated))
        metrics[f"{prefix}/retained_qa_pairs"] = float(len(domain_retained))
        metrics[f"{prefix}/qa_retention_rate"] = (
            len(domain_retained) / len(domain_generated) if domain_generated else 0.0
        )
    return metrics


def selected_examples(rows: list[dict], retained: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    retained_ids = {str(row["candidate_id"]) for row in retained}
    choices = []
    ranked = sorted(rows, key=lambda row: float(row["process_score"]))
    pools = [retained, list(reversed(ranked)), ranked]
    seen = set()
    for pool in pools:
        for row in pool:
            candidate_id = str(row["candidate_id"])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            choices.append({**row, "_retained": candidate_id in retained_ids})
            if len(choices) >= limit:
                return choices
    return choices


def make_comparison_image(row: dict, output: Path) -> Path | None:
    from PIL import Image, ImageDraw

    current_path = Path(row.get("current_image", ""))
    generated_path = Path(row.get("generated_image", ""))
    if not current_path.is_file() or not generated_path.is_file():
        return None
    current = Image.open(current_path).convert("RGB")
    generated = Image.open(generated_path).convert("RGB")
    height = min(current.height, generated.height)
    current = current.resize((round(current.width * height / current.height), height))
    generated = generated.resize((round(generated.width * height / generated.height), height))
    header = 34
    canvas = Image.new("RGB", (current.width + generated.width, height + header), "white")
    canvas.paste(current, (0, header))
    canvas.paste(generated, (current.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), "CURRENT", fill="black")
    draw.text((current.width + 8, 9), "WAM GENERATED", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)
    return output


def load_upload_state(path: Path) -> dict:
    if not path.is_file():
        return {"logged_collects": []}
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class OnlineCollectionLogger:
    """Log completed collection rounds into an already-active SwanLab run."""

    def __init__(
        self,
        swanlab_module,
        online_root: Path,
        state_path: Path,
        run_id: str,
        max_images_per_collect: int = 4,
    ) -> None:
        self.swanlab = swanlab_module
        self.online_root = Path(online_root)
        self.state_path = Path(state_path)
        self.run_id = run_id
        self.max_images_per_collect = max_images_per_collect
        upload_state = load_upload_state(self.state_path)
        self.logged = {
            int(value) for value in upload_state.get("logged_collects", [])
        }
        self.loaded: set[int] = set()
        self.cumulative_generated: list[dict] = []
        self.cumulative_retained: list[dict] = []

    def _completed_collects(self):
        for collect_dir in sorted((self.online_root / "collect").glob("collect_*")):
            scored_path = collect_dir / "dual_scored.jsonl"
            selected_path = collect_dir / "selected_winners.jsonl"
            summary_path = collect_dir / "selection_summary.json"
            if all(path.is_file() for path in (scored_path, selected_path, summary_path)):
                yield (
                    int(collect_dir.name.rsplit("_", 1)[-1]),
                    collect_dir,
                    scored_path,
                    selected_path,
                    summary_path,
                )

    def log_completed(self, through: int | None = None) -> int:
        uploaded = 0
        for (
            collect_index,
            collect_dir,
            scored_path,
            selected_path,
            summary_path,
        ) in self._completed_collects():
            if through is not None and collect_index > through:
                continue
            generated = read_jsonl(scored_path)
            retained = read_jsonl(selected_path)
            if collect_index not in self.loaded:
                self.cumulative_generated.extend(generated)
                self.cumulative_retained.extend(retained)
                self.loaded.add(collect_index)
            if collect_index in self.logged:
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = build_collect_metrics(
                generated,
                retained,
                summary,
                self.cumulative_generated,
                self.cumulative_retained,
            )
            examples = []
            image_root = (
                self.online_root / "swanlab_collection_images" / collect_dir.name
            )
            for image_index, row in enumerate(
                selected_examples(
                    generated,
                    retained,
                    self.max_images_per_collect,
                )
            ):
                path = make_comparison_image(
                    row,
                    image_root / f"example_{image_index:02d}.jpg",
                )
                if path is None:
                    continue
                action_score = float(
                    row.get("action_critic", {}).get("action_score", 0.0)
                )
                caption = (
                    f"{row.get('task', 'unknown')} | "
                    f"process={float(row['process_score']):+.1f} "
                    f"action={action_score:.3f} retained={bool(row['_retained'])}"
                )
                examples.append(self.swanlab.Image(str(path), caption=caption))
            if examples:
                metrics["audit/current_vs_wam"] = examples
            self.swanlab.log(metrics, step=collect_index)
            self.logged.add(collect_index)
            atomic_write_json(
                self.state_path,
                {
                    "run_id": self.run_id,
                    "logged_collects": sorted(self.logged),
                    "last_collect": max(self.logged),
                },
            )
            uploaded += 1
        return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--name", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--max-images-per-collect", type=int, default=4)
    args = parser.parse_args()

    api_key = os.getenv("SWANLAB_API_KEY")
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is required for collection logging")
    import swanlab

    swanlab.login(api_key=api_key, save=False)
    kwargs = {
        "project": args.project,
        "name": args.name,
        "config": {"stream": "online_rft_collection", "schema_version": 1},
        "mode": "online",
        "log_dir": str(args.log_dir),
        "id": args.run_id,
        "resume": "allow",
    }
    if args.group:
        kwargs["group"] = args.group
    run = swanlab.init(**kwargs)
    try:
        try:
            (args.online_root / "swanlab_collection_url.txt").write_text(
                str(run.url) + "\n", encoding="utf-8"
            )
        except (AttributeError, ValueError):
            pass
        logger = OnlineCollectionLogger(
            swanlab,
            args.online_root,
            args.state,
            args.run_id,
            args.max_images_per_collect,
        )
        uploaded = logger.log_completed()
        print(
            json.dumps(
                {
                    "uploaded_collects": uploaded,
                    "logged_collects": len(logger.logged),
                    "run_id": args.run_id,
                }
            )
        )
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
