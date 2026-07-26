"""Evaluate VLAC numeric pair scores, VOC, VROC, and antisymmetry on RoboTwin."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from .common import accumulate_progress, parse_score, read_jsonl, spearman_order, voc_f1, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--neutral-threshold", type=float, default=5.0)
    parser.add_argument(
        "--sampling",
        choices=("first", "random", "stratified"),
        default="first",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def load_critic(args: argparse.Namespace):
    vendor_root = Path(__file__).resolve().parent / "vendor" / "VLAC"
    if vendor_root.exists():
        for path in (vendor_root, vendor_root / "evo_vlac" / "utils"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    try:
        from evo_vlac import GAC_model
    except ImportError as exc:
        raise RuntimeError("Install the official InternRobotics/VLAC package before evaluation") from exc

    critic = GAC_model(tag="critic")
    critic.init_model(
        model_path=args.model,
        model_type="internvl2",
        device_map=args.device,
        adapter=args.adapter,
    )
    critic.temperature = 0.0
    critic.top_k = 1
    critic.max_tokens = 8
    critic.do_sample = False
    critic.set_config()
    critic.set_system_prompt()
    return critic


def infer_scores(
    critic,
    queries: Sequence[tuple[str, str, str]],
    batch_size: int,
) -> tuple[list[float], list[str], list[bool], float]:
    scores: list[float] = []
    raw_answers: list[str] = []
    parsed: list[bool] = []
    total_seconds = 0.0
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        prompts = [critic.get_score_prompt(task=task) for task, _, _ in batch]
        images = [[first, second] for _, first, second in batch]
        requests = critic.get_infer_requests(prompt=prompts, images=images)
        begin = time.perf_counter()
        responses, _ = critic.chat(requests)
        total_seconds += time.perf_counter() - begin
        answers, _ = critic.results_format(responses, requests, rich=False)
        for answer in answers:
            raw_answers.append(answer)
            try:
                scores.append(parse_score(answer))
                parsed.append(True)
            except ValueError:
                scores.append(0.0)
                parsed.append(False)
    return scores, raw_answers, parsed, total_seconds


def target_from_row(row: dict[str, Any]) -> float:
    metadata = row.get("metadata") or {}
    if "target" in metadata:
        return float(metadata["target"])
    return parse_score(row["messages"][-1]["content"])


def sign_correct(prediction: float, target: float, neutral_threshold: float) -> bool:
    if abs(target) < 0.05:
        return abs(prediction) <= neutral_threshold
    return bool(np.sign(prediction) == np.sign(target))


def pair_label(value: float, neutral_threshold: float) -> int:
    if abs(value) <= neutral_threshold:
        return 0
    return 1 if value > 0 else -1


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def pair_classification_metrics(
    targets: Sequence[float],
    predictions: Sequence[float],
    neutral_threshold: float,
) -> dict[str, Any]:
    target_labels = np.asarray(
        [pair_label(value, 0.05) for value in targets],
        dtype=np.int64,
    )
    prediction_labels = np.asarray(
        [pair_label(value, neutral_threshold) for value in predictions],
        dtype=np.int64,
    )
    predictions_array = np.asarray(predictions, dtype=np.float64)
    names = {-1: "negative", 0: "neutral", 1: "positive"}
    per_class_f1: dict[str, float] = {}
    per_class_auc: dict[str, float | None] = {}
    for label, name in names.items():
        true_positive = int(
            np.logical_and(target_labels == label, prediction_labels == label).sum()
        )
        false_positive = int(
            np.logical_and(target_labels != label, prediction_labels == label).sum()
        )
        false_negative = int(
            np.logical_and(target_labels == label, prediction_labels != label).sum()
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1[name] = (
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
        if label == -1:
            class_scores = -predictions_array
        elif label == 0:
            class_scores = -np.abs(predictions_array)
        else:
            class_scores = predictions_array
        per_class_auc[name] = binary_auc(target_labels == label, class_scores)

    finite_aucs = [value for value in per_class_auc.values() if value is not None]
    correlation = spearmanr(targets, predictions).statistic
    return {
        "labels": names,
        "accuracy": float(np.mean(target_labels == prediction_labels)),
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "per_class_f1": per_class_f1,
        "macro_ovr_auc": float(np.mean(finite_aucs)) if finite_aucs else None,
        "per_class_ovr_auc": per_class_auc,
        "target_spearman": (
            0.0 if not np.isfinite(correlation) else float(correlation)
        ),
    }


def summarize_prediction_records(
    pair_records: Sequence[dict[str, Any]],
    trajectory_records: Sequence[dict[str, Any]],
    neutral_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = [float(row["target"]) for row in pair_records]
    predictions = [float(row["prediction"]) for row in pair_records]
    pair_metrics = pair_classification_metrics(
        targets,
        predictions,
        neutral_threshold,
    )
    summary = {
        "pair_count": len(pair_records),
        "pair_numeric_parse_rate": float(
            np.mean([bool(row["parsed"]) for row in pair_records])
        ),
        "pair_mae": float(
            np.mean([abs(prediction - target) for prediction, target in zip(predictions, targets)])
        ),
        "pair_sign_accuracy": float(
            np.mean(
                [
                    sign_correct(prediction, target, neutral_threshold)
                    for prediction, target in zip(predictions, targets)
                ]
            )
        ),
        "pair_accuracy": pair_metrics["accuracy"],
        "pair_macro_f1": pair_metrics["macro_f1"],
        "pair_per_class_f1": pair_metrics["per_class_f1"],
        "pair_macro_ovr_auc": pair_metrics["macro_ovr_auc"],
        "pair_per_class_ovr_auc": pair_metrics["per_class_ovr_auc"],
        "pair_target_spearman": pair_metrics["target_spearman"],
        "trajectory_count": len(trajectory_records),
        "mean_voc": float(np.mean([row["voc"] for row in trajectory_records])),
        "mean_vroc": float(np.mean([row["vroc"] for row in trajectory_records])),
        "mean_voc_f1": float(np.mean([row["voc_f1"] for row in trajectory_records])),
        "mean_antisymmetry_mae": float(
            np.mean([row["antisymmetry_mae"] for row in trajectory_records])
        ),
        "trajectory_numeric_rate": float(
            np.mean([row["all_numeric"] for row in trajectory_records])
        ),
    }

    pair_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_records:
        pair_by_task[str(row["task"])].append(row)
    for row in trajectory_records:
        trajectory_by_task[str(row["task"])].append(row)

    task_names = sorted(set(pair_by_task) | set(trajectory_by_task))
    per_task: dict[str, Any] = {}
    for task in task_names:
        task_pairs = pair_by_task[task]
        task_trajectories = trajectory_by_task[task]
        task_result: dict[str, Any] = {
            "pair_count": len(task_pairs),
            "trajectory_count": len(task_trajectories),
        }
        if task_pairs:
            task_targets = [float(row["target"]) for row in task_pairs]
            task_predictions = [float(row["prediction"]) for row in task_pairs]
            metrics = pair_classification_metrics(
                task_targets,
                task_predictions,
                neutral_threshold,
            )
            task_result.update(
                {
                    "pair_mae": float(
                        np.mean(
                            np.abs(
                                np.asarray(task_predictions)
                                - np.asarray(task_targets)
                            )
                        )
                    ),
                    "pair_sign_accuracy": float(
                        np.mean(
                            [
                                sign_correct(prediction, target, neutral_threshold)
                                for prediction, target in zip(
                                    task_predictions, task_targets
                                )
                            ]
                        )
                    ),
                    "pair_accuracy": metrics["accuracy"],
                    "pair_macro_f1": metrics["macro_f1"],
                    "pair_macro_ovr_auc": metrics["macro_ovr_auc"],
                    "pair_target_spearman": metrics["target_spearman"],
                }
            )
        if task_trajectories:
            task_result.update(
                {
                    "mean_voc": float(
                        np.mean([row["voc"] for row in task_trajectories])
                    ),
                    "mean_vroc": float(
                        np.mean([row["vroc"] for row in task_trajectories])
                    ),
                    "mean_voc_f1": float(
                        np.mean([row["voc_f1"] for row in task_trajectories])
                    ),
                    "mean_antisymmetry_mae": float(
                        np.mean(
                            [
                                row["antisymmetry_mae"]
                                for row in task_trajectories
                            ]
                        )
                    ),
                }
            )
        per_task[task] = task_result

    return summary, {"task_count": len(task_names), "tasks": per_task}


def select_rows(
    rows: Sequence[dict[str, Any]],
    limit: int,
    sampling: str,
    seed: int,
    *,
    key_fn,
) -> list[dict[str, Any]]:
    rows = list(rows)
    if limit <= 0 or limit >= len(rows):
        return rows
    if sampling == "first":
        return rows[:limit]

    rng = random.Random(seed)
    if sampling == "random":
        return rng.sample(rows, limit)

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    for group in groups.values():
        rng.shuffle(group)

    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            group = groups[key]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_rows = read_jsonl(Path(args.data_dir) / "val.jsonl")
    trajectories = read_jsonl(Path(args.data_dir) / "val_trajectories.jsonl")
    val_rows = select_rows(
        val_rows,
        args.max_pairs,
        args.sampling,
        args.sample_seed,
        key_fn=lambda row: (
            row["metadata"]["task"],
            pair_label(target_from_row(row), 0.05),
        ),
    )
    trajectories = select_rows(
        trajectories,
        args.max_trajectories,
        args.sampling,
        args.sample_seed,
        key_fn=lambda row: row["task"],
    )
    critic = load_critic(args)

    pair_queries = [
        (row["metadata"]["task"], row["images"][0], row["images"][1]) for row in val_rows
    ]
    pair_scores, pair_answers, pair_parsed, pair_seconds = infer_scores(
        critic, pair_queries, args.batch_size
    )
    targets = [target_from_row(row) for row in val_rows]
    pair_records = []
    for row, target, prediction, answer, parsed in zip(
        val_rows, targets, pair_scores, pair_answers, pair_parsed
    ):
        pair_records.append(
            {
                "task": row["metadata"]["task"],
                "episode_index": row["metadata"]["episode_index"],
                "i": row["metadata"]["i"],
                "j": row["metadata"]["j"],
                "target": target,
                "prediction": prediction,
                "parsed": parsed,
                "raw_answer": answer,
                "absolute_error": abs(prediction - target),
                "sign_correct": sign_correct(prediction, target, args.neutral_threshold),
            }
        )
    write_jsonl(output_dir / "pair_predictions.jsonl", pair_records)
    trajectory_queries: list[tuple[str, str, str]] = []
    trajectory_slices: list[tuple[int, int]] = []
    for trajectory in trajectories:
        task = trajectory.get("text") or trajectory["task"]
        images = trajectory["images"]
        start = len(trajectory_queries)
        for first, second in zip(images[:-1], images[1:]):
            trajectory_queries.append((task, first, second))
            trajectory_queries.append((task, second, first))
        trajectory_slices.append((start, len(trajectory_queries)))

    trajectory_scores, trajectory_answers, trajectory_parsed, trajectory_seconds = infer_scores(
        critic, trajectory_queries, args.batch_size
    )
    trajectory_records = []
    for trajectory, (start, end) in zip(trajectories, trajectory_slices):
        scores = trajectory_scores[start:end]
        answers = trajectory_answers[start:end]
        parsed = trajectory_parsed[start:end]
        forward = scores[0::2]
        reverse_raw = scores[1::2]
        reverse_corrected = [-score for score in reverse_raw]
        forward_values = accumulate_progress(forward)
        reverse_values = accumulate_progress(reverse_corrected)
        voc = spearman_order(forward_values)
        vroc = spearman_order(reverse_values)
        trajectory_records.append(
            {
                "task": trajectory["task"],
                "episode_index": trajectory["episode_index"],
                "frame_indices": trajectory["frame_indices"],
                "forward_scores": forward,
                "reverse_raw_scores": reverse_raw,
                "reverse_corrected_scores": reverse_corrected,
                "forward_values": forward_values.tolist(),
                "reverse_values": reverse_values.tolist(),
                "voc": voc,
                "vroc": vroc,
                "voc_f1": voc_f1(voc, vroc),
                "antisymmetry_mae": float(
                    np.mean(np.abs(np.asarray(forward) + np.asarray(reverse_raw)))
                ),
                "all_numeric": bool(all(parsed)),
                "raw_answers": answers,
            }
        )
    write_jsonl(output_dir / "trajectory_predictions.jsonl", trajectory_records)

    record_summary, per_task_metrics = summarize_prediction_records(
        pair_records,
        trajectory_records,
        args.neutral_threshold,
    )
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        **record_summary,
        "pair_inference_seconds": pair_seconds,
        "trajectory_inference_seconds": trajectory_seconds,
    }
    summary["smoke_gates"] = {
        "rgb_manifest_nonempty": bool(val_rows and trajectories),
        "numeric_output_ok": summary["pair_numeric_parse_rate"] >= 0.99,
        "voc_finite": bool(np.isfinite(summary["mean_voc"])),
        "vroc_finite": bool(np.isfinite(summary["mean_vroc"])),
    }
    summary["smoke_passed"] = bool(all(summary["smoke_gates"].values()))
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "per_task_metrics.json").open("w") as handle:
        json.dump(per_task_metrics, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
