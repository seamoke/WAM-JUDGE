"""Aggregate kinematic Action Critic rejection causes across collect rounds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def analyze(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    violations = Counter()
    violation_groups = Counter()
    task_rejected = Counter()
    task_total = Counter()
    scores = []
    accepted_scores = []
    rejected_scores = []
    violations_per_candidate = Counter()
    exceedance_ratios: dict[str, list[float]] = defaultdict(list)
    all_threshold_ratios: dict[str, list[float]] = defaultdict(list)
    contexts: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        critic = row.get("action_critic", {})
        score = float(critic.get("action_score", float("nan")))
        accepted = bool(critic.get("accepted", False))
        scores.append(score)
        (accepted_scores if accepted else rejected_scores).append(score)
        task = str(row.get("task", "unknown"))
        task_total[task] += 1
        hard = [str(value) for value in critic.get("hard_violations", [])]
        contexts[str(row.get("context_id", row.get("source_context_id", "unknown")))].append(row)
        violations_per_candidate[len(hard)] += 1
        if not accepted:
            task_rejected[task] += 1
        for name in hard:
            violations[name] += 1
            if name.endswith("eef_workspace"):
                group = "workspace"
            elif "." in name:
                group = name.split(".", 1)[1]
            else:
                group = name
            violation_groups[group] += 1
        for name, diagnostic in critic.get("diagnostics", {}).items():
            hard_limit = float(diagnostic.get("hard", 0.0))
            maximum = float(diagnostic.get("maximum", 0.0))
            if hard_limit > 0.0:
                ratio = maximum / hard_limit
                all_threshold_ratios[str(name)].append(ratio)
                if ratio > 1.0:
                    exceedance_ratios[str(name)].append(ratio)
    accepted = len(accepted_scores)
    total = len(rows)
    per_task = {
        task: {
            "total": count,
            "rejected": task_rejected[task],
            "rejection_rate": task_rejected[task] / count,
        }
        for task, count in task_total.most_common()
    }

    def ratio_summary(values: list[float]) -> dict:
        return {
            "count": len(values),
            "median": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    def passes_policy(row: dict, score_threshold: float, multiplier: float) -> bool:
        critic = row.get("action_critic", {})
        if float(critic.get("action_score", float("-inf"))) < score_threshold:
            return False
        diagnostics = critic.get("diagnostics", {})
        for diagnostic in diagnostics.values():
            hard_limit = float(diagnostic.get("hard", 0.0))
            maximum = float(diagnostic.get("maximum", 0.0))
            if hard_limit <= 0.0 or maximum > hard_limit * multiplier:
                return False
        # Workspace and direct range violations remain strict safety gates.
        if any(
            bool(value.get("hard_violation", False))
            for value in critic.get("workspace", {}).values()
        ):
            return False
        diagnostic_names = set(diagnostics)
        direct_violations = [
            str(name)
            for name in critic.get("hard_violations", [])
            if str(name) not in diagnostic_names
            and not str(name).endswith("eef_workspace")
        ]
        return not direct_violations

    policy_sweep = {}
    for score_threshold in (0.5, 0.7, 0.75, 0.8):
        for multiplier in (1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, math.inf):
            candidate_passes = [
                passes_policy(row, score_threshold, multiplier) for row in rows
            ]
            contexts_with_candidate = sum(
                any(passes_policy(row, score_threshold, multiplier) for row in group)
                for group in contexts.values()
            )
            multiplier_name = "score_only" if math.isinf(multiplier) else str(multiplier)
            key = f"score>={score_threshold},hard_multiplier={multiplier_name},workspace=strict"
            policy_sweep[key] = {
                "candidate_acceptance_rate": (
                    sum(candidate_passes) / len(candidate_passes)
                    if candidate_passes
                    else 0.0
                ),
                "accepted_candidates": sum(candidate_passes),
                "contexts_with_candidate": contexts_with_candidate,
                "context_coverage": (
                    contexts_with_candidate / len(contexts) if contexts else 0.0
                ),
            }
    return {
        "files": len(paths),
        "candidates": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": accepted / total if total else 0.0,
        "action_score": {
            "min": min(scores) if scores else None,
            "p10": percentile(scores, 0.10),
            "p25": percentile(scores, 0.25),
            "median": percentile(scores, 0.50),
            "p75": percentile(scores, 0.75),
            "p90": percentile(scores, 0.90),
            "max": max(scores) if scores else None,
        },
        "hard_violations": dict(violations.most_common()),
        "hard_violation_groups": dict(violation_groups.most_common()),
        "hard_violations_per_candidate": {
            str(key): value for key, value in sorted(violations_per_candidate.items())
        },
        "hard_exceedance_ratio": {
            key: ratio_summary(values)
            for key, values in sorted(
                exceedance_ratios.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "all_max_over_hard_ratio": {
            key: ratio_summary(values)
            for key, values in sorted(all_threshold_ratios.items())
        },
        "policy_sweep": policy_sweep,
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.collect_root.glob("collect_*/action_scored.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No action_scored.jsonl under {args.collect_root}")
    report = analyze(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
