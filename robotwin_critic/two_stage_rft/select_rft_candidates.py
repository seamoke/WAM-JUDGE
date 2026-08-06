"""Apply action threshold, VLAC reranking, and the exact Stage-2 chunk budget."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def context_key(row: dict) -> str:
    return str(row["context_id"])


def group_key(row: dict) -> str:
    return f"{row['task']}/{row['domain']}"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_candidates(
    rows: list[dict],
    budgets: dict[str, int],
    *,
    mode: str = "dual",
    min_action_score: float,
    require_exact_budget: bool = True,
    expected_split_sha256: str | None = None,
) -> tuple[list[dict], dict]:
    if mode not in {"naive", "process", "action", "dual"}:
        raise ValueError(f"Unsupported selection mode: {mode}")
    if expected_split_sha256 is not None:
        observed = {str(row.get("split_manifest_sha256", "")) for row in rows}
        if observed != {expected_split_sha256}:
            raise ValueError(
                f"Candidate split hashes {observed} do not match budget split "
                f"{expected_split_sha256}"
            )
    grouped: dict[str, list[dict]] = defaultdict(list)
    rejected_action = rejected_parse = 0
    for row in rows:
        action = row.get("action_critic", {})
        if mode in {"process", "dual"} and not bool(
            row.get("process_critic", {}).get("numeric_parsed", True)
        ):
            rejected_parse += 1
            continue
        if mode in {"action", "dual"} and (
            not bool(action.get("accepted", False))
            or float(action.get("action_score", float("-inf"))) < min_action_score
        ):
            rejected_action += 1
            continue
        grouped[context_key(row)].append(row)

    winners = []
    for context_rows in grouped.values():
        if mode in {"process", "dual"}:
            key = lambda row: float(row["process_score"])
        elif mode == "action":
            key = lambda row: float(row["action_critic"]["action_score"])
        else:
            key = lambda row: -int(row.get("candidate_index", 0))
        winners.append(max(context_rows, key=key))

    winners_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in winners:
        winners_by_group[group_key(row)].append(row)
    selected = []
    shortfalls = {}
    for group, budget in sorted(budgets.items()):
        if mode in {"process", "dual"}:
            ranking = lambda row: float(row["process_score"])
        elif mode == "action":
            ranking = lambda row: float(row["action_critic"]["action_score"])
        else:
            ranking = lambda row: (
                -int(row.get("source_episode_index", 0)),
                -int(row.get("frame_index", 0)),
            )
        available = sorted(
            winners_by_group.get(group, []),
            key=ranking,
            reverse=True,
        )
        if len(available) < budget:
            shortfalls[group] = {"budget": budget, "available": len(available)}
        selected.extend(available[:budget])
    if require_exact_budget and shortfalls:
        raise RuntimeError(
            "Not enough accepted context winners for exact pseudo budget: "
            + json.dumps(shortfalls, sort_keys=True)
        )
    selected.sort(key=lambda row: (group_key(row), context_key(row)))
    selected = [
        {
            **row,
            "rft_selection": {
                "mode": mode,
                "min_action_score": float(min_action_score),
                "budget_group": group_key(row),
                "group_budget": int(budgets[group_key(row)]),
                "split_manifest_sha256": expected_split_sha256,
            },
        }
        for row in selected
    ]
    selected_counts = Counter(group_key(row) for row in selected)
    summary = {
        "input_candidates": len(rows),
        "action_rejected": rejected_action,
        "numeric_parse_rejected": rejected_parse,
        "contexts_with_mode_valid_candidate": len(grouped),
        "context_winners": len(winners),
        "selected": len(selected),
        "budget_total": sum(budgets.values()),
        "mode": mode,
        "require_exact_budget": require_exact_budget,
        "split_manifest_sha256": expected_split_sha256,
        "selected_by_group": dict(sorted(selected_counts.items())),
        "shortfalls": shortfalls,
        "selection_rule": ({
            "naive": "deterministic candidate without critic filtering",
            "process": "maximum VLAC process score per context",
            "action": "action threshold then maximum kinematic score per context",
            "dual": "action threshold then maximum VLAC process score per context",
        }[mode]
        + (
            "; require exact task/domain Stage-2 loader budget"
            if require_exact_budget
            else "; retain up to task/domain budget and report every shortfall"
        )),
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-action-score", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        choices=("naive", "process", "action", "dual"),
        default="dual",
    )
    parser.add_argument("--allow-shortfall", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    budget_document = json.loads(args.budget.read_text(encoding="utf-8"))
    selected, summary = select_candidates(
        rows,
        {key: int(value) for key, value in budget_document["groups"].items()},
        mode=args.mode,
        min_action_score=args.min_action_score,
        require_exact_budget=not args.allow_shortfall,
        expected_split_sha256=str(budget_document["split_manifest_sha256"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["output"] = str(args.output.resolve())
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ROBOTWIN_RFT_SELECTION_OK")


if __name__ == "__main__":
    main()
