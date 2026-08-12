"""Resumable context scheduling and replay-buffer updates for online Dual-RFT."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from robotwin_critic.two_stage_rft.protocol import sha256_file


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_state(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Online state does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def shuffled_indices(size: int, base_seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(base_seed + epoch * 1_000_003).shuffle(order)
    return order


def initialize_state(
    state_path: Path,
    contexts_path: Path,
    current_model: Path,
    *,
    base_seed: int,
) -> dict:
    if state_path.exists():
        state = load_state(state_path)
        expected = sha256_file(contexts_path)
        if state["contexts_sha256"] != expected:
            raise RuntimeError("Existing online state belongs to different contexts")
        return state
    state = {
        "version": 1,
        "contexts": str(contexts_path.resolve()),
        "contexts_sha256": sha256_file(contexts_path),
        "base_seed": int(base_seed),
        "context_epoch": 0,
        "next_context_index": 0,
        "collect_index": 0,
        "update_index": 0,
        "current_model": str(current_model.resolve()),
        "accepted_total": 0,
        "consumed_total": 0,
        "ready_buffer": None,
    }
    atomic_write_json(state_path, state)
    return state


def prepare_collect(
    state_path: Path,
    output_dir: Path,
    *,
    workers: int,
    q_per_worker: int,
) -> dict:
    state = load_state(state_path)
    if state.get("ready_buffer"):
        raise RuntimeError(
            f"Train pending ready buffer first: {state['ready_buffer']}"
        )
    contexts_path = Path(state["contexts"])
    if sha256_file(contexts_path) != state["contexts_sha256"]:
        raise RuntimeError("Context file changed after online state initialization")
    contexts = read_jsonl(contexts_path)
    if not contexts:
        raise ValueError("Context file is empty")
    total = workers * q_per_worker
    if workers <= 0 or q_per_worker <= 0:
        raise ValueError("workers and q_per_worker must be positive")
    epoch = int(state["context_epoch"])
    cursor = int(state["next_context_index"])
    selected: list[dict] = []
    while len(selected) < total:
        order = shuffled_indices(len(contexts), int(state["base_seed"]), epoch)
        available = min(total - len(selected), len(contexts) - cursor)
        for source_index in order[cursor : cursor + available]:
            row = dict(contexts[source_index])
            original = str(row["context_id"])
            ordinal = len(selected)
            row["source_context_id"] = original
            row["context_id"] = (
                f"{original}@e{epoch:04d}c{int(state['collect_index']):06d}"
                f"q{ordinal:04d}"
            )
            row["online_context_epoch"] = epoch
            row["online_source_index"] = source_index
            selected.append(row)
        cursor += available
        if cursor == len(contexts):
            epoch += 1
            cursor = 0

    output_dir.mkdir(parents=True, exist_ok=False)
    shards = []
    for worker in range(workers):
        shard = output_dir / f"contexts_worker_{worker:02d}.jsonl"
        rows = selected[worker * q_per_worker : (worker + 1) * q_per_worker]
        atomic_write_jsonl(shard, rows)
        shards.append(str(shard.resolve()))
    manifest = {
        "collect_index": int(state["collect_index"]),
        "model": state["current_model"],
        "workers": workers,
        "q_per_worker": q_per_worker,
        "global_q": total,
        "shards": shards,
        "next_context_epoch": epoch,
        "next_context_index": cursor,
        "state_collect_index": int(state["collect_index"]) + 1,
    }
    atomic_write_json(output_dir / "collect_manifest.json", manifest)
    return manifest


def split_jsonl(input_path: Path, output_dir: Path, workers: int) -> list[str]:
    rows = read_jsonl(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(workers)]
    for index, row in enumerate(rows):
        shards[index % workers].append(row)
    paths = []
    for worker, shard_rows in enumerate(shards):
        path = output_dir / f"shard_{worker:02d}.jsonl"
        atomic_write_jsonl(path, shard_rows)
        paths.append(str(path.resolve()))
    return paths


def merge_jsonl(inputs: list[Path], output: Path) -> int:
    rows = []
    for path in inputs:
        rows.extend(read_jsonl(path))
    atomic_write_jsonl(output, rows)
    return len(rows)


def select_online_winners(
    rows: list[dict],
    *,
    min_action_score: float,
    min_process_score: float,
    split_manifest_sha256: str,
    collect_index: int,
    max_per_context: int = 1,
) -> tuple[list[dict], dict]:
    if max_per_context < 0:
        raise ValueError("max_per_context must be non-negative; zero means all")
    grouped: dict[str, list[dict]] = defaultdict(list)
    rejected_action = rejected_process = rejected_parse = 0
    for row in rows:
        action = row.get("action_critic", {})
        process = row.get("process_critic", {})
        if not bool(process.get("numeric_parsed", True)):
            rejected_parse += 1
            continue
        if (
            not bool(action.get("accepted", False))
            or float(action.get("action_score", float("-inf")))
            < min_action_score
        ):
            rejected_action += 1
            continue
        if float(row.get("process_score", float("-inf"))) < min_process_score:
            rejected_process += 1
            continue
        grouped[str(row["context_id"])].append(row)
    winners: list[tuple[dict, int]] = []
    for context_id in sorted(grouped):
        ranked = sorted(
            grouped[context_id],
            key=lambda row: float(row["process_score"]),
            reverse=True,
        )
        if max_per_context:
            ranked = ranked[:max_per_context]
        winners.extend((row, rank) for rank, row in enumerate(ranked))
    selected = [
        {
            **row,
            "rft_selection": {
                "mode": "dual",
                "online": True,
                "collect_index": collect_index,
                "min_action_score": float(min_action_score),
                "min_process_score": float(min_process_score),
                "split_manifest_sha256": split_manifest_sha256,
                "rank_within_context": rank,
                "max_per_context": int(max_per_context),
            },
        }
        for row, rank in winners
    ]
    summary = {
        "input_candidates": len(rows),
        "valid_contexts": len(grouped),
        "selected_winners": len(selected),
        "eligible_candidates": sum(len(value) for value in grouped.values()),
        "max_per_context": int(max_per_context),
        "action_rejected": rejected_action,
        "process_rejected": rejected_process,
        "numeric_parse_rejected": rejected_parse,
        "min_action_score": min_action_score,
        "min_process_score": min_process_score,
    }
    return selected, summary


def commit_collect(
    state_path: Path,
    collect_dir: Path,
    scored_path: Path,
    pending_path: Path,
    buffers_dir: Path,
    *,
    capacity: int,
    min_action_score: float,
    min_process_score: float,
    split_manifest: Path,
    max_per_context: int = 1,
) -> dict:
    if capacity < 0:
        raise ValueError("capacity must be non-negative; zero is collect-only")
    state = load_state(state_path)
    manifest = json.loads(
        (collect_dir / "collect_manifest.json").read_text(encoding="utf-8")
    )
    if int(state["collect_index"]) != int(manifest["collect_index"]):
        raise RuntimeError("Collect manifest does not match resumable state")
    selected, summary = select_online_winners(
        read_jsonl(scored_path),
        min_action_score=min_action_score,
        min_process_score=min_process_score,
        split_manifest_sha256=sha256_file(split_manifest),
        collect_index=int(manifest["collect_index"]),
        max_per_context=max_per_context,
    )
    atomic_write_jsonl(collect_dir / "selected_winners.jsonl", selected)
    pending = read_jsonl(pending_path) if pending_path.is_file() else []
    known = {str(row["candidate_id"]) for row in pending}
    pending.extend(
        row for row in selected if str(row["candidate_id"]) not in known
    )
    ready_buffer = None
    if capacity > 0 and len(pending) >= capacity:
        buffers_dir.mkdir(parents=True, exist_ok=True)
        ready_buffer = (
            buffers_dir / f"buffer_update_{int(state['update_index']):06d}.jsonl"
        )
        if ready_buffer.exists():
            raise FileExistsError(f"Ready buffer already exists: {ready_buffer}")
        atomic_write_jsonl(ready_buffer, pending[:capacity])
        pending = pending[capacity:]
    atomic_write_jsonl(pending_path, pending)
    state.update(
        {
            "context_epoch": int(manifest["next_context_epoch"]),
            "next_context_index": int(manifest["next_context_index"]),
            "collect_index": int(manifest["state_collect_index"]),
            "accepted_total": int(state["accepted_total"]) + len(selected),
            "ready_buffer": str(ready_buffer.resolve()) if ready_buffer else None,
        }
    )
    atomic_write_json(state_path, state)
    summary.update(
        {
            "pending_after_commit": len(pending),
            "buffer_capacity": capacity,
            "ready_buffer": state["ready_buffer"],
            "collect_index": manifest["collect_index"],
        }
    )
    atomic_write_json(collect_dir / "selection_summary.json", summary)
    return summary


def complete_update(state_path: Path, model: Path) -> dict:
    state = load_state(state_path)
    ready = state.get("ready_buffer")
    if not ready:
        raise RuntimeError("No ready buffer is awaiting an update")
    consumed = len(read_jsonl(Path(ready)))
    state["current_model"] = str(model.resolve())
    state["consumed_total"] = int(state["consumed_total"]) + consumed
    state["update_index"] = int(state["update_index"]) + 1
    state["ready_buffer"] = None
    atomic_write_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--contexts", type=Path, required=True)
    init.add_argument("--model", type=Path, required=True)
    init.add_argument("--base-seed", type=int, default=42)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--state", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--workers", type=int, required=True)
    prepare.add_argument("--q-per-worker", type=int, required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--workers", type=int, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)

    commit = subparsers.add_parser("commit")
    commit.add_argument("--state", type=Path, required=True)
    commit.add_argument("--collect-dir", type=Path, required=True)
    commit.add_argument("--scored", type=Path, required=True)
    commit.add_argument("--pending", type=Path, required=True)
    commit.add_argument("--buffers-dir", type=Path, required=True)
    commit.add_argument(
        "--capacity",
        type=int,
        default=64,
        help="Training buffer size; zero accumulates candidates without updates.",
    )
    commit.add_argument("--min-action-score", type=float, default=0.5)
    commit.add_argument("--min-process-score", type=float, default=5.0)
    commit.add_argument(
        "--max-per-context",
        type=int,
        default=1,
        help="Maximum accepted candidates saved per Q; zero saves all.",
    )
    commit.add_argument("--split-manifest", type=Path, required=True)

    complete = subparsers.add_parser("complete-update")
    complete.add_argument("--state", type=Path, required=True)
    complete.add_argument("--model", type=Path, required=True)

    field = subparsers.add_parser("field")
    field.add_argument("--state", type=Path, required=True)
    field.add_argument("--name", required=True)

    args = parser.parse_args()
    if args.command == "init":
        result = initialize_state(
            args.state, args.contexts, args.model, base_seed=args.base_seed
        )
    elif args.command == "prepare":
        result = prepare_collect(
            args.state,
            args.output_dir,
            workers=args.workers,
            q_per_worker=args.q_per_worker,
        )
    elif args.command == "split":
        result = {
            "shards": split_jsonl(args.input, args.output_dir, args.workers)
        }
    elif args.command == "merge":
        result = {"rows": merge_jsonl(args.input, args.output)}
    elif args.command == "commit":
        result = commit_collect(
            args.state,
            args.collect_dir,
            args.scored,
            args.pending,
            args.buffers_dir,
            capacity=args.capacity,
            min_action_score=args.min_action_score,
            min_process_score=args.min_process_score,
            split_manifest=args.split_manifest,
            max_per_context=args.max_per_context,
        )
    elif args.command == "complete-update":
        result = complete_update(args.state, args.model)
    else:
        value = load_state(args.state).get(args.name)
        print("" if value is None else value)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
