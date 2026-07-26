#!/usr/bin/env python3
"""Merge and validate sharded RoboTwin evaluation seed caches."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--per-task", type=int, required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated expected tasks")
    parser.add_argument("--source-sha256", default="")
    return parser.parse_args()


def validate_entry(task: str, index: int, entry: object) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"{task}[{index}] is not an object")
    if not isinstance(entry.get("seed"), int):
        raise ValueError(f"{task}[{index}].seed is not an integer")
    info = entry.get("episode_info")
    if not isinstance(info, dict) or not info:
        raise ValueError(f"{task}[{index}].episode_info is empty or invalid")
    for key, value in info.items():
        if not isinstance(key, str) or not key.startswith("{") or not key.endswith("}"):
            raise ValueError(f"{task}[{index}] has invalid placeholder key: {key!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{task}[{index}][{key!r}] has invalid value: {value!r}")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_name = f.name
    os.replace(tmp_name, path)


def main() -> None:
    args = parse_args()
    expected_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if len(expected_tasks) != len(set(expected_tasks)):
        raise ValueError("Expected task list contains duplicates")

    merged_tasks: dict[str, list[dict]] = {}
    seed_base = None
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_config") != args.task_config:
            raise ValueError(
                f"{path}: task_config={payload.get('task_config')!r}, "
                f"expected {args.task_config!r}"
            )
        if int(payload.get("per_task", -1)) != args.per_task:
            raise ValueError(
                f"{path}: per_task={payload.get('per_task')!r}, expected {args.per_task}"
            )
        current_seed_base = int(payload.get("seed_base", -1))
        if seed_base is None:
            seed_base = current_seed_base
        elif current_seed_base != seed_base:
            raise ValueError(f"{path}: seed_base={current_seed_base}, expected {seed_base}")

        tasks = payload.get("tasks")
        if not isinstance(tasks, dict):
            raise ValueError(f"{path}: tasks is not an object")
        for task, entries in tasks.items():
            if task in merged_tasks:
                raise ValueError(f"Task appears in multiple shards: {task}")
            if not isinstance(entries, list):
                raise ValueError(f"{path}: entries for {task} are not a list")
            merged_tasks[task] = entries

    missing = sorted(set(expected_tasks) - set(merged_tasks))
    extra = sorted(set(merged_tasks) - set(expected_tasks))
    if missing or extra:
        raise ValueError(f"Task set mismatch: missing={missing}, extra={extra}")

    for task in expected_tasks:
        entries = merged_tasks[task]
        if len(entries) != args.per_task:
            raise ValueError(f"{task}: got {len(entries)} entries, expected {args.per_task}")
        seeds = []
        for index, entry in enumerate(entries):
            validate_entry(task, index, entry)
            seeds.append(entry["seed"])
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{task}: duplicate seeds")

    output = {
        "version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "root": "/workspace/lingbot-va",
        "task_config": args.task_config,
        "seed_base": seed_base,
        "per_task": args.per_task,
        "generation": {
            "episode_info_mode": "play_once_without_motion",
            "expert_check": False,
            "source_sha256": args.source_sha256,
            "inputs": [str(path) for path in args.inputs],
        },
        "tasks": {task: merged_tasks[task] for task in expected_tasks},
    }
    atomic_write_json(args.output, output)
    print(
        f"Validated and wrote {len(expected_tasks)} tasks x {args.per_task} seeds "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
