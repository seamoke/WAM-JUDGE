"""Persist one-shot collection plateau state across launcher restarts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_plateau(
    state: dict | None,
    *,
    collect_index: int,
    selected: int,
    min_delta: int,
    patience: int,
) -> dict:
    if min_delta < 0 or patience <= 0 or selected < 0:
        raise ValueError("min_delta/selected must be non-negative and patience positive")
    if state is None:
        return {
            "schema_version": 1,
            "last_collect_index": collect_index,
            "last_selected": selected,
            "selected_delta": None,
            "consecutive_low_growth": 0,
            "min_delta": min_delta,
            "patience": patience,
            "stopped": False,
        }
    previous_index = int(state["last_collect_index"])
    if collect_index < previous_index:
        raise ValueError(
            f"collect_index moved backwards: {collect_index} < {previous_index}"
        )
    if collect_index == previous_index:
        return state
    delta = selected - int(state["last_selected"])
    consecutive = (
        int(state.get("consecutive_low_growth", 0)) + 1
        if delta < min_delta
        else 0
    )
    return {
        "schema_version": 1,
        "last_collect_index": collect_index,
        "last_selected": selected,
        "selected_delta": delta,
        "consecutive_low_growth": consecutive,
        "min_delta": min_delta,
        "patience": patience,
        "stopped": bool(state.get("stopped", False) or consecutive >= patience),
    }


def atomic_write(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--collect-index", type=int, required=True)
    parser.add_argument("--selected", type=int, required=True)
    parser.add_argument("--min-delta", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    state = update_plateau(
        read_state(args.state),
        collect_index=args.collect_index,
        selected=args.selected,
        min_delta=args.min_delta,
        patience=args.patience,
    )
    atomic_write(args.state, state)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
