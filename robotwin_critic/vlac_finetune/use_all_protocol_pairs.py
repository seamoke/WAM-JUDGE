"""Promote the complete fixed-protocol VLAC pair pool into training.

This intentionally makes the existing validation manifest monitoring-only.
Generalization claims must come from held-out tasks/seeds or downstream
RoboTwin evaluation, not from this duplicated monitor split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotwin_critic.vlac_finetune.common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    train = read_jsonl(train_path)
    val = read_jsonl(val_path)
    if not train or not val:
        raise ValueError("Both train.jsonl and val.jsonl must be non-empty")
    backup = args.data_dir / "train_episode_holdout.jsonl"
    if backup.exists():
        raise FileExistsError(f"Refusing to repeat all-pairs promotion: {backup}")
    train_path.replace(backup)
    write_jsonl(train_path, train + val)
    summary_path = args.data_dir / "build_summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["original_train_samples"] = len(train)
    summary["monitor_only_val_samples"] = len(val)
    summary["final_train_samples"] = len(train) + len(val)
    summary["all_fixed_protocol_pairs_used_for_optimization"] = True
    summary["validation_warning"] = (
        "val.jsonl is duplicated into train.jsonl and is monitoring-only; "
        "use held-out simulator seeds/tasks for generalization metrics"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("VLAC_ALL_FIXED_PROTOCOL_PAIRS_PROMOTED")


if __name__ == "__main__":
    main()
