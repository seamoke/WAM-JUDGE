#!/usr/bin/env python3
"""Audit and atomically finalize an interrupted two-stage preparation."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path

from prepare_robotwin_two_stage_dataset import (
    audit_prepared_root,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparing-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-missing-latent-segments", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparing_root = args.preparing_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    manifest = read_json(preparing_root / "split_manifest.json")
    if len(manifest.get("tasks", [])) != 50:
        raise ValueError("Interrupted manifest does not contain all 50 tasks")
    summary = audit_prepared_root(
        preparing_root,
        manifest,
        allow_missing_latent_segments=args.allow_missing_latent_segments,
        require_complete_marker=False,
    )
    complete = {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "recovered_from_interrupted_preparation": True,
        "manifest_sha256": summary["manifest_sha256"],
        "summary": summary,
    }
    write_json(preparing_root / "PREPARATION_COMPLETE.json", complete)
    os.replace(preparing_root, output_root)
    print(f"TWO_STAGE_DATASET_FINALIZED {output_root}")


if __name__ == "__main__":
    main()
