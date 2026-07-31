"""Score candidate action arrays and append analytic critic diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robotwin_critic.two_stage_rft.kinematic_action_critic import (
    KinematicProfile,
    score_relative_actions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.5)
    args = parser.parse_args()
    profile = KinematicProfile.from_json(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = accepted = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            start_state = row.get("start_state")
            result = score_relative_actions(
                np.load(row["action_path"]),
                profile,
                start_state=None if start_state is None else np.asarray(start_state),
            )
            result["accepted"] = bool(
                result["accepted"] and result["action_score"] >= args.min_score
            )
            row["action_critic"] = result
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            accepted += int(result["accepted"])
    print(json.dumps({"total": total, "accepted": accepted, "output": str(args.output)}))
    print("ACTION_CHUNK_SCORING_OK")


if __name__ == "__main__":
    main()
