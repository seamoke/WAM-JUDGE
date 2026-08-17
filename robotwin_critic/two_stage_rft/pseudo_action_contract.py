"""Torch-free validation of pseudo action metadata used by RFT training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from robotwin_critic.two_stage_rft.pseudo_artifact_contract import (
    validate_pseudo_artifact_rows,
)


EXECUTABLE_ACTION_SEMANTICS = (
    "relative executable actions; first VA_Server conditioning block removed"
)
MAX_EXAMPLES_PER_ISSUE = 3
LEGACY_PSEUDO_ACTION_GATE_POLICY = "score_with_safety_gates"


class PseudoActionContractError(ValueError):
    """Aggregate pseudo action contract validation failure."""

    def __init__(
        self,
        row_count: int,
        counts: Counter[str],
        examples: dict[str, list[str]],
    ) -> None:
        self.row_count = row_count
        self.counts = counts
        self.examples = examples
        details = "; ".join(
            f"{issue}={count} [{', '.join(examples[issue])}]"
            for issue, count in sorted(counts.items())
        )
        super().__init__(
            f"Unsafe or ambiguous pseudo action contract in {row_count} nonblank "
            f"rows: {details}. Re-collect pseudo actions with strict gates."
        )


def _record(
    counts: Counter[str],
    examples: dict[str, list[str]],
    issue: str,
    line_number: int,
    row: object,
) -> None:
    counts[issue] += 1
    values = examples.setdefault(issue, [])
    if len(values) >= MAX_EXAMPLES_PER_ISSUE:
        return
    identity = "unknown"
    if isinstance(row, dict):
        identity = row.get("candidate_id", row.get("context_id", "unknown"))
    values.append(f"line {line_number} ({identity})")


def validate_pseudo_action_contract(
    rows: Iterable[dict],
    *,
    expected_latent_frames: int,
    action_per_frame: int,
    row_numbers: Iterable[int] | None = None,
    initial_issues: Iterable[tuple[str, int, object]] = (),
    allow_legacy_pseudo_action_metadata: bool = False,
) -> int:
    """Validate all rows and return their count, or raise one aggregate error."""
    if (
        type(expected_latent_frames) is not int
        or type(action_per_frame) is not int
        or expected_latent_frames <= 0
        or action_per_frame <= 0
    ):
        raise ValueError("Expected dimensions must be positive integers")
    expected_action_steps = (expected_latent_frames - 1) * action_per_frame
    rows = list(rows)
    numbers = list(row_numbers) if row_numbers is not None else list(range(1, len(rows) + 1))
    if len(numbers) != len(rows):
        raise ValueError("row_numbers must have exactly one entry per row")

    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for issue, row_number, row in initial_issues:
        _record(counts, examples, issue, row_number, row)

    for row_number, row in zip(numbers, rows):
        if not isinstance(row, dict):
            _record(counts, examples, "json_object", row_number, row)
            continue
        if row.get("action_semantics") != EXECUTABLE_ACTION_SEMANTICS:
            _record(counts, examples, "action_semantics", row_number, row)
        latent_frames = row.get("latent_frames")
        if type(latent_frames) is not int or latent_frames != expected_latent_frames:
            _record(counts, examples, "latent_frames", row_number, row)
        executable_action_steps = row.get("executable_action_steps")
        if (
            type(executable_action_steps) is not int
            or executable_action_steps != expected_action_steps
        ):
            _record(counts, examples, "executable_action_steps", row_number, row)

        critic = row.get("action_critic")
        if not isinstance(critic, dict):
            _record(counts, examples, "action_critic", row_number, row)
            continue
        if critic.get("accepted") is not True:
            _record(counts, examples, "action_critic.accepted", row_number, row)
        if not allow_legacy_pseudo_action_metadata and critic.get("hard_violations") != []:
            _record(counts, examples, "action_critic.hard_violations", row_number, row)
        if critic.get("gate_violations") != []:
            _record(counts, examples, "action_critic.gate_violations", row_number, row)
        expected_gate_policy = (
            LEGACY_PSEUDO_ACTION_GATE_POLICY
            if allow_legacy_pseudo_action_metadata
            else "strict"
        )
        if critic.get("gate_policy") != expected_gate_policy:
            _record(counts, examples, "action_critic.gate_policy", row_number, row)

    row_count = len(rows) + counts["malformed_json"]
    if row_count == 0:
        _record(counts, examples, "no_nonblank_rows", 0, {})
    if counts:
        raise PseudoActionContractError(row_count, counts, examples)
    return row_count


def verify_legacy_pseudo_action_waiver(
    path: str | Path, *, expected_sha256: str | None, expected_rows: int | None
) -> bool:
    """Verify the exact JSONL bytes and nonblank row count before it is parsed."""
    if (expected_sha256 is None) != (expected_rows is None):
        raise ValueError(
            "Legacy pseudo action waiver requires both expected SHA-256 and expected rows"
        )
    if expected_sha256 is None:
        return False
    if (
        len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError("Legacy pseudo action waiver SHA-256 must be 64 lowercase hex characters")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("Legacy pseudo action waiver rows must be a positive integer")
    digest = hashlib.sha256()
    actual_rows = 0
    with Path(path).expanduser().open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                actual_rows += 1
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Legacy pseudo action waiver SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    if actual_rows != expected_rows:
        raise ValueError(
            f"Legacy pseudo action waiver row-count mismatch: expected {expected_rows}, "
            f"got {actual_rows}"
        )
    return True


def validate_pseudo_action_jsonl(
    path: str | Path,
    *,
    expected_latent_frames: int,
    action_per_frame: int,
    legacy_pseudo_action_waiver_sha256: str | None = None,
    legacy_pseudo_action_waiver_rows: int | None = None,
) -> int:
    """Parse and validate every nonblank JSONL row without importing torch."""
    legacy_waiver = verify_legacy_pseudo_action_waiver(
        path,
        expected_sha256=legacy_pseudo_action_waiver_sha256,
        expected_rows=legacy_pseudo_action_waiver_rows,
    )
    rows: list[dict] = []
    row_numbers: list[int] = []
    parse_issues: list[tuple[str, int, object]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_issues.append(("malformed_json", line_number, None))
                continue
            rows.append(row)
            row_numbers.append(line_number)
    count = validate_pseudo_action_contract(
        rows,
        expected_latent_frames=expected_latent_frames,
        action_per_frame=action_per_frame,
        row_numbers=row_numbers,
        initial_issues=parse_issues,
        allow_legacy_pseudo_action_metadata=legacy_waiver,
    )
    validate_pseudo_artifact_rows(
        rows, jsonl_parent=Path(path).expanduser().resolve().parent,
        row_numbers=row_numbers,
    )
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected-latent-frames", type=int, required=True)
    parser.add_argument("--action-per-frame", type=int, required=True)
    parser.add_argument("--legacy-pseudo-action-waiver-sha256")
    parser.add_argument("--legacy-pseudo-action-waiver-rows", type=int)
    args = parser.parse_args(argv)
    try:
        count = validate_pseudo_action_jsonl(
            args.jsonl,
            expected_latent_frames=args.expected_latent_frames,
            action_per_frame=args.action_per_frame,
            legacy_pseudo_action_waiver_sha256=args.legacy_pseudo_action_waiver_sha256,
            legacy_pseudo_action_waiver_rows=args.legacy_pseudo_action_waiver_rows,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Pseudo action contract validation failed: {error}\n")
    executable = (args.expected_latent_frames - 1) * args.action_per_frame
    print(
        f"Pseudo action contract valid: rows={count} "
        f"latent_frames={args.expected_latent_frames} "
        f"executable_action_steps={executable}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
