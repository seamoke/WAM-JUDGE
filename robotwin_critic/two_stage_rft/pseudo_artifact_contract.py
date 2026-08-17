"""Torch-free SHA-256 binding for pseudo training artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


ARTIFACT_PATH_FIELDS = ("latent_path", "text_emb_path", "action_path")
HASH_SUFFIX = "_sha256"


def sha256_stream(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_artifact_hashes(row: dict, *, jsonl_parent: str | Path | None = None) -> dict:
    """Bind artifact bytes and emit paths that keep identifying those exact bytes."""
    result = dict(row)
    parent = Path(jsonl_parent) if jsonl_parent is not None else None
    for field in ARTIFACT_PATH_FIELDS:
        raw = result.get(field)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"missing {field}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            if parent is None:
                raise ValueError(f"relative {field} requires jsonl_parent")
            path = parent / path
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{field} is not a file: {path}")
        result[field] = str(path)
        result[field + HASH_SUFFIX] = sha256_stream(path)
    return result


class PseudoArtifactContractError(ValueError):
    def __init__(self, row_count: int, counts: Counter[str], examples: dict[str, list[int]]):
        self.row_count, self.counts, self.examples = row_count, counts, examples
        detail = "; ".join(
            f"{issue}={count} (lines {','.join(map(str, examples[issue]))})"
            for issue, count in sorted(counts.items())
        )
        super().__init__(f"Pseudo artifact contract failed for {row_count} nonblank rows: {detail}")


def validate_pseudo_artifact_rows(
    rows: Iterable[object], *, jsonl_parent: str | Path, row_numbers: Iterable[int]
) -> int:
    parent = Path(jsonl_parent)
    rows, numbers = list(rows), list(row_numbers)
    if len(rows) != len(numbers):
        raise ValueError("row_numbers must have exactly one entry per row")
    counts: Counter[str] = Counter()
    examples: dict[str, list[int]] = {}

    def record(issue: str, line: int) -> None:
        counts[issue] += 1
        if len(examples.setdefault(issue, [])) < 3:
            examples[issue].append(line)

    for line, row in zip(numbers, rows):
        if not isinstance(row, dict):
            record("json_object", line)
            continue
        for field in ARTIFACT_PATH_FIELDS:
            raw, expected = row.get(field), row.get(field + HASH_SUFFIX)
            if not isinstance(raw, str) or not raw:
                record(f"missing_{field}", line)
                continue
            if not isinstance(expected, str) or len(expected) != 64 or any(
                char not in "0123456789abcdef" for char in expected
            ):
                record(f"missing_or_invalid_{field}{HASH_SUFFIX}", line)
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = parent / path
            try:
                actual = sha256_stream(path)
            except OSError:
                record(f"unreadable_{field}", line)
                continue
            if actual != expected:
                record(f"mismatched_{field}{HASH_SUFFIX}", line)
    if not rows:
        record("no_nonblank_rows", 0)
    if counts:
        raise PseudoArtifactContractError(len(rows), counts, examples)
    return len(rows)


def validate_pseudo_artifact_jsonl(path: str | Path) -> int:
    jsonl = Path(path).expanduser().resolve()
    rows, numbers = [], []
    counts: Counter[str] = Counter()
    examples: dict[str, list[int]] = {}
    with jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
                numbers.append(line_number)
            except json.JSONDecodeError:
                counts["malformed_json"] += 1
                if len(examples.setdefault("malformed_json", [])) < 3:
                    examples["malformed_json"].append(line_number)
    try:
        count = validate_pseudo_artifact_rows(
            rows, jsonl_parent=jsonl.parent, row_numbers=numbers
        )
    except PseudoArtifactContractError as error:
        counts.update(error.counts)
        for issue, lines in error.examples.items():
            examples.setdefault(issue, []).extend(lines[: 3 - len(examples.get(issue, []))])
        raise PseudoArtifactContractError(len(rows) + counts["malformed_json"], counts, examples)
    if counts:
        raise PseudoArtifactContractError(count + counts["malformed_json"], counts, examples)
    return count
