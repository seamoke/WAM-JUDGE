"""Normalize provenance fields stored with Stage-2 rollout records."""

from __future__ import annotations


def canonical_source_stage(row: dict) -> str:
    value = str(row.get("source_stage") or row.get("stage") or "unknown").lower()
    if value.startswith("stage1"):
        return "stage1"
    if value.startswith("stage2"):
        return "stage2"
    return value


def with_rollout_provenance(row: dict) -> dict:
    """Return a record with stable task/stage provenance without dropping metadata."""
    result = dict(row)
    result["source_task"] = str(row.get("source_task") or row.get("task") or "unknown")
    result["source_stage"] = canonical_source_stage(row)
    return result
