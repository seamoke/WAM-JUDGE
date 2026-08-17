"""Torch-free checks for complete WAM transformer artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


SNAPSHOT_MARKER = "inference_snapshot.json"


def is_complete_transformer(path: str | Path) -> bool:
    """Return whether a transformer has config and non-empty official weights."""
    transformer = Path(path)
    try:
        config = transformer / "config.json"
        if not config.is_file() or not isinstance(json.loads(config.read_text()), dict):
            return False
        weights = transformer / "diffusion_pytorch_model.safetensors"
        if weights.is_file():
            return weights.stat().st_size > 0
        index = transformer / "diffusion_pytorch_model.safetensors.index.json"
        manifest = json.loads(index.read_text())
        shards = sorted(set(manifest["weight_map"].values()))
        return bool(shards) and all(
            (transformer / shard).is_file()
            and (transformer / shard).stat().st_size > 0
            and shard.endswith(".safetensors")
            for shard in shards
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def require_complete_transformer(path: str | Path, *, context: str) -> None:
    """Raise when an expected transformer artifact is absent or incomplete."""
    transformer = Path(path)
    if not is_complete_transformer(transformer):
        raise RuntimeError(f"{context} is incomplete: {transformer}")


def reject_existing_snapshot_targets(save_root: str | Path, steps: list[int]) -> None:
    """Fail closed rather than accepting artifacts left by another invocation."""
    checkpoints = Path(save_root) / "checkpoints"
    existing = [checkpoints / f"checkpoint_step_{step}" for step in steps]
    existing = [path for path in existing if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing stale/existing inference snapshot target(s): "
            + ", ".join(str(path) for path in existing)
        )


def write_snapshot_marker(checkpoint_dir: str | Path, invocation_id: str) -> Path:
    """Atomically attest that this invocation validated an inference snapshot."""
    if not invocation_id:
        raise ValueError("invocation_id must be non-empty")
    checkpoint = Path(checkpoint_dir)
    marker = checkpoint / SNAPSHOT_MARKER
    payload = {
        "artifact_kind": "inference_model_snapshot",
        "invocation_id": invocation_id,
        "resumable_optimizer_checkpoint": False,
        "transformer": "transformer",
    }
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=checkpoint,
            prefix=f".{marker.name}.", delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, marker)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return marker


def require_snapshot_invocation(checkpoint_dir: str | Path, invocation_id: str) -> None:
    """Require a validated inference snapshot belonging to this invocation."""
    checkpoint = Path(checkpoint_dir)
    require_complete_transformer(checkpoint / "transformer", context="inference snapshot")
    try:
        marker = json.loads((checkpoint / SNAPSHOT_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing/invalid inference snapshot marker: {checkpoint}") from exc
    expected = {
        "artifact_kind": "inference_model_snapshot",
        "invocation_id": invocation_id,
        "resumable_optimizer_checkpoint": False,
        "transformer": "transformer",
    }
    if marker != expected:
        raise RuntimeError(f"inference snapshot marker does not match this invocation: {checkpoint}")
