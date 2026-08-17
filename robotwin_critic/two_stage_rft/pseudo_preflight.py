"""CPU-side compatibility audit for real and pseudo RFT dataset items."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:  # Keep the module importable in lightweight environments.
    torch = None  # type: ignore[assignment]


FIELDS = ("latents", "text_emb", "actions", "actions_mask")
NUMERIC_FIELDS = ("latents", "text_emb", "actions")
EXPECTED_RANK = {"latents": 4, "text_emb": 2, "actions": 4, "actions_mask": 4}
MAX_PSEUDO_SCALE_RATIO = 10.0
MIN_REFERENCE_STD = 1e-6
# Quantiles use at most this many values.  The sample is deterministic: it uses
# evenly spread positions (including both endpoints) in the logical flattened
# concatenation of the validated tensors.  Counts and moments still use every
# value.
QUANTILE_SAMPLE_MAX_VALUES = 262_144


def unexpected_pseudo_preflight_failure_report(
    exc: Exception,
    *,
    sample_count: int,
    seed: int,
    frame_chunk_size: int,
) -> dict[str, Any]:
    """Build a summary-compatible report for an unexpected local audit failure."""
    sources = ("real", "pseudo")
    return {
        "sample_count": sample_count,
        "seed": seed,
        "frame_chunk_size": frame_chunk_size,
        "source_counts": {source: None for source in sources},
        "sampled_indices": {source: [] for source in sources},
        "observed_shapes": {
            source: {field: [] for field in FIELDS} for source in sources
        },
        "finite": {
            source: {field: False for field in NUMERIC_FIELDS} for source in sources
        },
        "stats": {
            source: {field: _empty_stats() for field in NUMERIC_FIELDS}
            | {"actions_mask": {"count": 0, "true_fraction": None}}
            for source in sources
        },
        "violations": [
            f"unexpected local preflight exception: {type(exc).__name__}: {exc}"
        ],
        "ok": False,
    }


def flatten_preflight_summary(report: dict[str, Any]) -> dict[str, int | float | bool | None]:
    """Select compact scalar preflight metrics for a dataset report."""
    summary: dict[str, int | float | bool | None] = {
        "pseudo_preflight_ok": bool(report["ok"]),
        "pseudo_preflight_requested_samples": int(report["sample_count"]),
        "pseudo_preflight_real_samples": len(report["sampled_indices"]["real"]),
        "pseudo_preflight_pseudo_samples": len(report["sampled_indices"]["pseudo"]),
    }
    for source in ("real", "pseudo"):
        for field in ("latents", "text_emb", "actions"):
            for stat in ("mean", "std", "min", "max", "p01", "p50", "p99"):
                summary[f"pseudo_preflight_{source}_{field}_{stat}"] = report[
                    "stats"
                ][source][field][stat]
        summary[f"pseudo_preflight_{source}_action_mask_true_fraction"] = report[
            "stats"
        ][source]["actions_mask"]["true_fraction"]
    return summary


class PseudoPreflightError(ValueError):
    """Raised after an audit report containing all detected violations is built."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("pseudo preflight failed: " + "; ".join(report["violations"]))


def spread_sample_indices(length: int, sample_count: int, seed: int) -> list[int]:
    """Choose one deterministic pseudorandom index from each spread stratum."""
    if length < 0:
        raise ValueError("dataset length must be non-negative")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(length, sample_count)
    if count == 0:
        return []
    rng = random.Random(seed)
    return [
        rng.randrange((slot * length) // count, ((slot + 1) * length) // count)
        for slot in range(count)
    ]


def _empty_stats() -> dict[str, int | float | None]:
    return {key: None for key in ("min", "max", "mean", "std", "p01", "p50", "p99")} | {"count": 0}


def _quantile_sample_positions(count: int) -> Any:
    """Return bounded, deterministic positions in a flattened value stream."""
    sample_count = min(count, QUANTILE_SAMPLE_MAX_VALUES)
    if sample_count <= 0:
        return torch.empty(0, dtype=torch.int64)
    if sample_count == count:
        return torch.arange(count, dtype=torch.int64)
    # Integer arithmetic makes the selection stable across PyTorch versions and
    # avoids rounding differences from linspace.  Since sample_count <= count,
    # these positions are unique and cover the complete stream from end to end.
    steps = torch.arange(sample_count, dtype=torch.int64)
    quotient, remainder = divmod(count - 1, sample_count - 1)
    # Split the product so positions remain safe for any tensor count that fits
    # in int64; `steps * (count - 1)` could otherwise overflow first.
    return steps * quotient + torch.div(
        steps * remainder, sample_count - 1, rounding_mode="floor"
    )


def _stats(tensors: list[Any]) -> dict[str, int | float | None]:
    if not tensors:
        return _empty_stats()
    count = sum(int(value.numel()) for value in tensors)
    if count == 0:
        return _empty_stats()

    positions = _quantile_sample_positions(count)
    sample_parts = []
    sample_offset = 0
    stream_offset = 0
    combined_count = 0
    combined_mean = 0.0
    combined_m2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for tensor in tensors:
        values = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        value_count = int(values.numel())
        if value_count == 0:
            continue

        tensor_variance, tensor_mean = torch.var_mean(values, unbiased=False)
        tensor_mean_value = float(tensor_mean)
        tensor_m2 = float(tensor_variance) * value_count
        if combined_count:
            delta = tensor_mean_value - combined_mean
            next_count = combined_count + value_count
            combined_m2 += tensor_m2 + delta * delta * combined_count * value_count / next_count
            combined_mean += delta * value_count / next_count
            combined_count = next_count
        else:
            combined_count = value_count
            combined_mean = tensor_mean_value
            combined_m2 = tensor_m2
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))

        sample_end = int(
            torch.searchsorted(positions, stream_offset + value_count, right=False)
        )
        if sample_end > sample_offset:
            local_positions = positions[sample_offset:sample_end] - stream_offset
            sample_parts.append(values[local_positions])
            sample_offset = sample_end
        stream_offset += value_count

    sample = torch.cat(sample_parts)
    quantiles = torch.quantile(
        sample, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64)
    )
    return {
        "count": count,
        "min": minimum,
        "max": maximum,
        "mean": combined_mean,
        "std": (combined_m2 / combined_count) ** 0.5,
        "p01": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p99": float(quantiles[2]),
    }


def _mask_stats(tensors: list[Any]) -> dict[str, int | float | None]:
    count = sum(value.numel() for value in tensors)
    return {
        "count": count,
        "true_fraction": (
            sum(int(value.sum().item()) for value in tensors) / count if count else None
        ),
    }


def _catastrophic_distribution_violations(
    stats: dict[str, dict[str, dict[str, int | float | None]]],
) -> list[str]:
    """Reject only clear collapse/scale failures, leaving ordinary drift observable."""
    violations = []
    for field in NUMERIC_FIELDS:
        real = stats["real"][field]
        pseudo = stats["pseudo"][field]
        if not real["count"] or not pseudo["count"]:
            continue
        real_std = float(real["std"])
        pseudo_std = float(pseudo["std"])
        if real_std > MIN_REFERENCE_STD and pseudo_std <= MIN_REFERENCE_STD:
            violations.append(
                f"pseudo {field} distribution collapsed: std={pseudo_std:.6g} "
                f"while real std={real_std:.6g}"
            )
        real_scale = max(abs(float(real["p01"])), abs(float(real["p99"])))
        pseudo_scale = max(abs(float(pseudo["p01"])), abs(float(pseudo["p99"])))
        allowed_scale = MAX_PSEUDO_SCALE_RATIO * max(real_scale, MIN_REFERENCE_STD)
        if pseudo_scale > allowed_scale:
            violations.append(
                f"pseudo {field} catastrophic scale drift: |p01/p99|={pseudo_scale:.6g} "
                f"exceeds {MAX_PSEUDO_SCALE_RATIO:g}x real scale={real_scale:.6g}"
            )
    return violations


def _normalized_shape(field: str, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if field == "text_emb" and len(shape) == 3 and shape[0] == 1:
        return shape[1:]
    if len(shape) == EXPECTED_RANK[field]:
        return shape
    return None


def build_pseudo_preflight_report(
    real_dataset: Any,
    pseudo_dataset: Any,
    frame_chunk_size: int,
    sample_count: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a JSON-serializable report; validation problems are data, not raises."""
    if torch is None:
        raise RuntimeError("PyTorch is required to run the pseudo preflight audit")
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    counts = {"real": len(real_dataset), "pseudo": len(pseudo_dataset)}
    indices = {
        "real": spread_sample_indices(counts["real"], sample_count, seed),
        "pseudo": spread_sample_indices(counts["pseudo"], sample_count, seed + 1),
    }
    report: dict[str, Any] = {
        "sample_count": sample_count,
        "seed": seed,
        "frame_chunk_size": frame_chunk_size,
        "source_counts": counts,
        "sampled_indices": indices,
        "observed_shapes": {source: {field: [] for field in FIELDS} for source in counts},
        "finite": {source: {field: True for field in NUMERIC_FIELDS} for source in counts},
        "stats": {
            source: {field: _empty_stats() for field in NUMERIC_FIELDS}
            | {"actions_mask": {"count": 0, "true_fraction": None}}
            for source in counts
        },
        "violations": [],
        "ok": False,
    }
    for source, count in counts.items():
        if count == 0:
            report["violations"].append(f"{source} dataset is empty")

    valid = {source: {field: [] for field in FIELDS} for source in counts}
    normalized = {source: {field: [] for field in FIELDS} for source in counts}

    python_state = random.getstate()
    numpy_state = np.random.get_state() if np is not None else None
    cpu_state = torch.random.get_rng_state().clone()
    cuda_initialized = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()] if cuda_initialized else None
    try:
        for source, dataset in (("real", real_dataset), ("pseudo", pseudo_dataset)):
            for index in indices[source]:
                try:
                    item = dataset[index]
                except Exception as exc:
                    report["violations"].append(
                        f"{source}[{index}] dataset access failed: {type(exc).__name__}: {exc}"
                    )
                    continue
                values: dict[str, Any] = {}
                for field in FIELDS:
                    value = item.get(field) if hasattr(item, "get") else None
                    values[field] = value
                    if not isinstance(value, torch.Tensor):
                        report["violations"].append(
                            f"{source}[{index}].{field} must be a torch.Tensor, got {type(value).__name__}"
                        )
                        continue
                    shape = tuple(int(size) for size in value.shape)
                    report["observed_shapes"][source][field].append(
                        {"index": index, "shape": list(shape)}
                    )
                    norm = _normalized_shape(field, shape)
                    if norm is None:
                        expected = "[L,D] or [1,L,D]" if field == "text_emb" else {
                            "latents": "[C,F,H,W]",
                            "actions": "[C,F,N,1]",
                            "actions_mask": "[C,F,N,1]",
                        }[field]
                        report["violations"].append(
                            f"{source}[{index}].{field} shape {shape} is invalid; expected {expected}"
                        )
                    else:
                        normalized[source][field].append(norm)
                        if field in ("actions", "actions_mask") and shape[-1] != 1:
                            report["violations"].append(
                                f"{source}[{index}].{field} last dimension must be 1, got {shape[-1]}"
                            )

                    if field == "actions_mask":
                        if value.dtype != torch.bool:
                            report["violations"].append(
                                f"{source}[{index}].actions_mask must have dtype torch.bool, got {value.dtype}"
                            )
                        else:
                            valid[source][field].append(value)
                    else:
                        try:
                            is_finite = bool(torch.isfinite(value).all().item())
                        except (RuntimeError, TypeError):
                            is_finite = False
                        report["finite"][source][field] &= is_finite
                        if not is_finite:
                            report["violations"].append(
                                f"{source}[{index}].{field} must be numeric and contain only finite values"
                            )
                        else:
                            valid[source][field].append(value)

                actions = values["actions"]
                mask = values["actions_mask"]
                if isinstance(actions, torch.Tensor) and isinstance(mask, torch.Tensor):
                    if tuple(actions.shape) != tuple(mask.shape):
                        report["violations"].append(
                            f"{source}[{index}].actions_mask shape {tuple(mask.shape)} must equal "
                            f"actions shape {tuple(actions.shape)}"
                        )
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    for source in counts:
        for field in NUMERIC_FIELDS:
            report["stats"][source][field] = _stats(valid[source][field])
        report["stats"][source]["actions_mask"] = _mask_stats(valid[source]["actions_mask"])

    report["violations"].extend(
        _catastrophic_distribution_violations(report["stats"])
    )

    for field in ("latents", "actions", "actions_mask"):
        for shape in normalized["pseudo"][field]:
            if shape[1] != frame_chunk_size:
                report["violations"].append(
                    f"pseudo {field} F must equal frame_chunk_size={frame_chunk_size}, got {shape[1]}"
                )

    signatures = {
        "latents": lambda shape: (shape[0], shape[2], shape[3]),
        "text_emb": lambda shape: shape,
        "actions": lambda shape: (shape[0], shape[2], shape[3]),
    }
    for field, signature in signatures.items():
        real_shapes = {signature(shape) for shape in normalized["real"][field]}
        pseudo_shapes = {signature(shape) for shape in normalized["pseudo"][field]}
        if len(real_shapes) > 1:
            report["violations"].append(
                f"real {field} comparison shapes are inconsistent: {sorted(real_shapes)}"
            )
        if len(pseudo_shapes) > 1:
            report["violations"].append(
                f"pseudo {field} comparison shapes are inconsistent: {sorted(pseudo_shapes)}"
            )
        if real_shapes and pseudo_shapes and real_shapes != pseudo_shapes:
            report["violations"].append(
                f"pseudo vs real {field} shape mismatch: real={sorted(real_shapes)}, "
                f"pseudo={sorted(pseudo_shapes)}"
            )
    report["ok"] = not report["violations"]
    return report


def audit_pseudo_preflight(
    real_dataset: Any,
    pseudo_dataset: Any,
    frame_chunk_size: int,
    sample_count: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a passing report, or raise with the complete failing report."""
    report = build_pseudo_preflight_report(
        real_dataset, pseudo_dataset, frame_chunk_size, sample_count, seed
    )
    if not report["ok"]:
        raise PseudoPreflightError(report)
    return report


def write_preflight_report(report: dict[str, Any], path: str | Path, *, atomic: bool = True) -> Path:
    """Write a report as JSON, optionally using same-directory atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if not atomic:
        destination.write_text(payload, encoding="utf-8")
        return destination
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination
