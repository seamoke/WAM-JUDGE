#!/usr/bin/env python3
"""Stream model-weight deltas between a LingBot checkpoint and its base model."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


WEIGHT_BASENAME = "diffusion_pytorch_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare")
    add_compare_args(compare)

    watch = subparsers.add_parser("watch")
    add_compare_args(watch)
    watch.add_argument("--checkpoint-root", required=True, type=Path)
    watch.add_argument("--steps", default="5000,10000,15000,20000")
    watch.add_argument("--poll-seconds", type=int, default=60)
    watch.add_argument("--stable-polls", type=int, default=2)
    watch.add_argument("--retry-seconds", type=int, default=600)
    return parser.parse_args()


def add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-chunk-elements", type=int, default=8_000_000)
    parser.add_argument("--top-k", type=int, default=100)


def transformer_dir(path: Path) -> Path:
    candidate = path / "transformer"
    return candidate if candidate.is_dir() else path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_info(root: Path) -> dict[str, Any]:
    path = transformer_dir(root) / "config.json"
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_weight_map(root: Path) -> tuple[Path, dict[str, Path], list[Path]]:
    directory = transformer_dir(root)
    index_path = directory / f"{WEIGHT_BASENAME}.safetensors.index.json"
    mapping: dict[str, Path] = {}

    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        for key, filename in payload["weight_map"].items():
            mapping[key] = directory / filename
    else:
        direct = directory / f"{WEIGHT_BASENAME}.safetensors"
        files = [direct] if direct.is_file() else sorted(directory.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors weights found under {directory}")
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in mapping:
                        raise ValueError(f"Duplicate tensor key {key!r}")
                    mapping[key] = path

    files = sorted(set(mapping.values()))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing weight shards: {missing}")
    return directory, mapping, files


@dataclass
class Accumulator:
    elements: int = 0
    changed_elements: int = 0
    base_nonfinite: int = 0
    checkpoint_nonfinite: int = 0
    base_sq: float = 0.0
    checkpoint_sq: float = 0.0
    delta_sq: float = 0.0
    dot: float = 0.0
    abs_delta_sum: float = 0.0
    max_abs_delta: float = 0.0

    def add(self, base: torch.Tensor, checkpoint: torch.Tensor) -> None:
        base = base.detach().to(dtype=torch.float32, device="cpu")
        checkpoint = checkpoint.detach().to(dtype=torch.float32, device="cpu")
        delta = checkpoint - base
        abs_delta = delta.abs()
        self.elements += delta.numel()
        self.changed_elements += int(torch.count_nonzero(checkpoint != base))
        self.base_nonfinite += int(torch.count_nonzero(~torch.isfinite(base)))
        self.checkpoint_nonfinite += int(
            torch.count_nonzero(~torch.isfinite(checkpoint))
        )
        self.base_sq += float(torch.sum(base * base, dtype=torch.float64))
        self.checkpoint_sq += float(
            torch.sum(checkpoint * checkpoint, dtype=torch.float64)
        )
        self.delta_sq += float(torch.sum(delta * delta, dtype=torch.float64))
        self.dot += float(torch.sum(base * checkpoint, dtype=torch.float64))
        self.abs_delta_sum += float(torch.sum(abs_delta, dtype=torch.float64))
        if delta.numel():
            self.max_abs_delta = max(
                self.max_abs_delta, float(torch.max(abs_delta))
            )

    def merge(self, other: "Accumulator") -> None:
        for field_name in (
            "elements",
            "changed_elements",
            "base_nonfinite",
            "checkpoint_nonfinite",
            "base_sq",
            "checkpoint_sq",
            "delta_sq",
            "dot",
            "abs_delta_sum",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.max_abs_delta = max(self.max_abs_delta, other.max_abs_delta)

    def summary(self) -> dict[str, Any]:
        base_l2 = math.sqrt(max(self.base_sq, 0.0))
        checkpoint_l2 = math.sqrt(max(self.checkpoint_sq, 0.0))
        delta_l2 = math.sqrt(max(self.delta_sq, 0.0))
        cosine_denominator = base_l2 * checkpoint_l2
        return {
            "elements": self.elements,
            "changed_elements": self.changed_elements,
            "changed_fraction": (
                self.changed_elements / self.elements if self.elements else None
            ),
            "base_nonfinite": self.base_nonfinite,
            "checkpoint_nonfinite": self.checkpoint_nonfinite,
            "base_l2": base_l2,
            "checkpoint_l2": checkpoint_l2,
            "delta_l2": delta_l2,
            "relative_l2_delta": delta_l2 / base_l2 if base_l2 else None,
            "cosine_similarity": (
                self.dot / cosine_denominator if cosine_denominator else None
            ),
            "rms_delta": (
                math.sqrt(self.delta_sq / self.elements) if self.elements else None
            ),
            "mean_abs_delta": (
                self.abs_delta_sum / self.elements if self.elements else None
            ),
            "max_abs_delta": self.max_abs_delta,
        }


def module_group(key: str) -> str:
    parts = key.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else key


def tensor_chunks(
    base_handle: Any,
    checkpoint_handle: Any,
    key: str,
    max_chunk_elements: int,
):
    base_slice = base_handle.get_slice(key)
    checkpoint_slice = checkpoint_handle.get_slice(key)
    base_shape = tuple(base_slice.get_shape())
    checkpoint_shape = tuple(checkpoint_slice.get_shape())
    if base_shape != checkpoint_shape:
        raise ValueError(
            f"Shape mismatch for {key}: base={base_shape}, checkpoint={checkpoint_shape}"
        )
    if not base_shape:
        yield base_handle.get_tensor(key), checkpoint_handle.get_tensor(key)
        return

    row_elements = math.prod(base_shape[1:]) if len(base_shape) > 1 else 1
    rows_per_chunk = max(1, max_chunk_elements // max(1, row_elements))
    for start in range(0, base_shape[0], rows_per_chunk):
        end = min(base_shape[0], start + rows_per_chunk)
        yield base_slice[start:end], checkpoint_slice[start:end]


def compare_models(
    base_root: Path,
    checkpoint_root: Path,
    max_chunk_elements: int,
    top_k: int,
) -> dict[str, Any]:
    started = time.time()
    base_dir, base_map, base_files = build_weight_map(base_root)
    checkpoint_dir, checkpoint_map, checkpoint_files = build_weight_map(
        checkpoint_root
    )
    base_keys = set(base_map)
    checkpoint_keys = set(checkpoint_map)
    common_keys = sorted(base_keys & checkpoint_keys)
    missing_in_checkpoint = sorted(base_keys - checkpoint_keys)
    unexpected_in_checkpoint = sorted(checkpoint_keys - base_keys)
    if not common_keys:
        raise ValueError("Base and checkpoint have no tensor keys in common")

    total = Accumulator()
    groups: dict[str, Accumulator] = defaultdict(Accumulator)
    tensor_summaries: list[dict[str, Any]] = []

    with contextlib.ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in sorted(set(base_files + checkpoint_files))
        }
        for index, key in enumerate(common_keys, start=1):
            tensor_acc = Accumulator()
            for base_tensor, checkpoint_tensor in tensor_chunks(
                handles[base_map[key]],
                handles[checkpoint_map[key]],
                key,
                max_chunk_elements,
            ):
                tensor_acc.add(base_tensor, checkpoint_tensor)
            total.merge(tensor_acc)
            groups[module_group(key)].merge(tensor_acc)
            tensor_summary = {"name": key, **tensor_acc.summary()}
            tensor_summaries.append(tensor_summary)
            if index % 100 == 0 or index == len(common_keys):
                print(
                    f"[compare] tensors={index}/{len(common_keys)} "
                    f"elements={total.elements}",
                    flush=True,
                )

    top_relative = sorted(
        tensor_summaries,
        key=lambda item: item["relative_l2_delta"] or -1.0,
        reverse=True,
    )[:top_k]
    top_absolute = sorted(
        tensor_summaries,
        key=lambda item: item["delta_l2"],
        reverse=True,
    )[:top_k]
    result = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "base": str(base_root.resolve()),
        "checkpoint": str(checkpoint_root.resolve()),
        "base_transformer": str(base_dir.resolve()),
        "checkpoint_transformer": str(checkpoint_dir.resolve()),
        "base_config": config_info(base_root),
        "checkpoint_config": config_info(checkpoint_root),
        "base_weight_files": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in base_files
        ],
        "checkpoint_weight_files": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in checkpoint_files
        ],
        "tensor_key_audit": {
            "base_count": len(base_keys),
            "checkpoint_count": len(checkpoint_keys),
            "matched_count": len(common_keys),
            "missing_in_checkpoint": missing_in_checkpoint,
            "unexpected_in_checkpoint": unexpected_in_checkpoint,
        },
        "overall": total.summary(),
        "module_groups": {
            name: accumulator.summary()
            for name, accumulator in sorted(groups.items())
        },
        "top_tensors_by_relative_l2_delta": top_relative,
        "top_tensors_by_absolute_l2_delta": top_absolute,
        "elapsed_seconds": time.time() - started,
    }
    result["audit_ok"] = (
        not missing_in_checkpoint
        and not unexpected_in_checkpoint
        and result["overall"]["base_nonfinite"] == 0
        and result["overall"]["checkpoint_nonfinite"] == 0
    )
    return result


def write_result(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)

    text_path = output.with_suffix(".txt")
    overall = result["overall"]
    lines = [
        f"audit_ok={result['audit_ok']}",
        f"base={result['base']}",
        f"checkpoint={result['checkpoint']}",
        f"matched_tensors={result['tensor_key_audit']['matched_count']}",
        f"elements={overall['elements']}",
        f"changed_fraction={overall['changed_fraction']:.12g}",
        f"relative_l2_delta={overall['relative_l2_delta']:.12g}",
        f"cosine_similarity={overall['cosine_similarity']:.12g}",
        f"rms_delta={overall['rms_delta']:.12g}",
        f"mean_abs_delta={overall['mean_abs_delta']:.12g}",
        f"max_abs_delta={overall['max_abs_delta']:.12g}",
        f"base_nonfinite={overall['base_nonfinite']}",
        f"checkpoint_nonfinite={overall['checkpoint_nonfinite']}",
        f"elapsed_seconds={result['elapsed_seconds']:.3f}",
    ]
    text_path.write_text("\n".join(lines) + "\n")


def write_json_atomic(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)


def checkpoint_signature(checkpoint: Path) -> tuple[tuple[str, int, int], ...] | None:
    try:
        _, _, files = build_weight_map(checkpoint)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return None
    config = transformer_dir(checkpoint) / "config.json"
    if not config.is_file():
        return None
    paths = files + [config]
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    )


def update_watch_status(
    output_dir: Path,
    steps: list[int],
    completed: list[int],
    failures: dict[int, str],
) -> None:
    status = {
        "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "expected_steps": steps,
        "completed_steps": sorted(completed),
        "pending_steps": sorted(set(steps) - set(completed)),
        "failures": {str(key): value for key, value in sorted(failures.items())},
    }
    write_json_atomic(status, output_dir / "watch_status.json")


def valid_audit_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("audit_ok"))


def watch(args: argparse.Namespace) -> int:
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    completed = [
        step
        for step in steps
        if valid_audit_output(
            args.output / f"checkpoint_step_{step}_vs_base.json"
        )
    ]
    signatures: dict[int, tuple[tuple[str, int, int], ...]] = {}
    stable_counts: dict[int, int] = defaultdict(int)
    failures: dict[int, str] = {}
    retry_after: dict[int, float] = defaultdict(float)

    while set(completed) != set(steps):
        for step in steps:
            if step in completed or time.time() < retry_after[step]:
                continue
            checkpoint = args.checkpoint_root / f"checkpoint_step_{step}"
            signature = checkpoint_signature(checkpoint)
            if signature is None:
                stable_counts[step] = 0
                continue
            if signatures.get(step) == signature:
                stable_counts[step] += 1
            else:
                signatures[step] = signature
                stable_counts[step] = 1
            if stable_counts[step] < args.stable_polls:
                continue

            print(f"[watch] comparing checkpoint_step_{step}", flush=True)
            try:
                result = compare_models(
                    args.base,
                    checkpoint,
                    args.max_chunk_elements,
                    args.top_k,
                )
                result["checkpoint_step"] = step
                output = args.output / f"checkpoint_step_{step}_vs_base.json"
                write_result(result, output)
                if not result["audit_ok"]:
                    raise RuntimeError(
                        "Tensor-key or non-finite audit failed; inspect "
                        f"{output}"
                    )
                completed.append(step)
                failures.pop(step, None)
                print(
                    f"[watch] completed checkpoint_step_{step}: "
                    f"relative_l2_delta={result['overall']['relative_l2_delta']:.6g} "
                    f"cosine={result['overall']['cosine_similarity']:.9f}",
                    flush=True,
                )
            except Exception as exc:  # Keep the watcher alive for later retry.
                failures[step] = f"{type(exc).__name__}: {exc}"
                retry_after[step] = time.time() + args.retry_seconds
                print(f"[watch] checkpoint_step_{step} failed: {failures[step]}", flush=True)

        update_watch_status(args.output, steps, completed, failures)
        if set(completed) != set(steps):
            time.sleep(args.poll_seconds)
    print("[watch] all checkpoint comparisons completed", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "compare":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for compare")
        result = compare_models(
            args.base, args.checkpoint, args.max_chunk_elements, args.top_k
        )
        write_result(result, args.output)
        print(json.dumps(result["overall"], indent=2, sort_keys=True))
        return 0 if result["audit_ok"] else 3
    return watch(args)


if __name__ == "__main__":
    raise SystemExit(main())
