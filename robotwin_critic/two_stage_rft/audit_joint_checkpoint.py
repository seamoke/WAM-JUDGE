"""Verify representative video, shared, and action weights changed in full RFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from robotwin_critic.two_stage_rft.audit_action_only_checkpoint import (
    KNOWN_UNUSED_BASE_KEYS,
    load_tensor,
    weight_map,
)


REQUIRED_GROUPS = {
    "video_input": ("patch_embedding_mlp.",),
    "video_output": ("proj_out.",),
    "shared_transformer": ("blocks.",),
    "text_condition": ("condition_embedder.text_embedder.",),
    "video_time_condition": (
        "condition_embedder.time_embedder.",
        "condition_embedder.time_proj.",
    ),
    "action_input": ("action_embedder.",),
    "action_output": ("action_proj_out.",),
    "action_time_condition": (
        "condition_embedder_action.time_embedder.",
        "condition_embedder_action.time_proj.",
    ),
}


def _matches(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def compare(base_dir: Path, checkpoint_dir: Path) -> dict:
    base = weight_map(base_dir)
    checkpoint = weight_map(checkpoint_dir)
    missing = sorted(set(base) - set(checkpoint))
    extra = sorted(set(checkpoint) - set(base))
    unexpected_missing = sorted(set(missing) - KNOWN_UNUSED_BASE_KEYS)
    if unexpected_missing or extra:
        raise ValueError(
            f"Checkpoint key mismatch: missing={unexpected_missing[:5]} extra={extra[:5]}"
        )
    comparable = sorted(set(base) & set(checkpoint))
    base_tensor_finite = {
        key: bool(torch.isfinite(load_tensor(base, key)).all()) for key in base
    }
    checkpoint_tensor_finite = {
        key: bool(torch.isfinite(load_tensor(checkpoint, key)).all())
        for key in checkpoint
    }
    group_results = {}
    base_finite = all(base_tensor_finite.values())
    checkpoint_finite = all(checkpoint_tensor_finite.values())
    for group, prefixes in REQUIRED_GROUPS.items():
        keys = [key for key in comparable if _matches(key, prefixes)]
        if not keys:
            raise ValueError(f"No checkpoint tensors found for required group {group}")
        changed = []
        maximum_delta = 0.0
        group_finite = True
        for key in keys:
            before = load_tensor(base, key)
            after = load_tensor(checkpoint, key)
            if before.shape != after.shape or before.dtype != after.dtype:
                raise ValueError(f"Tensor contract changed for {key}")
            before_finite = base_tensor_finite[key]
            after_finite = checkpoint_tensor_finite[key]
            group_finite = group_finite and before_finite and after_finite
            if not before_finite or not after_finite:
                continue
            if not torch.equal(before, after):
                changed.append(key)
                maximum_delta = max(
                    maximum_delta,
                    float((before.float() - after.float()).abs().max()),
                )
        group_results[group] = {
            "tensors": len(keys),
            "changed_tensors": len(changed),
            "changed_examples": changed[:8],
            "maximum_delta": maximum_delta,
            "finite": group_finite,
            "passed": group_finite and bool(changed),
        }
    result = {
        "base": str(base_dir.resolve()),
        "checkpoint": str(checkpoint_dir.resolve()),
        "base_tensors": len(base),
        "checkpoint_tensors": len(checkpoint),
        "known_unused_base_keys_omitted_on_save": missing,
        "base_finite": base_finite,
        "checkpoint_finite": checkpoint_finite,
        "finite": base_finite and checkpoint_finite,
        "nonfinite_base_tensors": sorted(
            key for key, finite in base_tensor_finite.items() if not finite
        ),
        "nonfinite_checkpoint_tensors": sorted(
            key for key, finite in checkpoint_tensor_finite.items() if not finite
        ),
        "groups": group_results,
        "passed": (
            base_finite
            and checkpoint_finite
            and all(group["passed"] for group in group_results.values())
        ),
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-transformer", type=Path, required=True)
    parser.add_argument("--checkpoint-transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.base_transformer, args.checkpoint_transformer)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("JOINT_RFT_CHECKPOINT_AUDIT_OK")


if __name__ == "__main__":
    main()
