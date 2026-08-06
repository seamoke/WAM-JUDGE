"""Verify that an action-only RFT checkpoint changed only intended modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from robotwin_critic.two_stage_rft.train_action_only_rft import (
    ACTION_MODULES,
    is_action_parameter,
)


KNOWN_UNUSED_BASE_KEYS = {"patch_embedding.bias", "patch_embedding.weight"}


def weight_map(directory: Path) -> dict[str, Path]:
    index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {
            key: directory / filename
            for key, filename in index["weight_map"].items()
        }
    single = directory / "diffusion_pytorch_model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint found in {directory}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single for key in handle.keys()}


def load_tensor(mapping: dict[str, Path], key: str) -> torch.Tensor:
    with safe_open(mapping[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def evenly_spaced(keys: list[str], count: int) -> list[str]:
    if count <= 0:
        raise ValueError("frozen sample count must be positive")
    if count == 1:
        return keys[:1]
    if len(keys) <= count:
        return keys
    return sorted(
        {keys[round(i * (len(keys) - 1) / (count - 1))] for i in range(count)}
    )


def compare(base_dir: Path, checkpoint_dir: Path, frozen_samples: int) -> dict:
    base = weight_map(base_dir)
    checkpoint = weight_map(checkpoint_dir)
    missing = sorted(set(base) - set(checkpoint))
    extra = sorted(set(checkpoint) - set(base))
    unexpected_missing = sorted(set(missing) - KNOWN_UNUSED_BASE_KEYS)
    if unexpected_missing or extra:
        raise ValueError(f"Checkpoint key mismatch: missing={missing[:5]} extra={extra[:5]}")

    comparable_keys = set(base) & set(checkpoint)

    action_keys = sorted(key for key in comparable_keys if is_action_parameter(key))
    frozen_keys = sorted(key for key in comparable_keys if not is_action_parameter(key))
    if not action_keys:
        raise ValueError(f"No keys matched action modules {ACTION_MODULES}")

    changed_action = []
    unchanged_action = []
    maximum_action_delta = 0.0
    action_parameters = 0
    for key in action_keys:
        before = load_tensor(base, key)
        after = load_tensor(checkpoint, key)
        action_parameters += before.numel()
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"Action tensor contract changed for {key}")
        if torch.equal(before, after):
            unchanged_action.append(key)
        else:
            changed_action.append(key)
            maximum_action_delta = max(
                maximum_action_delta,
                float((before.float() - after.float()).abs().max()),
            )

    sampled_frozen = evenly_spaced(frozen_keys, frozen_samples)
    changed_frozen = []
    for key in sampled_frozen:
        before = load_tensor(base, key)
        after = load_tensor(checkpoint, key)
        if before.shape != after.shape or before.dtype != after.dtype:
            changed_frozen.append(key)
        elif not torch.equal(before, after):
            changed_frozen.append(key)

    changed_action_modules = {
        module: any(f".{module}." in f".{key}." for key in changed_action)
        for module in ACTION_MODULES
    }
    result = {
        "base": str(base_dir.resolve()),
        "checkpoint": str(checkpoint_dir.resolve()),
        "action_modules": list(ACTION_MODULES),
        "base_tensors": len(base),
        "checkpoint_tensors": len(checkpoint),
        "known_unused_base_keys_omitted_on_save": missing,
        "action_tensors": len(action_keys),
        "action_parameters": action_parameters,
        "changed_action_tensors": len(changed_action),
        "changed_action_tensor_names": changed_action,
        "unchanged_action_tensors": len(unchanged_action),
        "unchanged_action_tensor_names": unchanged_action,
        "changed_action_modules": changed_action_modules,
        "maximum_action_delta": maximum_action_delta,
        "sampled_frozen_tensors": len(sampled_frozen),
        "changed_sampled_frozen_tensors": changed_frozen,
        "passed": all(changed_action_modules.values()) and not changed_frozen,
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-transformer", type=Path, required=True)
    parser.add_argument("--checkpoint-transformer", type=Path, required=True)
    parser.add_argument("--frozen-samples", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(
        args.base_transformer,
        args.checkpoint_transformer,
        args.frozen_samples,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("ACTION_ONLY_CHECKPOINT_AUDIT_OK")


if __name__ == "__main__":
    main()
