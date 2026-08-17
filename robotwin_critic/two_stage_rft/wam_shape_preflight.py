"""Fail-fast WAM attention-shape checks for distributed RFT startup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


_ATTN_WEIGHT = re.compile(r"(?:^|\.)blocks\.(\d+)\.attn2\.to_([qkv])\.weight$")


def _weight_files(transformer: Path) -> list[Path]:
    index = transformer / "diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        manifest = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(manifest["weight_map"].values()))
        return [transformer / name for name in names]
    direct = transformer / "diffusion_pytorch_model.safetensors"
    if direct.is_file():
        return [direct]
    files = sorted(transformer.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors weights under {transformer}")
    return files


def checkpoint_attention_report(transformer_path: str | Path, *, rank: int) -> dict[str, Any]:
    """Inspect config and safetensors headers without loading model tensors."""
    from safetensors import safe_open

    transformer = Path(transformer_path).expanduser().resolve()
    config_path = transformer / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    expected = heads * head_dim
    if heads <= 0 or head_dim <= 0:
        raise RuntimeError(f"rank={rank} invalid attention config: heads={heads}, head_dim={head_dim}")

    tensor_shapes: dict[str, list[int]] = {}
    projection_shapes: dict[str, list[int]] = {}
    for weight_file in _weight_files(transformer):
        if not weight_file.is_file():
            raise RuntimeError(f"rank={rank} missing checkpoint shard: {weight_file}")
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shape = list(handle.get_slice(name).get_shape())
                tensor_shapes[name] = shape
                if _ATTN_WEIGHT.search(name):
                    projection_shapes[name] = shape

    if not projection_shapes:
        raise RuntimeError(f"rank={rank} found no blocks.*.attn2.to_[qkv].weight tensors in {transformer}")
    malformed = {
        name: shape
        for name, shape in projection_shapes.items()
        if len(shape) != 2 or shape[0] != expected or shape[1] != expected
    }
    if malformed:
        raise RuntimeError(
            f"rank={rank} checkpoint attention projection mismatch: expected=[{expected}, {expected}], "
            f"actual={json.dumps(malformed, sort_keys=True)} path={transformer}"
        )
    shape_digest = hashlib.sha256(
        json.dumps(tensor_shapes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "rank": rank,
        "transformer_path": str(transformer),
        "num_attention_heads": heads,
        "attention_head_dim": head_dim,
        "expected_attention_inner_dim": expected,
        "attention_projection_count": len(projection_shapes),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "tensor_shape_manifest_sha256": shape_digest,
    }


def require_matching_rank_reports(local_report: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    """Reject logical checkpoint differences across ranks; paths may differ by node."""
    errors = [report for report in reports if "error" in report]
    if errors:
        raise RuntimeError(
            "distributed WAM checkpoint preflight failed before model load: "
            + json.dumps(reports, sort_keys=True)
        )
    if local_report not in reports:
        raise RuntimeError("local checkpoint report is absent from gathered rank reports")
    comparable_keys = (
        "num_attention_heads",
        "attention_head_dim",
        "expected_attention_inner_dim",
        "attention_projection_count",
        "config_sha256",
        "tensor_shape_manifest_sha256",
    )
    signatures = {
        tuple(report[key] for key in comparable_keys)
        for report in reports
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "distributed WAM checkpoint/config mismatch before model load: "
            + json.dumps(reports, sort_keys=True)
        )


def loaded_model_attention_report(model: Any, *, rank: int) -> dict[str, Any]:
    """Validate logical Linear metadata after activation-checkpoint/FSDP wrapping."""
    expected = int(model.num_attention_heads) * int(model.attention_head_dim)
    malformed: dict[str, dict[str, int]] = {}
    blocks = list(model.blocks)
    for index, block in enumerate(blocks):
        attention = block.attn2
        for projection_name in ("to_q", "to_k", "to_v"):
            projection = getattr(attention, projection_name)
            actual = {
                "in_features": int(projection.in_features),
                "out_features": int(projection.out_features),
            }
            if actual != {"in_features": expected, "out_features": expected}:
                malformed[f"blocks.{index}.attn2.{projection_name}"] = actual
    if malformed:
        raise RuntimeError(
            f"rank={rank} loaded/FSDP model attention mismatch: expected_in_out={expected}, "
            f"actual={json.dumps(malformed, sort_keys=True)}"
        )
    return {
        "rank": rank,
        "blocks": len(blocks),
        "expected_attention_inner_dim": expected,
        "checked_attention_projections": len(blocks) * 3,
    }
