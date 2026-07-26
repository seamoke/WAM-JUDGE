"""Report VLAC training prerequisites without downloading or mutating the environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def executable_path(name: str) -> str | None:
    discovered = shutil.which(name)
    if discovered:
        return discovered
    for bin_dir in (Path(sys.executable).parent, Path(sys.base_prefix) / "bin"):
        candidate = bin_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def model_status(path: str) -> dict:
    model = Path(path)
    weights = sorted(model.glob("*.safetensors")) if model.is_dir() else []
    return {
        "path": str(model),
        "exists": model.is_dir(),
        "config": (model / "config.json").is_file(),
        "weight_files": [item.name for item in weights],
        "weight_bytes": sum(item.stat().st_size for item in weights),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-2b", default="/data/lingbot-va/models/vlac/VLAC-2B")
    parser.add_argument("--model-8b", default="/data/lingbot-va/models/vlac/VLAC-8B")
    parser.add_argument(
        "--vendor-root",
        default=str(Path(__file__).resolve().parent / "vendor" / "VLAC"),
    )
    args = parser.parse_args()
    vendor_root = Path(args.vendor_root)
    vendor_files = [
        vendor_root / "evo_vlac" / "__init__.py",
        vendor_root / "evo_vlac" / "utils" / "model_utils.py",
        vendor_root / "evo_vlac" / "utils" / "data_processing_vlm.py",
        vendor_root / "evo_vlac" / "utils" / "video_tool.py",
    ]
    report = {
        "executables": {
            name: executable_path(name) for name in ("swift", "torchrun", "ffmpeg")
        },
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "ms-swift",
                "datasets",
                "peft",
                "accelerate",
                "deepspeed",
                "opencv-python",
                "av",
                "flash-attn",
                "loguru",
                "scipy",
                "sentencepiece",
                "timm",
            )
        },
        "models": {
            "vlac_2b": model_status(args.model_2b),
            "vlac_8b": model_status(args.model_8b),
        },
        "vendor_runtime": {
            "path": str(vendor_root),
            "required_files": {str(path.relative_to(vendor_root)): path.is_file() for path in vendor_files},
        },
    }
    report["vendor_runtime_ready"] = all(
        report["vendor_runtime"]["required_files"].values()
    )
    report["training_ready"] = bool(
        report["executables"]["swift"]
        and report["packages"]["datasets"] == "3.6.0"
        and report["packages"]["flash-attn"]
        and report["packages"]["loguru"]
        and report["packages"]["sentencepiece"] == "0.1.99"
        and report["packages"]["timm"]
        and report["models"]["vlac_2b"]["config"]
        and report["models"]["vlac_2b"]["weight_bytes"] > 1 << 30
    )
    report["lora_ready"] = report["training_ready"] and bool(report["packages"]["peft"])
    report["full_8b_ready"] = bool(
        report["training_ready"]
        and report["packages"]["deepspeed"]
        and report["models"]["vlac_8b"]["weight_files"]
    )
    report["evaluation_ready"] = bool(
        report["training_ready"] and report["vendor_runtime_ready"]
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
