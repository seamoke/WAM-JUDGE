"""Compose a complete WAM root around an updated transformer checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def stage_model(base_model: Path, transformer: Path, output: Path) -> dict:
    base_model = base_model.expanduser().resolve()
    transformer = transformer.expanduser().resolve()
    output = output.expanduser().resolve()
    if not (base_model / "transformer" / "config.json").is_file():
        raise FileNotFoundError(f"Base model is incomplete: {base_model}")
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(f"Updated transformer is incomplete: {transformer}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite staged model: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        for entry in base_model.iterdir():
            if entry.name in {"transformer", "online_rft_model.json"}:
                continue
            (temporary / entry.name).symlink_to(entry.resolve())
        (temporary / "transformer").symlink_to(transformer)
        manifest = {
            "base_model": str(base_model),
            "updated_transformer": str(transformer),
            "parameter_scope": "full_transformer",
        }
        (temporary / "online_rft_model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            for entry in temporary.iterdir():
                entry.unlink()
            temporary.rmdir()
        raise
    return {**manifest, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            stage_model(args.base_model, args.transformer, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
