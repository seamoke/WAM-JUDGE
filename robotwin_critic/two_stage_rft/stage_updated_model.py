"""Compose a complete WAM root around an updated transformer checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def stage_model(
    base_model: Path,
    transformer: Path,
    output: Path,
    *,
    move_transformer: bool = False,
    copy_transformer: bool = False,
) -> dict:
    base_model = base_model.expanduser().resolve()
    transformer = transformer.expanduser().resolve()
    output = output.expanduser().resolve()
    if not (base_model / "transformer" / "config.json").is_file():
        raise FileNotFoundError(f"Base model is incomplete: {base_model}")
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(f"Updated transformer is incomplete: {transformer}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite staged model: {output}")
    if move_transformer and copy_transformer:
        raise ValueError("move_transformer and copy_transformer are mutually exclusive")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        for entry in base_model.iterdir():
            if entry.name in {"transformer", "online_rft_model.json"}:
                continue
            (temporary / entry.name).symlink_to(entry.resolve())
        if move_transformer:
            transformer.rename(temporary / "transformer")
        elif copy_transformer:
            shutil.copytree(transformer, temporary / "transformer")
        else:
            (temporary / "transformer").symlink_to(transformer)
        transformer_storage = (
            "materialized"
            if move_transformer
            else "copied"
            if copy_transformer
            else "symlink"
        )
        manifest = {
            "base_model": str(base_model),
            "updated_transformer": str(output / "transformer")
            if move_transformer or copy_transformer
            else str(transformer),
            "parameter_scope": "full_transformer",
            "transformer_storage": transformer_storage,
        }
        (temporary / "online_rft_model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            if move_transformer and (temporary / "transformer").exists():
                (temporary / "transformer").rename(transformer)
            shutil.rmtree(temporary)
        raise
    return {**manifest, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--move-transformer", action="store_true")
    parser.add_argument("--copy-transformer", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            stage_model(
                args.base_model,
                args.transformer,
                args.output,
                move_transformer=args.move_transformer,
                copy_transformer=args.copy_transformer,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
