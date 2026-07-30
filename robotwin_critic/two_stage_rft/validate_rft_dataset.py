"""Validate RFT metadata and optionally instantiate the unchanged WAM loader."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path


def structural_validate(root: Path) -> dict:
    with (root / "rft_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    repos = sorted(root.glob("rft_selected_chunks/*/meta/info.json"))
    episodes = 0
    for info_path in repos:
        repo = info_path.parent.parent
        with info_path.open(encoding="utf-8") as handle:
            info = json.load(handle)
        with (repo / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
            episode_rows = [json.loads(line) for line in handle if line.strip()]
        if len(episode_rows) != int(info["total_episodes"]):
            raise ValueError(f"{repo}: episode count differs from info.json")
        for row in episode_rows:
            episode = int(row["episode_index"])
            for config in row["action_config"]:
                start, end = int(config["start_frame"]), int(config["end_frame"])
                for camera in (
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                ):
                    latent = (
                        repo
                        / "latents"
                        / "chunk-000"
                        / camera
                        / f"episode_{episode:06d}_{start}_{end}.pth"
                    )
                    if not latent.is_file():
                        raise FileNotFoundError(latent)
            parquet = (
                repo / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
            )
            if not parquet.is_file():
                raise FileNotFoundError(parquet)
        episodes += len(episode_rows)
    if episodes != int(manifest["episodes"]):
        raise ValueError("RFT manifest episode count is inconsistent")
    return {"repos": len(repos), "episodes": episodes}


def loader_validate(root: Path, max_items: int) -> dict:
    os.environ["ROBOTWIN_DATASET_PATH"] = str(root)
    os.environ["ROBOTWIN_EMPTY_EMB_PATH"] = str(root / "empty_emb.pt")
    # Import the unchanged dataset/config modules without executing
    # wan_va/__init__.py, whose eager model imports are irrelevant here and can
    # fail when a data-only environment has a different diffusers/peft pair.
    code_root = Path(__file__).resolve().parents[2]
    package_paths = {
        "wan_va": code_root / "wan_va",
        "wan_va.configs": code_root / "wan_va" / "configs",
        "wan_va.dataset": code_root / "wan_va" / "dataset",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    config_module = importlib.import_module("wan_va.configs.va_robotwin_train_cfg")
    dataset_module = importlib.import_module(
        "wan_va.dataset.lerobot_latent_dataset"
    )
    va_robotwin_train_cfg = config_module.va_robotwin_train_cfg
    LatentLeRobotDataset = dataset_module.LatentLeRobotDataset

    va_robotwin_train_cfg.cfg_prob = 0.0
    repos = [
        path.parent.parent
        for path in root.glob("**/meta/info.json")
    ]
    datasets = [
        LatentLeRobotDataset(str(repo), config=va_robotwin_train_cfg)
        for repo in repos
    ]
    shapes = []
    total_items = sum(len(dataset.new_metas) for dataset in datasets)
    for dataset in datasets:
        for index in range(len(dataset.new_metas)):
            if len(shapes) >= max_items:
                break
            item = dataset[index]
            if item["latents"].shape[1] != item["actions"].shape[1]:
                raise ValueError(
                    f"Item {index}: latent/action temporal lengths differ: "
                    f"{item['latents'].shape} vs {item['actions'].shape}"
                )
            shapes.append(
                {
                    "latents": list(item["latents"].shape),
                    "actions": list(item["actions"].shape),
                    "actions_mask": list(item["actions_mask"].shape),
                    "text_emb": list(item["text_emb"].shape),
                }
            )
    return {
        "dataset_items": total_items,
        "checked_items": len(shapes),
        "shapes": shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--with-loader", action="store_true")
    parser.add_argument(
        "--loader-only",
        action="store_true",
        help="Skip rft_manifest structural checks and validate every repo via loader.",
    )
    parser.add_argument("--max-items", type=int, default=4)
    args = parser.parse_args()
    result = {}
    if not args.loader_only:
        result["structural"] = structural_validate(args.dataset_root)
    if args.with_loader or args.loader_only:
        result["unchanged_wam_loader"] = loader_validate(
            args.dataset_root, args.max_items
        )
    print(json.dumps(result, indent=2))
    print("RFT_DATASET_VALIDATION_OK")


if __name__ == "__main__":
    main()
