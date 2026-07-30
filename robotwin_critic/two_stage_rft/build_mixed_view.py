"""Build an immutable Stage-1 plus selected-chunk RFT training view.

The view contains tiny real metadata directories and relative symlinks to
payload directories. This lets the unchanged recursive WAM dataset discovery
see both sources without copying or modifying either source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path


PAYLOAD_DIRS = ("data", "latents", "videos")
CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_repos(root: Path) -> list[Path]:
    return sorted(path.parent.parent for path in root.glob("**/meta/info.json"))


def count_items(repo: Path) -> int:
    with (repo / "meta" / "info.json").open(encoding="utf-8") as handle:
        chunks_size = int(json.load(handle).get("chunks_size", 1000))
    count = 0
    for episode in read_jsonl(repo / "meta" / "episodes.jsonl"):
        episode_index = int(episode["episode_index"])
        chunk = episode_index // chunks_size
        for config in episode.get("action_config", []):
            start = int(config["start_frame"])
            end = int(config["end_frame"])
            if all(
                (
                    repo
                    / "latents"
                    / f"chunk-{chunk:03d}"
                    / camera
                    / f"episode_{episode_index:06d}_{start}_{end}.pth"
                ).is_file()
                for camera in CAMERAS
            ):
                count += 1
    return count


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def expose_repo(source: Path, destination: Path) -> None:
    metadata = source / "meta"
    if not (metadata / "info.json").is_file():
        raise FileNotFoundError(metadata / "info.json")
    destination_metadata = destination / "meta"
    destination_metadata.mkdir(parents=True)
    for path in metadata.iterdir():
        if path.is_file():
            link_file(path, destination_metadata / path.name)
    for name in PAYLOAD_DIRS:
        payload = source / name
        if payload.exists():
            link_directory(payload, destination / name)


def compute_rft_repeats(
    stage1_items: int,
    rft_items: int,
    target_fraction: float,
) -> int:
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("RFT target fraction must be between zero and one")
    if stage1_items <= 0 or rft_items <= 0:
        raise ValueError("Both Stage-1 and RFT views must contain training items")
    desired_rft_items = (
        target_fraction / (1.0 - target_fraction) * stage1_items
    )
    return max(1, int(math.ceil(desired_rft_items / rft_items)))


def build_view(
    stage1_root: Path,
    rft_root: Path,
    output_root: Path,
    *,
    rft_target_fraction: float,
) -> dict:
    stage1_root = stage1_root.expanduser().resolve()
    rft_root = rft_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite mixed view: {output_root}")
    stage1_repos = discover_repos(stage1_root)
    rft_repos = discover_repos(rft_root)
    if not stage1_repos or not rft_repos:
        raise ValueError(
            f"Missing repositories: stage1={len(stage1_repos)}, rft={len(rft_repos)}"
        )
    stage1_items = sum(count_items(repo) for repo in stage1_repos)
    rft_items = sum(count_items(repo) for repo in rft_repos)
    repeats = compute_rft_repeats(
        stage1_items, rft_items, rft_target_fraction
    )
    preparing = output_root.with_name(f"{output_root.name}.preparing")
    if preparing.exists():
        raise FileExistsError(f"Incomplete mixed view exists: {preparing}")
    preparing.mkdir(parents=True)
    try:
        empty_embedding = stage1_root / "empty_emb.pt"
        if not empty_embedding.is_file():
            raise FileNotFoundError(empty_embedding)
        link_file(empty_embedding, preparing / "empty_emb.pt")
        for index, repo in enumerate(stage1_repos):
            relative = repo.relative_to(stage1_root)
            expose_repo(
                repo,
                preparing / "stage1" / f"repo-{index:04d}" / relative.name,
            )
        for repeat in range(repeats):
            for index, repo in enumerate(rft_repos):
                relative = repo.relative_to(rft_root)
                expose_repo(
                    repo,
                    preparing
                    / "rft"
                    / f"repeat-{repeat:03d}"
                    / f"repo-{index:04d}"
                    / relative.name,
                )
        effective_rft_items = repeats * rft_items
        total_items = stage1_items + effective_rft_items
        manifest = {
            "schema_version": 1,
            "stage1_root": str(stage1_root),
            "rft_root": str(rft_root),
            "stage1_repositories": len(stage1_repos),
            "rft_repositories": len(rft_repos),
            "stage1_items": stage1_items,
            "rft_unique_items": rft_items,
            "rft_repeats": repeats,
            "rft_effective_items": effective_rft_items,
            "total_items": total_items,
            "requested_rft_fraction": rft_target_fraction,
            "effective_rft_fraction": effective_rft_items / total_items,
            "stage1_split_manifest_sha256": sha256_file(
                stage1_root.parent / "split_manifest.json"
            ),
            "rft_manifest_sha256": sha256_file(
                rft_root / "rft_manifest.json"
            ),
        }
        with (preparing / "mixed_view_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        preparing.rename(output_root)
        return manifest
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--rft-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rft-target-fraction", type=float, default=0.25)
    args = parser.parse_args()
    manifest = build_view(
        args.stage1_root,
        args.rft_root,
        args.output_root,
        rft_target_fraction=args.rft_target_fraction,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("ROBOTWIN_MIXED_RFT_VIEW_OK")


if __name__ == "__main__":
    main()
