"""Count the exact Stage-2 WAM-loader chunk budget without reading hidden actions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from robotwin_critic.two_stage_rft.data_access import (
    episode_metadata,
    latent_segment_exists,
)
from robotwin_critic.two_stage_rft.protocol import iter_episode_refs, sha256_file


def count_budget(
    prepared_root: Path,
    output: Path,
    *,
    max_episode_frames: int = 500,
) -> dict:
    metadata_cache: dict[Path, dict[int, dict]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    skipped_long = skipped_missing_latent = 0
    for ref in iter_episode_refs(prepared_root, stages=("stage2",)):
        if ref.repo not in metadata_cache:
            metadata_cache[ref.repo] = episode_metadata(ref.repo)
        episode = metadata_cache[ref.repo][ref.output_episode_index]
        for config in episode.get("action_config", []):
            start = int(config["start_frame"])
            end = int(config["end_frame"])
            if end <= start:
                continue
            if max_episode_frames and end - start > max_episode_frames:
                skipped_long += 1
                continue
            if not latent_segment_exists(
                ref.repo, ref.output_episode_index, start, end
            ):
                skipped_missing_latent += 1
                continue
            counts[(ref.task, ref.domain)] += 1
    groups = {
        f"{task}/{domain}": int(value)
        for (task, domain), value in sorted(counts.items())
    }
    result = {
        "schema_version": 1,
        "definition": "valid_action_config_items_in_original_WAM_loader_for_stage2",
        "prepared_root": str(prepared_root.resolve()),
        "split_manifest_sha256": sha256_file(
            prepared_root / "split_manifest.json"
        ),
        "reads_action_column": False,
        "max_episode_frames": max_episode_frames,
        "groups": groups,
        "total": sum(groups.values()),
        "skipped_long": skipped_long,
        "skipped_missing_latent": skipped_missing_latent,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episode-frames", type=int, default=500)
    args = parser.parse_args()
    result = count_budget(
        args.prepared_root,
        args.output,
        max_episode_frames=args.max_episode_frames,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("ROBOTWIN_PSEUDO_BUDGET_OK")


if __name__ == "__main__":
    main()
