"""Build a VLAC RGB index from all episodes in the fixed 50+50 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotwin_critic.two_stage_rft.protocol import (
    DOMAINS,
    STAGES,
    audit_protocol,
    iter_episode_refs,
)
from robotwin_critic.vlac_finetune.build_rgb_index import parquet_files, videos_exist
from robotwin_critic.vlac_finetune.common import DEFAULT_CAMERAS, write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_index(
    prepared_root: Path,
    output: Path,
    *,
    stages: tuple[str, ...] = STAGES,
    domains: tuple[str, ...] = DOMAINS,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    max_tasks: int = 0,
) -> dict:
    audit = audit_protocol(prepared_root)
    refs = list(iter_episode_refs(prepared_root, stages=stages, domains=domains))
    selected_tasks = sorted({ref.task for ref in refs})
    if max_tasks:
        selected_tasks = selected_tasks[:max_tasks]
    selected_task_set = set(selected_tasks)

    metadata_cache: dict[Path, tuple[dict, dict[int, dict], dict[int, Path]]] = {}
    rows = []
    skipped_rgb = 0
    for ref in refs:
        if ref.task not in selected_task_set:
            continue
        if ref.repo not in metadata_cache:
            with (ref.repo / "meta" / "info.json").open(encoding="utf-8") as handle:
                info = json.load(handle)
            episodes = {
                int(row["episode_index"]): row
                for row in read_jsonl(ref.repo / "meta" / "episodes.jsonl")
            }
            metadata_cache[ref.repo] = (info, episodes, parquet_files(ref.repo))
        info, episodes, parquets = metadata_cache[ref.repo]
        episode = episodes[ref.output_episode_index]
        parquet = parquets[ref.output_episode_index]
        if not videos_exist(
            ref.repo, parquet, ref.output_episode_index, list(cameras)
        ):
            skipped_rgb += 1
            continue
        text = (
            episode.get("tasks", [""])[0]
            if episode.get("tasks")
            else episode.get("action_config", [{}])[0].get("action_text", "")
        )
        rows.append(
            {
                "dataset_root": str(prepared_root.resolve()),
                "dataset_split": f"{ref.stage}:{ref.domain}",
                "stage": ref.stage,
                "domain": ref.domain,
                "task_dir": str(ref.repo.resolve()),
                "task_name": ref.task,
                "task": ref.task,
                "episode_index": ref.output_episode_index,
                "source_episode_index": ref.source_episode_index,
                "length": int(episode["length"]),
                "fps": info.get("fps"),
                "text": text,
                "parquet_path": str(parquet.resolve()),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    summary = {
        "protocol": audit,
        "stages": list(stages),
        "domains": list(domains),
        "tasks": len(selected_tasks),
        "episodes": len(rows),
        "skipped_missing_rgb": skipped_rgb,
        "uses_all_fixed_protocol_episodes": (
            not max_tasks
            and stages == STAGES
            and domains == DOMAINS
            and len(rows) + skipped_rgb == 5000
        ),
    }
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--max-tasks", type=int, default=0)
    args = parser.parse_args()
    summary = build_index(
        args.prepared_root,
        args.output,
        stages=tuple(args.stages),
        domains=tuple(args.domains),
        max_tasks=args.max_tasks,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("VLAC_FIXED_PROTOCOL_INDEX_OK")


if __name__ == "__main__":
    main()
