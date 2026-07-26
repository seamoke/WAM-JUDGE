"""Download a Hugging Face model snapshot into shared persistent storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="InternRobotics/VLAC")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
        token=args.token,
        resume_download=True,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "output_dir": str(output_dir.resolve()),
                "resolved_path": resolved,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
