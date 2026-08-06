"""Run online RFT while owning one continuous SwanLab session."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from robotwin_critic.two_stage_rft.log_online_collection_swanlab import (
    OnlineCollectionLogger,
)

COLLECTION_EVENT = "SWANLAB_COLLECTION_EVENT "
METRIC_EVENT = "SWANLAB_METRIC_EVENT "


def parse_metric_event(line: str) -> tuple[dict, int | None] | None:
    if not line.startswith(METRIC_EVENT):
        return None
    payload = json.loads(line[len(METRIC_EVENT) :])
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("SwanLab metric event must contain a metrics object")
    step = payload.get("step")
    return metrics, None if step is None else int(step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--project", default="lingbot-va-robotwin-rft")
    parser.add_argument("--group", default="robotwin-stage1-real-stage2-pseudo")
    parser.add_argument("--name", default="robotwin-stage1-15000-dual-rft-1000")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--max-images-per-collect", type=int, default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child RFT command is required after --")

    api_key = os.getenv("SWANLAB_API_KEY")
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is required")
    import swanlab

    args.online_root.mkdir(parents=True, exist_ok=True)
    run_id_path = args.online_root / "swanlab_run_id"
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = f"rft-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.getpid()}"
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
    log_dir = args.log_dir or args.online_root / "swanlab"

    swanlab.login(api_key=api_key, save=False)
    run = swanlab.init(
        project=args.project,
        group=args.group,
        name=args.name,
        config={
            "stream": "online_dual_rft",
            "schema_version": 2,
            "swanlab_owner": "main_orchestrator",
        },
        mode="online",
        log_dir=str(log_dir),
        id=run_id,
        resume="allow",
    )
    try:
        try:
            (args.online_root / "swanlab_url.txt").write_text(
                str(run.url) + "\n", encoding="utf-8"
            )
        except (AttributeError, ValueError):
            pass
        collection_logger = OnlineCollectionLogger(
            swanlab,
            args.online_root,
            args.online_root / "swanlab_upload_state.json",
            run_id,
            args.max_images_per_collect,
        )
        collection_logger.log_completed()

        child_env = os.environ.copy()
        child_env["SWANLAB_PARENT_DRIVER"] = "1"
        child_env["SWANLAB_PARENT_RUN_ID"] = run_id
        child_env["LINGBOT_SWANLAB_EXTERNAL"] = "1"
        process = subprocess.Popen(
            args.command,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith(COLLECTION_EVENT):
                payload = json.loads(line[len(COLLECTION_EVENT) :])
                collect_index = int(payload["collect_index"])
                collection_logger.log_completed(through=collect_index)
                continue
            metric_event = parse_metric_event(line)
            if metric_event is not None:
                metrics, step = metric_event
                swanlab.log(metrics, step=step)
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
        swanlab.log({"system/child_exit_code": float(return_code)})
        if return_code:
            raise subprocess.CalledProcessError(return_code, args.command)
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
