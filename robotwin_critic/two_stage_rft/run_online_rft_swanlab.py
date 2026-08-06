"""Run online RFT while owning one continuous SwanLab session."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from robotwin_critic.two_stage_rft.log_online_collection_swanlab import (
    OnlineCollectionLogger,
)

COLLECTION_EVENT = "SWANLAB_COLLECTION_EVENT "
METRIC_EVENT = "SWANLAB_METRIC_EVENT "
TORCHRUN_PREFIX = re.compile(r"^\[[^\]]+\]:")
UPDATE_OK = re.compile(r"ONLINE_UPDATE_OK index=(\d+)")
COLLECT_START = re.compile(r"ONLINE_COLLECT_START index=(\d+)")


def parse_metric_event(line: str) -> tuple[dict, int | None] | None:
    line = TORCHRUN_PREFIX.sub("", line, count=1)
    if not line.startswith(METRIC_EVENT):
        return None
    payload = json.loads(line[len(METRIC_EVENT) :])
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("SwanLab metric event must contain a metrics object")
    step = payload.get("step")
    return metrics, None if step is None else int(step)


def _load_metric_upload_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"logged_updates": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault("logged_updates", [])
    return state


def _save_metric_upload_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def mark_training_metrics_logged(state_path: Path, update_index: int) -> None:
    state = _load_metric_upload_state(state_path)
    logged = {int(index) for index in state["logged_updates"]}
    logged.add(update_index)
    state["logged_updates"] = sorted(logged)
    _save_metric_upload_state(state_path, state)


def replay_completed_training_metrics(
    swanlab_module: Any, online_root: Path, state_path: Path
) -> list[int]:
    state = _load_metric_upload_state(state_path)
    logged = {int(index) for index in state["logged_updates"]}
    replayed: list[int] = []
    for update_dir in sorted((online_root / "updates").glob("update_*")):
        try:
            update_index = int(update_dir.name.removeprefix("update_"))
        except ValueError:
            continue
        train_log = update_dir / "train" / "train.log"
        if update_index in logged or not train_log.is_file():
            continue
        if not (update_dir / "model").is_dir():
            continue
        for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
            metric_event = parse_metric_event(line)
            if metric_event is None:
                continue
            metrics, step = metric_event
            swanlab_module.log(metrics, step=step)
        logged.add(update_index)
        replayed.append(update_index)
    if replayed:
        state["logged_updates"] = sorted(logged)
        _save_metric_upload_state(state_path, state)
    return replayed


def log_startup_status(swanlab_module: Any, online_root: Path) -> dict[str, float]:
    state_path = online_root / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    metrics = {
        "online/status/active": 1.0,
        "online/status/update_index": float(state.get("update_index", 0)),
        "rft/update_round": float(state.get("update_index", 0)),
        "online/status/collect_index": float(state.get("collect_index", 0)),
        "online/status/accepted_total": float(state.get("accepted_total", 0)),
        "online/status/consumed_total": float(state.get("consumed_total", 0)),
    }
    for environment_name, metric_name in (
        ("BUFFER_CAPACITY", "online/config/buffer_capacity"),
        ("Q_PER_ROUND", "online/config/q_per_round"),
        ("CANDIDATES_PER_Q", "online/config/candidates_per_q"),
        ("TRAIN_GLOBAL_BATCH", "online/config/train_global_batch"),
        ("PSEUDO_EPOCHS_PER_UPDATE", "online/config/pseudo_epochs_per_update"),
        ("REAL_FRACTION", "online/config/real_fraction"),
    ):
        value = os.getenv(environment_name)
        if value is not None:
            metrics[metric_name] = float(value)
    swanlab_module.log(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--project", default="lingbot-va-robotwin")
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
        metric_state_path = args.online_root / "swanlab_metric_upload_state.json"
        replay_completed_training_metrics(swanlab, args.online_root, metric_state_path)
        startup_metrics = log_startup_status(swanlab, args.online_root)
        current_update_round = int(startup_metrics["rft/update_round"])

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
                swanlab.log(
                    {
                        "online/status/collect_completed": float(collect_index + 1),
                        "online/status/collect_running": 0.0,
                        "rft/update_round": float(current_update_round),
                    },
                    step=collect_index,
                )
                continue
            metric_event = parse_metric_event(line)
            if metric_event is not None:
                metrics, step = metric_event
                swanlab.log(metrics, step=step)
                continue
            update_ok = UPDATE_OK.search(line)
            if update_ok is not None:
                update_index = int(update_ok.group(1))
                current_update_round = update_index + 1
                mark_training_metrics_logged(metric_state_path, update_index)
                swanlab.log(
                    {
                        "online/status/update_completed": float(update_index + 1),
                        "online/status/training_running": 0.0,
                        "rft/update_round": float(current_update_round),
                    }
                )
            collect_start = COLLECT_START.search(line)
            if collect_start is not None:
                collect_index = int(collect_start.group(1))
                swanlab.log(
                    {
                        "online/status/collect_started": float(collect_index + 1),
                        "online/status/collect_running": 1.0,
                        "rft/update_round": float(current_update_round),
                    },
                    step=collect_index,
                )
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
