"""Run online RFT while owning one continuous SwanLab session."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
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
ONE_SHOT_BUFFER_EVENT = "ONE_SHOT_BUFFER_EVENT "
TORCHRUN_PREFIX = re.compile(r"^\[[^\]]+\]:")
UPDATE_OK = re.compile(r"ONLINE_UPDATE_OK index=(\d+)")
UPDATE_START = re.compile(r"ONLINE_UPDATE_START index=(\d+)")
COLLECT_START = re.compile(r"ONLINE_COLLECT_START index=(\d+)")


RUNTIME_CONFIG_ENV = {
    "paths.project_root": "PROJECT_ROOT",
    "paths.lingbot_root": "LINGBOT_ROOT",
    "paths.prepared_data_root": "PREPARED_DATA_ROOT",
    "paths.real_data_root": "REAL_DATA_ROOT",
    "paths.part2_root": "PART2_ROOT",
    "paths.online_root": "ONLINE_ROOT",
    "paths.initial_model": "INITIAL_MODEL",
    "paths.wam_model": "WAM_MODEL",
    "paths.base_model": "BASE_MODEL",
    "paths.vlac_model": "VLAC_MODEL",
    "paths.vlac_adapter": "VLAC_ADAPTER",
    "paths.contexts": "CONTEXTS",
    "paths.action_profile": "ACTION_PROFILE",
    "paths.split_manifest": "SPLIT_MANIFEST",
    "data.real_data_mode": "REAL_DATA_MODE",
    "data.expected_per_domain_total": "EXPECTED_PER_DOMAIN_TOTAL",
    "data.expected_stage1_per_domain": "EXPECTED_STAGE1_PER_DOMAIN",
    "data.allow_missing_latent_segments": "ALLOW_MISSING_LATENT_SEGMENTS",
    "data.history_frames": "HISTORY_FRAMES",
    "data.context_pool_multiplier": "CONTEXT_POOL_MULTIPLIER",
    "data.max_episode_frames": "MAX_EPISODE_FRAMES",
    "sampling.gpu_ids": "INFER_GPU_IDS",
    "sampling.remote_workers": "REMOTE_INFER_WORKERS",
    "sampling.remote_gpu_ids": "REMOTE_GPU_IDS",
    "sampling.q_per_round": "Q_PER_ROUND",
    "sampling.batch_size_per_gpu": "INFER_BATCH_SIZE_PER_GPU",
    "sampling.candidates_per_q": "CANDIDATES_PER_Q",
    "sampling.base_seed": "BASE_SEED",
    "critic.vlac_batch_size_per_gpu": "VLAC_BATCH_SIZE_PER_GPU",
    "critic.min_action_score": "MIN_ACTION_SCORE",
    "critic.min_process_score": "MIN_PROCESS_SCORE",
    "critic.max_pseudo_per_context": "MAX_PSEUDO_PER_CONTEXT",
    "critic.action_gate_policy": "ACTION_GATE_POLICY",
    "critic.action_workspace_scope": "ACTION_WORKSPACE_SCOPE",
    "replay.buffer_capacity": "BUFFER_CAPACITY",
    "replay.real_fraction": "REAL_FRACTION",
    "training.batch_size_per_gpu": "TRAIN_BATCH_SIZE_PER_GPU",
    "training.global_batch_size": "TRAIN_GLOBAL_BATCH",
    "training.gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
    "training.nnodes": "TRAIN_NNODES",
    "training.local_gpus_per_node": "TRAIN_LOCAL_NGPU",
    "training.master_addr": "TRAIN_MASTER_ADDR",
    "training.nccl_ib_disable": "NCCL_IB_DISABLE",
    "training.nccl_net": "NCCL_NET",
    "training.nccl_socket_ifname": "NCCL_SOCKET_IFNAME",
    "training.nccl_socket_family": "NCCL_SOCKET_FAMILY",
    "training.nccl_cumem_host_enable": "NCCL_CUMEM_HOST_ENABLE",
    "training.activation_checkpointing": "TRAIN_ACTIVATION_CHECKPOINTING",
    "training.pseudo_epochs_per_update": "PSEUDO_EPOCHS_PER_UPDATE",
    "training.update_steps_override": "UPDATE_STEPS",
    "training.max_updates": "MAX_UPDATES",
    "training.model_save_every_updates": "MODEL_SAVE_EVERY_UPDATES",
    "one_shot.enabled": "ONE_SHOT_MODE",
    "one_shot.target": "ONE_SHOT_TARGET",
    "one_shot.data_fraction": "ONE_SHOT_DATA_FRACTION",
    "one_shot.collect_root": "ONE_SHOT_COLLECT_ROOT",
    "one_shot.epochs": "ONE_SHOT_TRAIN_EPOCHS",
    "one_shot.plateau_min_delta": "ONE_SHOT_PLATEAU_MIN_DELTA",
    "one_shot.plateau_patience": "ONE_SHOT_PLATEAU_PATIENCE",
    "one_shot.max_per_episode": "ONE_SHOT_MAX_PER_EPISODE",
    "one_shot.progress_bins": "ONE_SHOT_PROGRESS_BINS",
    "one_shot.min_action_distance": "ONE_SHOT_MIN_ACTION_DISTANCE",
    "one_shot.min_mean_luma": "ONE_SHOT_MIN_MEAN_LUMA",
    "one_shot.min_std_luma": "ONE_SHOT_MIN_STD_LUMA",
    "one_shot.max_dark_fraction": "ONE_SHOT_MAX_DARK_FRACTION",
}

INTEGER_ENV = {
    "Q_PER_ROUND",
    "REMOTE_INFER_WORKERS",
    "INFER_BATCH_SIZE_PER_GPU",
    "CANDIDATES_PER_Q",
    "BASE_SEED",
    "VLAC_BATCH_SIZE_PER_GPU",
    "MAX_PSEUDO_PER_CONTEXT",
    "BUFFER_CAPACITY",
    "TRAIN_BATCH_SIZE_PER_GPU",
    "TRAIN_GLOBAL_BATCH",
    "GRADIENT_ACCUMULATION_STEPS",
    "TRAIN_NNODES",
    "TRAIN_LOCAL_NGPU",
    "TRAIN_ACTIVATION_CHECKPOINTING",
    "PSEUDO_EPOCHS_PER_UPDATE",
    "UPDATE_STEPS",
    "MAX_UPDATES",
    "MODEL_SAVE_EVERY_UPDATES",
    "EXPECTED_PER_DOMAIN_TOTAL",
    "EXPECTED_STAGE1_PER_DOMAIN",
    "ALLOW_MISSING_LATENT_SEGMENTS",
    "HISTORY_FRAMES",
    "MAX_EPISODE_FRAMES",
    "ONE_SHOT_MODE",
    "ONE_SHOT_TARGET",
    "ONE_SHOT_TRAIN_EPOCHS",
    "ONE_SHOT_PLATEAU_MIN_DELTA",
    "ONE_SHOT_PLATEAU_PATIENCE",
    "ONE_SHOT_MAX_PER_EPISODE",
    "ONE_SHOT_PROGRESS_BINS",
}

FLOAT_ENV = {
    "MIN_ACTION_SCORE",
    "MIN_PROCESS_SCORE",
    "REAL_FRACTION",
    "ONE_SHOT_DATA_FRACTION",
    "CONTEXT_POOL_MULTIPLIER",
    "ONE_SHOT_MIN_ACTION_DISTANCE",
    "ONE_SHOT_MIN_MEAN_LUMA",
    "ONE_SHOT_MIN_STD_LUMA",
    "ONE_SHOT_MAX_DARK_FRACTION",
}


def _environment_value(name: str) -> str | int | float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    if name in INTEGER_ENV:
        return int(value)
    if name in FLOAT_ENV:
        return float(value)
    return value


def build_runtime_config(online_root: Path, run_id: str) -> dict[str, Any]:
    """Capture every online-RFT setting once on the parent SwanLab run."""
    config: dict[str, Any] = {
        "stream": "online_dual_rft",
        "schema_version": 3,
        "swanlab_owner": "main_orchestrator",
        "run_lifecycle": "one_online_root_one_swanlab_run",
        "run_id": run_id,
        "paths.online_root": str(online_root.resolve()),
    }
    for config_name, environment_name in RUNTIME_CONFIG_ENV.items():
        value = _environment_value(environment_name)
        if value is not None:
            config[config_name] = value

    gpu_ids = [
        value.strip()
        for value in str(config.get("sampling.gpu_ids", "")).split(",")
        if value.strip()
    ]
    remote_workers = config.get("sampling.remote_workers", 0)
    if not isinstance(remote_workers, int):
        remote_workers = 0
    total_sampling_workers = len(gpu_ids) + remote_workers
    config["sampling.local_num_gpus"] = len(gpu_ids)
    config["sampling.num_gpus"] = total_sampling_workers
    q_per_round = config.get("sampling.q_per_round")
    if (
        total_sampling_workers > 0
        and isinstance(q_per_round, int)
        and q_per_round % total_sampling_workers == 0
    ):
        config["sampling.q_per_gpu"] = q_per_round // total_sampling_workers

    per_gpu = config.get("training.batch_size_per_gpu")
    global_batch = config.get("training.global_batch_size")
    nnodes = config.get("training.nnodes", 1)
    local_gpus = config.get("training.local_gpus_per_node", len(gpu_ids))
    training_world_size = (
        nnodes * local_gpus
        if isinstance(nnodes, int) and isinstance(local_gpus, int)
        else len(gpu_ids)
    )
    config["training.world_size"] = training_world_size
    if (
        training_world_size > 0
        and isinstance(per_gpu, int)
        and isinstance(global_batch, int)
        and per_gpu > 0
        and global_batch % (per_gpu * training_world_size) == 0
    ):
        config["training.gradient_accumulation_steps"] = global_batch // (
            per_gpu * training_world_size
        )

    capacity = config.get("replay.buffer_capacity")
    epochs = config.get("training.pseudo_epochs_per_update")
    real_fraction = config.get("replay.real_fraction")
    if (
        not config.get("training.update_steps_override")
        and isinstance(capacity, int)
        and isinstance(epochs, int)
        and isinstance(global_batch, int)
        and isinstance(real_fraction, float)
        and 0.0 < real_fraction < 1.0
    ):
        config["training.effective_update_steps"] = math.ceil(
            capacity * epochs / (global_batch * (1.0 - real_fraction))
        )
    elif config.get("training.update_steps_override"):
        config["training.effective_update_steps"] = config[
            "training.update_steps_override"
        ]
    return config


def runtime_config_metrics(config: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    for key, value in config.items():
        if isinstance(value, bool):
            metrics[f"online/config/{key}"] = float(value)
        elif isinstance(value, (int, float)):
            metrics[f"online/config/{key}"] = float(value)
    return metrics


def acquire_parent_lock(online_root: Path):
    """Prevent two parent drivers from writing the same online run concurrently."""
    lock_path = online_root / ".swanlab_parent.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"another SwanLab parent already owns ONLINE_ROOT: {online_root}"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


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


def log_startup_status(
    swanlab_module: Any, online_root: Path, runtime_config: dict[str, Any] | None = None
) -> dict[str, float]:
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
    if runtime_config is not None:
        metrics.update(runtime_config_metrics(runtime_config))
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
    parent_lock = acquire_parent_lock(args.online_root)
    run_id_path = args.online_root / "swanlab_run_id"
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = f"rft-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.getpid()}"
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
    log_dir = args.log_dir or args.online_root / "swanlab"

    swanlab.login(api_key=api_key, save=False)
    runtime_config = build_runtime_config(args.online_root, run_id)
    runtime_config.update(
        {
            "swanlab.project": args.project,
            "swanlab.group": args.group,
            "swanlab.name": args.name,
            "launcher.child_command": args.command,
        }
    )
    (args.online_root / "swanlab_runtime_config.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run = swanlab.init(
        project=args.project,
        group=args.group,
        name=args.name,
        config=runtime_config,
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
        startup_metrics = log_startup_status(
            swanlab, args.online_root, runtime_config
        )
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
            if line.startswith(ONE_SHOT_BUFFER_EVENT):
                payload = json.loads(line[len(ONE_SHOT_BUFFER_EVENT) :])
                swanlab.log(
                    {
                        f"one_shot_buffer/{key}": float(value)
                        for key, value in payload.items()
                        if isinstance(value, (int, float, bool))
                    },
                    step=int(payload.get("collect_index", 0)),
                )
                continue
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
            update_start = UPDATE_START.search(line)
            if update_start is not None:
                update_index = int(update_start.group(1))
                update_metrics = runtime_config_metrics(runtime_config)
                update_metrics.update(
                    {
                        "online/status/update_started": float(update_index + 1),
                        "online/status/training_running": 1.0,
                        "rft/update_round": float(update_index),
                    }
                )
                swanlab.log(update_metrics)
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
        parent_lock.close()


if __name__ == "__main__":
    main()
