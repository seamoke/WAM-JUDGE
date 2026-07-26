#!/usr/bin/env python3
"""Precompute stable RoboTwin eval seeds and episode metadata.

The eval client can consume this JSON through ROBOTWIN_SEED_CACHE to avoid
repeating the seed hunt and episode-info generation for every checkpoint run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ROBOTWIN_DIR", str(ROOT / "third_party" / "RoboTwin"))
if os.environ.get("ROBOTWIN_VULKAN_GPU") in (None, "", "void"):
    os.environ["ROBOTWIN_VULKAN_GPU"] = "0"
if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "void"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["ROBOTWIN_VULKAN_GPU"]
if os.environ.get("NVIDIA_VISIBLE_DEVICES") in (None, "", "void"):
    os.environ["NVIDIA_VISIBLE_DEVICES"] = os.environ["ROBOTWIN_VULKAN_GPU"]

from evaluation.robotwin.eval_polict_client_openpi import (  # noqa: E402
    _env_flag,
    class_decorator,
    get_embodiment_config,
)
from evaluation.robotwin.vulkan_env import vulkan_gpu_sim_slot  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402


ALL_TASKS = [
    "stack_bowls_three", "handover_block", "hanging_mug", "scan_object", "lift_pot",
    "put_object_cabinet", "stack_blocks_three", "place_shoe",
    "adjust_bottle", "place_mouse_pad", "dump_bin_bigbin", "move_pillbottle_pad",
    "pick_dual_bottles", "shake_bottle", "place_fan", "turn_switch",
    "shake_bottle_horizontally", "place_container_plate", "rotate_qrcode",
    "place_object_stand", "put_bottles_dustbin", "move_stapler_pad",
    "place_burger_fries", "place_bread_basket",
    "pick_diverse_bottles", "open_microwave", "beat_block_hammer", "press_stapler",
    "click_bell", "move_playingcard_away", "open_laptop", "move_can_pot",
    "stack_bowls_two", "place_a2b_right", "stamp_seal", "place_object_basket",
    "handover_mic", "place_bread_skillet", "stack_blocks_two", "place_cans_plasticbox",
    "click_alarmclock", "blocks_ranking_size", "place_phone_stand", "place_can_basket",
    "place_object_scale", "place_a2b_left", "grab_roller", "place_dual_shoes",
    "place_empty_cup", "blocks_ranking_rgb",
]

_MOTION_PLAN_METHODS = (
    "back_to_origin",
    "close_gripper",
    "grasp_actor",
    "move_by_displacement",
    "move_to_pose",
    "open_gripper",
    "place_actor",
)
_MOTION_EXEC_METHODS = ("delay", "move")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_name = f.name
    os.replace(tmp_name, path)


def _empty_motion_plan(_self, *_args, **_kwargs):
    return []


def _skip_motion(_self, *_args, **_kwargs):
    return None


def _successful_check(_self, *_args, **_kwargs):
    return True


def fill_episode_info_without_motion(task_env) -> dict:
    """Populate language metadata without executing the expert trajectory."""
    fill_info = getattr(task_env, "fill_episode_info_for_eval", None)
    if callable(fill_info):
        fill_info()
        info = task_env.info.get("info") or {}
        if info:
            return info

    for name in _MOTION_PLAN_METHODS:
        if hasattr(task_env, name):
            setattr(task_env, name, MethodType(_empty_motion_plan, task_env))
    for name in _MOTION_EXEC_METHODS:
        if hasattr(task_env, name):
            setattr(task_env, name, MethodType(_skip_motion, task_env))
    if hasattr(task_env, "check_success"):
        task_env.check_success = MethodType(_successful_check, task_env)

    episode_info = task_env.play_once()
    info = (episode_info or {}).get("info") or task_env.info.get("info") or {}
    if not info:
        raise RuntimeError(
            f"{type(task_env).__name__}.play_once() produced no episode info "
            "when motion execution was disabled"
        )
    return info


def load_or_init_cache(path: Path, task_config: str, seed_base: int, per_task: int) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
        "root": str(ROOT),
        "task_config": task_config,
        "seed_base": seed_base,
        "per_task": per_task,
        "tasks": {},
    }


def prepare_task_args(task_name: str, task_config: str) -> dict:
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "seed_cache"
    args["save_root"] = str(ROOT / "train_out" / "robotwin" / "eval_seed_cache")
    args["eval_mode"] = True
    args["render_freq"] = 0

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {name}")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def install_vulkan_icd() -> None:
    src = ROOT / "script" / "nvidia_icd.json"
    dst = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if not src.is_file():
        raise FileNotFoundError(f"Missing Vulkan ICD source: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        dst.write_bytes(src.read_bytes())
        print(f"Installed NVIDIA Vulkan ICD at {dst}")


def precompute_task(task_name: str, args: argparse.Namespace, cache: dict) -> None:
    entries = cache.setdefault("tasks", {}).setdefault(task_name, [])
    seen = {int(item["seed"]) for item in entries if "seed" in item}
    next_seed = max(seen) + 1 if seen else args.seed_base
    expert_check = args.expert_check
    failures = 0

    task_args = prepare_task_args(task_name, args.task_config)
    task_env = class_decorator(task_name)
    print(f"[{task_name}] existing={len(entries)} target={args.per_task} start_seed={next_seed}")

    while len(entries) < args.per_task:
        seed = next_seed
        next_seed += 1
        if seed in seen:
            continue

        try:
            task_env.setup_demo(now_ep_num=len(entries), seed=seed, is_test=True, **task_args)
            if expert_check:
                episode_info = task_env.play_once()
                valid = task_env.plan_success and task_env.check_success()
                info = episode_info.get("info", {}) if valid else {}
            else:
                info = fill_episode_info_without_motion(task_env)
                valid = True
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            failures += 1
            continue
        except Exception as exc:
            task_env.close_env()
            print(f"[{task_name}] seed={seed} failed: {exc}")
            if args.verbose_errors:
                traceback.print_exc()
            failures += 1
            if failures >= args.max_attempts_per_task and len(entries) < args.per_task:
                raise RuntimeError(
                    f"[{task_name}] reached {failures} failed attempts without enough valid seeds "
                    f"({len(entries)}/{args.per_task})"
                ) from exc
            continue

        if not valid:
            failures += 1
            continue

        entries.append({"seed": seed, "episode_info": jsonable(info)})
        seen.add(seed)
        failures = 0
        cache["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(args.output, cache)
        print(f"[{task_name}] cached {len(entries)}/{args.per_task} seed={seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default=os.environ.get("TASK_CONFIG", "demo_clean"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--per-task", type=int, default=int(os.environ.get("SEEDS_PER_TASK", "100")))
    parser.add_argument("--tasks", default="", help="Comma-separated task names. Empty means all tasks.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-attempts-per-task", type=int, default=int(os.environ.get("MAX_SEED_ATTEMPTS_PER_TASK", "10000")))
    parser.add_argument("--verbose-errors", action="store_true")
    parser.add_argument(
        "--expert-check",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ROBOTWIN_EXPERT_CHECK", False),
        help="Run expert play_once/check_success while sampling seeds. Slower but stricter.",
    )
    args = parser.parse_args()
    args.seed_base = 10000 * (1 + args.seed)
    if args.output is None:
        args.output = (
            ROOT
            / "train_out"
            / "robotwin"
            / "eval_seed_cache"
            / f"{args.task_config}_seed{args.seed}_n{args.per_task}.json"
        )
    return args


def main() -> None:
    args = parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()] or ALL_TASKS
    unknown = sorted(set(tasks) - set(ALL_TASKS))
    if unknown:
        raise SystemExit(f"Unknown task(s): {', '.join(unknown)}")

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    install_vulkan_icd()
    cache = load_or_init_cache(args.output, args.task_config, args.seed_base, args.per_task)
    with vulkan_gpu_sim_slot():
        for task_name in tasks:
            precompute_task(task_name, args, cache)
    print(f"Seed cache ready: {args.output}")


if __name__ == "__main__":
    main()
