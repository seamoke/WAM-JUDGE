#!/usr/bin/env python3
"""Convert official LIBERO HDF5 demos to LingBot-VA LeRobot v2.1 layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import jsonlines
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from libero.libero.benchmark.libero_suite_task_map import libero_task_map

SUITES = ("libero_spatial", "libero_object", "libero_goal")
LIBERO_FPS = 20
IMAGE_FLIP_AXIS = 0  # match evaluation/libero/client.py vertical flip


def _task_instruction_from_hdf5(hdf5_path: Path, suite: str) -> str:
    with h5py.File(hdf5_path, "r") as hf:
        problem_info = hf["data"].attrs.get("problem_info")
        if problem_info is not None:
            if isinstance(problem_info, bytes):
                problem_info = problem_info.decode("utf-8")
            info = json.loads(problem_info)
            instruction = info.get("language_instruction")
            if instruction:
                return instruction.strip()

    stem = hdf5_path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    for task_name in libero_task_map.get(suite, []):
        if stem == task_name or stem.endswith(task_name):
            return task_name.replace("_", " ")
    return stem.replace("_", " ")


def _compose_state(ee_pos: np.ndarray, ee_ori: np.ndarray, gripper_states: np.ndarray) -> np.ndarray:
    return np.concatenate([ee_pos, ee_ori, gripper_states], axis=-1).astype(np.float32)


def _flip_image(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1])


def _lerobot_features() -> dict:
    video_info = {
        "video.height": 128,
        "video.width": 128,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": LIBERO_FPS,
        "video.channels": 3,
        "has_audio": False,
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            },
        },
        "observation.images.agentview_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "rgb"],
            "info": video_info,
        },
        "observation.images.eye_in_hand_rgb": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "rgb"],
            "info": dict(video_info),
        },
    }


def _output_root(output_dir: Path, suite: str) -> Path:
    return output_dir / suite / "0.0.0" / f"{suite}_0.0.0_lerobot_part_0"


def _add_action_config(meta_dir: Path) -> None:
    episodes_path = meta_dir / "episodes.jsonl"
    backup_path = meta_dir / "episodes_ori.jsonl"
    if not backup_path.exists():
        shutil.copy2(episodes_path, backup_path)

    entries = []
    with jsonlines.open(episodes_path) as reader:
        for episode in reader:
            task_text = episode["tasks"][0] if episode.get("tasks") else ""
            episode["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": episode["length"],
                    "action_text": task_text,
                    "skill": "",
                }
            ]
            entries.append(episode)

    with jsonlines.open(episodes_path, mode="w") as writer:
        writer.write_all(entries)


def convert_suite(hdf5_dir: Path, output_dir: Path, suite: str, overwrite: bool = False) -> Path:
    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {hdf5_dir}")

    root = _output_root(output_dir, suite)
    if root.exists():
        if not overwrite:
            print(f"[skip] {suite}: output exists at {root}")
            return root
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=suite,
        fps=LIBERO_FPS,
        features=_lerobot_features(),
        root=root,
        robot_type="Franka",
        use_videos=True,
        image_writer_threads=4,
    )

    for hdf5_path in tqdm(hdf5_files, desc=f"convert {suite}"):
        task = _task_instruction_from_hdf5(hdf5_path, suite)
        with h5py.File(hdf5_path, "r") as hf:
            demo_keys = sorted(hf["data"].keys(), key=lambda key: int(key.split("_")[1]))
            for demo_key in demo_keys:
                demo = hf["data"][demo_key]
                actions = demo["actions"][:].astype(np.float32)
                obs = demo["obs"]
                agentview = obs["agentview_rgb"][:]
                wrist = obs["eye_in_hand_rgb"][:]
                ee_pos = obs["ee_pos"][:]
                ee_ori = obs["ee_ori"][:]
                gripper = obs["gripper_states"][:]

                for frame_idx in range(len(actions)):
                    frame = {
                        "observation.state": _compose_state(ee_pos[frame_idx], ee_ori[frame_idx], gripper[frame_idx]),
                        "action": actions[frame_idx],
                        "observation.images.agentview_rgb": _flip_image(agentview[frame_idx]),
                        "observation.images.eye_in_hand_rgb": _flip_image(wrist[frame_idx]),
                    }
                    dataset.add_frame(frame, task=task)
                dataset.save_episode()

    dataset.stop_image_writer()
    dataset.meta.update_video_info()
    _add_action_config(root / "meta")
    print(f"[ok] {suite}: {dataset.meta.total_episodes} episodes -> {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 suites to LingBot-VA LeRobot v2.1 format")
    parser.add_argument("--hdf5-root", type=Path, required=True, help="Root containing libero_spatial/object/goal folders")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for converted LeRobot datasets")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(SUITES),
        choices=SUITES,
        help="Suites to convert",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted suites")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for suite in args.suites:
        hdf5_dir = args.hdf5_root / suite
        if not hdf5_dir.is_dir():
            raise FileNotFoundError(f"Missing suite directory: {hdf5_dir}")
        converted.append(convert_suite(hdf5_dir, args.output_dir, suite, overwrite=args.overwrite))

    print("\nConverted suites:")
    for path in converted:
        print(f"  {path / 'meta' / 'info.json'}")


if __name__ == "__main__":
    main()
