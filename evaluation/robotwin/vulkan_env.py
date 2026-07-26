"""Configure Vulkan before SAPIEN is imported (multi-GPU eval)."""
from __future__ import annotations

import fcntl
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


def _physical_gpu_id() -> str:
    return os.environ.get(
        "ROBOTWIN_VULKAN_GPU",
        os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip(),
    )


def _gpu_uuid(gpu_id: str) -> str:
    env_key = f"ROBOTWIN_GPU_{gpu_id}_UUID"
    cached = os.environ.get(env_key)
    if cached:
        return cached
    try:
        uuid = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        uuid = gpu_id
    os.environ[env_key] = uuid
    return uuid


def configure_robotwin_vulkan() -> None:
    """Pin Vulkan/CUDA sim to the chosen physical GPU (see sim_gpu_for_server_gpu)."""
    gpu_id = _physical_gpu_id()
    os.environ["ROBOTWIN_VULKAN_GPU"] = gpu_id
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", gpu_id)

    if not os.environ.get("VK_ICD_FILENAMES"):
        for candidate in (
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            str(Path(__file__).resolve().parents[2] / "script" / "nvidia_icd.json"),
        ):
            if os.path.isfile(candidate):
                os.environ["VK_ICD_FILENAMES"] = candidate
                break

    os.environ.setdefault("VK_LOADER_LAYERS_ENABLE", "VK_LAYER_MESA_device_select")
    # UUID is more reliable than numeric index when multiple processes enumerate GPUs.
    os.environ["MESA_VK_DEVICE_SELECT"] = _gpu_uuid(gpu_id)


@contextmanager
def vulkan_gpu_init_lock():
    """Serialize SAPIEN renderer creation per sim GPU (brief, init only)."""
    gpu_id = _physical_gpu_id()
    lock_path = Path(f"/tmp/lingbot_robotwin_vulkan_gpu{gpu_id}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


@contextmanager
def vulkan_gpu_sim_slot():
    """Limit concurrent SAPIEN sims per GPU (default 3, matches CLIENTS_PER_GPU)."""
    import time

    gpu_id = _physical_gpu_id()
    max_slots = int(os.environ.get("ROBOTWIN_VULKAN_SIM_SLOTS", "3"))
    acquired = None
    try:
        while acquired is None:
            for slot in range(max_slots):
                path = Path(f"/tmp/lingbot_robotwin_vulkan_gpu{gpu_id}_sim{slot}.lock")
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(path, "w", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = handle
                    break
                except BlockingIOError:
                    handle.close()
            if acquired is None:
                time.sleep(0.5)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(acquired.fileno(), fcntl.LOCK_UN)
            acquired.close()
