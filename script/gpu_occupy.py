#!/usr/bin/env python3
"""Occupy GPUs by allocating VRAM and running light compute."""

import argparse
import os
import signal
import sys
import time

import torch


running = True


def handle_signal(signum, _frame):
    global running
    print(f"[gpu_occupy] received signal {signum}, exiting...", flush=True)
    running = False


def occupy_gpu(gpu_id: int, memory_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    torch.cuda.set_device(gpu_id)
    props = torch.cuda.get_device_properties(gpu_id)
    total = props.total_memory
    # Leave headroom for driver / fragmentation.
    target = int(total * memory_fraction)
    elem_size = 4  # float32
    numel = target // elem_size
    buf = torch.empty(numel, dtype=torch.float32, device=f"cuda:{gpu_id}")
    work = torch.randn(4096, 4096, device=f"cuda:{gpu_id}")
    torch.cuda.synchronize(gpu_id)
    used_mb = torch.cuda.memory_allocated(gpu_id) / (1024**2)
    print(
        f"[gpu_occupy] GPU {gpu_id} ({props.name}): allocated ~{used_mb:.0f} MiB",
        flush=True,
    )
    return buf, work


def main() -> int:
    parser = argparse.ArgumentParser(description="Occupy idle GPUs")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.90,
        help="Fraction of VRAM to allocate per GPU (default: 0.90)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help='GPU ids comma-separated, or "all" (default: all)',
    )
    parser.add_argument(
        "--compute-interval",
        type=float,
        default=30.0,
        help="Seconds between light matmul bursts (default: 30)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[gpu_occupy] CUDA not available", file=sys.stderr)
        return 1

    if args.gpus == "all":
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]

    if not gpu_ids:
        print("[gpu_occupy] no GPUs selected", file=sys.stderr)
        return 1

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    buffers: list[torch.Tensor] = []
    work_tensors: list[torch.Tensor] = []
    for gid in gpu_ids:
        buf, work = occupy_gpu(gid, args.memory_fraction)
        buffers.append(buf)
        work_tensors.append(work)

    pid = os.getpid()
    print(f"[gpu_occupy] holding {len(gpu_ids)} GPU(s), pid={pid}", flush=True)

    while running:
        for i, gid in enumerate(gpu_ids):
            if not running:
                break
            torch.cuda.set_device(gid)
            _ = work_tensors[i] @ work_tensors[i]
            torch.cuda.synchronize(gid)
        time.sleep(args.compute_interval)

    print("[gpu_occupy] releasing GPU memory...", flush=True)
    del buffers, work_tensors
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
