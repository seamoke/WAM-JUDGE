"""Minimal NCCL smoke test for the two-node online-RFT topology."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce={value.item()} expected={expected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(local_rank),
        "all_reduce": value.item(),
    }
    (args.output_dir / f"rank_{rank}.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"MULTINODE_SMOKE_OK {json.dumps(payload, sort_keys=True)}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
