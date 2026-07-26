from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT
from robotwin_critic.datasets import ProcessPairDataset
from robotwin_critic.models import RobotWinProcessCritic
from robotwin_critic.train_process_critic import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboTwin process critic.")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_val.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_critic" / "best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neutral-margin", type=float, default=0.05)
    args = parser.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = RobotWinProcessCritic(
        hidden_dim=int(model_args.get("hidden_dim", 512)),
        task_buckets=int(model_args.get("task_buckets", 4096)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    ds = ProcessPairDataset(args.jsonl, task_buckets=int(model_args.get("task_buckets", 4096)))
    metrics = evaluate(model, DataLoader(ds, batch_size=args.batch_size), device, args.neutral_margin)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

