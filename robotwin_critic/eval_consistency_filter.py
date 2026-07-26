from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT
from robotwin_critic.datasets import ConsistencyPairDataset
from robotwin_critic.models import RobotWinConsistencyFilter
from robotwin_critic.train_consistency_filter import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RoboTwin consistency filter.")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_pairs_val.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_filter" / "best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-false-accept", type=float, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = RobotWinConsistencyFilter(
        hidden_dim=int(model_args.get("hidden_dim", 512)),
        task_buckets=int(model_args.get("task_buckets", 4096)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    ds = ConsistencyPairDataset(args.jsonl, task_buckets=int(model_args.get("task_buckets", 4096)))
    threshold = float(args.threshold if args.threshold is not None else model_args.get("threshold", 0.5))
    target_false_accept = float(
        args.target_false_accept
        if args.target_false_accept is not None
        else model_args.get("target_false_accept", 0.1)
    )
    metrics = evaluate(
        model,
        DataLoader(ds, batch_size=args.batch_size),
        device,
        threshold,
        target_false_accept,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

