from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT
from robotwin_critic.datasets import ConsistencyPairDataset
from robotwin_critic.models import RobotWinConsistencyFilter


def infer_pos_weight(rows: list[dict]) -> float:
    pos = sum(1 for row in rows if float(row["label"]) > 0.5)
    neg = max(0, len(rows) - pos)
    return float(neg / max(1, pos))


@torch.no_grad()
def _threshold_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict:
    preds = (torch.sigmoid(logits) >= threshold).float()
    neg = labels == 0
    pos = labels == 1
    false_accept_rate = float(((preds == 1) & neg).float().sum().item() / max(1, neg.float().sum().item()))
    false_reject_rate = float(((preds == 0) & pos).float().sum().item() / max(1, pos.float().sum().item()))
    return {
        "accuracy": float((preds == labels).float().mean().item()),
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "negative_accuracy": 1.0 - false_accept_rate,
        "positive_accuracy": 1.0 - false_reject_rate,
        "balanced_accuracy": (2.0 - false_accept_rate - false_reject_rate) / 2.0,
    }


def _operating_point(logits: torch.Tensor, labels: torch.Tensor, target_false_accept: float) -> dict:
    candidates = []
    for threshold in torch.linspace(0.05, 0.95, steps=91).tolist():
        metrics = _threshold_metrics(logits, labels, float(threshold))
        metrics["threshold"] = float(threshold)
        candidates.append(metrics)
    feasible = [m for m in candidates if m["false_accept_rate"] <= target_false_accept]
    if feasible:
        return max(feasible, key=lambda m: (m["positive_accuracy"], m["negative_accuracy"], m["threshold"]))
    return max(candidates, key=lambda m: (m["negative_accuracy"], m["positive_accuracy"], m["threshold"]))


def evaluate(
    model: RobotWinConsistencyFilter,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    target_false_accept: float,
) -> dict:
    model.eval()
    logits_all, labels_all = [], []
    type_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for batch in loader:
        logits = model(
            batch["state"].to(device),
            batch["future"].to(device),
            batch["action"].to(device),
            batch["text_emb"].to(device),
            batch["task_id"].to(device),
        )
        labels = batch["label"].to(device)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
        for pred, label, neg_type in zip(preds.cpu().tolist(), labels.cpu().tolist(), batch["negative_type"]):
            type_stats[neg_type][0] += int(pred == label)
            type_stats[neg_type][1] += 1
    if not logits_all:
        return {"loss": 0.0, "accuracy": 0.0, "false_accept_rate": 0.0, "false_reject_rate": 0.0}
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    base_metrics = _threshold_metrics(logits, labels, threshold)
    operating = _operating_point(logits, labels, target_false_accept)
    out = {
        "loss": float(F.binary_cross_entropy_with_logits(logits, labels).item()),
        **base_metrics,
        "threshold": threshold,
        "operating_threshold": operating["threshold"],
        "operating_false_accept_rate": operating["false_accept_rate"],
        "operating_false_reject_rate": operating["false_reject_rate"],
        "operating_negative_accuracy": operating["negative_accuracy"],
        "operating_positive_accuracy": operating["positive_accuracy"],
        "operating_balanced_accuracy": operating["balanced_accuracy"],
    }
    out["filter_score"] = out["negative_accuracy"] + 0.25 * out["positive_accuracy"]
    for neg_type, (correct, total) in type_stats.items():
        out[f"acc_{neg_type}"] = correct / max(1, total)
    return out


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    train_ds = ConsistencyPairDataset(args.train_jsonl, task_buckets=args.task_buckets)
    val_ds = ConsistencyPairDataset(args.val_jsonl, task_buckets=args.task_buckets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = RobotWinConsistencyFilter(hidden_dim=args.hidden_dim, task_buckets=args.task_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight_value = args.pos_weight if args.pos_weight > 0 else infer_pos_weight(train_ds.rows)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    print(json.dumps({"pos_weight": pos_weight_value}, ensure_ascii=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    history = []
    step = 0
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        metrics = ckpt.get("metrics", {})
        step = int(metrics.get("step", 0))
        best_score = float(metrics.get("filter_score", -1.0))
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_score = float(best_ckpt.get("metrics", {}).get("filter_score", best_score))
        print(json.dumps({"resumed_from": str(args.resume), "step": step, "best_score": best_score}, ensure_ascii=False))
    pbar = tqdm(total=args.max_steps, desc="train consistency filter")
    if step:
        pbar.update(step)
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["state"].to(device),
                batch["future"].to(device),
                batch["action"].to(device),
                batch["text_emb"].to(device),
                batch["task_id"].to(device),
            )
            labels = batch["label"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            pbar.update(1)
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, val_loader, device, args.threshold, args.target_false_accept)
                metrics["step"] = step
                metrics["train_loss"] = float(loss.detach().cpu().item())
                history.append(metrics)
                print(json.dumps(metrics, ensure_ascii=False))
                ckpt = {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                    "model_class": "RobotWinConsistencyFilter",
                }
                torch.save(ckpt, args.output_dir / "last.pt")
                if metrics["filter_score"] > best_score:
                    best_score = metrics["filter_score"]
                    torch.save(ckpt, args.output_dir / "best.pt")
            if step >= args.max_steps:
                break
    pbar.close()
    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")
    return history[-1] if history else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoboTwin action-video consistency filter.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_pairs_train.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_pairs_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "consistency_filter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--task-buckets", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-false-accept", type=float, default=0.1)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
