from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT
from robotwin_critic.datasets import ProcessPairDataset
from robotwin_critic.metrics import macro_f1, process_predictions, spearman
from robotwin_critic.models import RobotWinProcessCritic


def process_loss(delta_u: torch.Tensor, label: torch.Tensor, score_reg: torch.Tensor, reg_weight: float) -> torch.Tensor:
    pos = label == 1
    neg = label == -1
    neu = label == 0
    loss = torch.zeros_like(delta_u)
    loss[pos] = F.softplus(-delta_u[pos])
    loss[neg] = F.softplus(delta_u[neg])
    loss[neu] = delta_u[neu].square()
    return loss.mean() + reg_weight * score_reg.mean()


@torch.no_grad()
def evaluate(model: RobotWinProcessCritic, loader: DataLoader, device: torch.device, neutral_margin: float) -> dict:
    model.eval()
    losses, preds, labels, deltas, time_deltas = [], [], [], [], []
    for batch in loader:
        state_i = batch["state_i"].to(device)
        state_j = batch["state_j"].to(device)
        final_state = batch["state_final"].to(device)
        text_emb = batch["text_emb"].to(device)
        task_id = batch["task_id"].to(device)
        label = batch["label"].to(device)
        u_i = model(state_i, final_state, text_emb, task_id)
        u_j = model(state_j, final_state, text_emb, task_id)
        delta_u = u_j - u_i
        loss = process_loss(delta_u, label, u_i.square() + u_j.square(), 0.0)
        pred = process_predictions(delta_u, neutral_margin)
        losses.append(loss.detach().cpu())
        preds.append(pred.cpu())
        labels.append(label.cpu())
        deltas.append(delta_u.cpu())
        time_deltas.append((batch["frame_j"] - batch["frame_i"]).cpu())
    if not losses:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "spearman": 0.0}
    pred_all = torch.cat(preds)
    label_all = torch.cat(labels)
    delta_all = torch.cat(deltas)
    time_all = torch.cat(time_deltas)
    out = {
        "loss": float(torch.stack(losses).mean().item()),
        "accuracy": float((pred_all == label_all).float().mean().item()),
        "spearman": spearman(delta_all, time_all),
    }
    out.update(macro_f1(pred_all, label_all))
    return out


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    train_ds = ProcessPairDataset(args.train_jsonl, task_buckets=args.task_buckets)
    val_ds = ProcessPairDataset(args.val_jsonl, task_buckets=args.task_buckets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = RobotWinProcessCritic(hidden_dim=args.hidden_dim, task_buckets=args.task_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    history = []
    step = 0
    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        metrics = ckpt.get("metrics", {})
        step = int(metrics.get("step", 0))
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_acc = float(best_ckpt.get("metrics", {}).get("accuracy", -1.0))
        history_path = args.output_dir / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except json.JSONDecodeError:
                history = []
        print(json.dumps({"resumed_from": str(args.resume), "step": step, "best_acc": best_acc}, ensure_ascii=False))
    pbar = tqdm(total=args.max_steps, initial=step, desc="train process critic")
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            opt.zero_grad(set_to_none=True)
            state_i = batch["state_i"].to(device)
            state_j = batch["state_j"].to(device)
            final_state = batch["state_final"].to(device)
            text_emb = batch["text_emb"].to(device)
            task_id = batch["task_id"].to(device)
            label = batch["label"].to(device)
            u_i = model(state_i, final_state, text_emb, task_id)
            u_j = model(state_j, final_state, text_emb, task_id)
            delta_u = u_j - u_i
            loss = process_loss(delta_u, label, u_i.square() + u_j.square(), args.reg_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            step += 1
            pbar.update(1)
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, val_loader, device, args.neutral_margin)
                metrics["step"] = step
                metrics["train_loss"] = float(loss.detach().cpu().item())
                history.append(metrics)
                print(json.dumps(metrics, ensure_ascii=False))
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                    "model_class": "RobotWinProcessCritic",
                }
                torch.save(ckpt, args.output_dir / "last.pt")
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    torch.save(ckpt, args.output_dir / "best.pt")
            if step >= args.max_steps:
                break
    pbar.close()
    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")
    return history[-1] if history else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoboTwin process value critic.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_critic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--neutral-margin", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--task-buckets", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

