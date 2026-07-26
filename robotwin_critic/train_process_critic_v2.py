from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, read_jsonl, stable_int, state_feature, text_feature
from robotwin_critic.datasets import _RetryBadRowsMixin
from robotwin_critic.metrics import macro_f1, process_predictions, spearman
from robotwin_critic.models import RobotWinProcessCritic
from robotwin_critic.train_process_critic import process_loss


class ProgressProcessPairDataset(_RetryBadRowsMixin, Dataset):
    def __init__(self, jsonl_path: str | Path, task_buckets: int = 4096, drop_neutral: bool = False):
        rows = read_jsonl(Path(jsonl_path))
        if drop_neutral:
            rows = [row for row in rows if int(row["label"]) != 0]
        self.rows = rows
        self.task_buckets = task_buckets

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self._load_with_retry(idx, self._build_item)

    def _build_item(self, row: dict) -> dict:
        latents = row["latents"]
        frame_i = int(row["frame_i"])
        frame_j = int(row["frame_j"])
        length = max(1, int(row.get("length", 1)) - 1)
        return {
            "state_i": state_feature(latents, frame_i),
            "state_j": state_feature(latents, frame_j),
            "state_final": state_feature(latents, int(row["final_frame"])),
            "text_emb": text_feature(latents),
            "task_id": torch.tensor(stable_int(row["task_name"], self.task_buckets), dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "frame_i": torch.tensor(frame_i, dtype=torch.float32),
            "frame_j": torch.tensor(frame_j, dtype=torch.float32),
            "progress_delta": torch.tensor((frame_j - frame_i) / length, dtype=torch.float32),
        }


def _load_batch(model: RobotWinProcessCritic, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_i = batch["state_i"].to(device)
    state_j = batch["state_j"].to(device)
    final_state = batch["state_final"].to(device)
    text_emb = batch["text_emb"].to(device)
    task_id = batch["task_id"].to(device)
    u_i = model(state_i, final_state, text_emb, task_id)
    u_j = model(state_j, final_state, text_emb, task_id)
    return u_i, u_j, u_j - u_i


def combined_loss(
    delta_u: torch.Tensor,
    label: torch.Tensor,
    target_delta: torch.Tensor,
    score_reg: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    pair_loss = process_loss(delta_u, label, score_reg, args.reg_weight)
    reg_target = target_delta * args.target_scale
    regression = F.smooth_l1_loss(delta_u, reg_target)
    return pair_loss + args.progress_weight * regression


@torch.no_grad()
def evaluate(model: RobotWinProcessCritic, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict:
    model.eval()
    losses, preds, labels, deltas, time_deltas, progress_deltas = [], [], [], [], [], []
    for batch in loader:
        label = batch["label"].to(device)
        progress_delta = batch["progress_delta"].to(device)
        u_i, u_j, delta_u = _load_batch(model, batch, device)
        loss = combined_loss(delta_u, label, progress_delta, u_i.square() + u_j.square(), args)
        pred = process_predictions(delta_u, args.neutral_margin)
        losses.append(loss.detach().cpu())
        preds.append(pred.cpu())
        labels.append(label.cpu())
        deltas.append(delta_u.cpu())
        time_deltas.append((batch["frame_j"] - batch["frame_i"]).cpu())
        progress_deltas.append(progress_delta.cpu())

    if not losses:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "spearman": 0.0}

    pred_all = torch.cat(preds)
    label_all = torch.cat(labels)
    delta_all = torch.cat(deltas)
    time_all = torch.cat(time_deltas)
    progress_all = torch.cat(progress_deltas)
    non_neutral = label_all != 0
    out = {
        "loss": float(torch.stack(losses).mean().item()),
        "accuracy": float((pred_all == label_all).float().mean().item()),
        "spearman": spearman(delta_all, time_all),
        "progress_spearman": spearman(delta_all, progress_all),
    }
    if non_neutral.any():
        out["non_neutral_accuracy"] = float((pred_all[non_neutral] == label_all[non_neutral]).float().mean().item())
        out["sign_accuracy"] = float((torch.sign(delta_all[non_neutral]) == label_all[non_neutral].float()).float().mean().item())
    else:
        out["non_neutral_accuracy"] = 0.0
        out["sign_accuracy"] = 0.0
    out.update(macro_f1(pred_all, label_all))
    return out


def metric_value(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    if value is None:
        raise KeyError(f"metric {name!r} missing from evaluation metrics")
    return float(value)


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    train_ds = ProgressProcessPairDataset(args.train_jsonl, task_buckets=args.task_buckets, drop_neutral=args.drop_neutral)
    val_ds = ProgressProcessPairDataset(args.val_jsonl, task_buckets=args.task_buckets, drop_neutral=args.drop_neutral)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = RobotWinProcessCritic(hidden_dim=args.hidden_dim, task_buckets=args.task_buckets).to(device)
    if args.init_checkpoint and args.init_checkpoint.exists():
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(json.dumps({"initialized_from": str(args.init_checkpoint)}, ensure_ascii=False))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    history = []
    step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        metrics = ckpt.get("metrics", {})
        step = int(metrics.get("step", 0))
        history_path = args.output_dir / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except json.JSONDecodeError:
                history = []
        best_path = args.output_dir / "best.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_score = metric_value(best_ckpt.get("metrics", {}), args.best_metric)
        print(json.dumps({"resumed_from": str(args.resume), "step": step, "best_score": best_score}, ensure_ascii=False))

    pbar = tqdm(total=args.max_steps, initial=step, desc="train process critic v2")
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            opt.zero_grad(set_to_none=True)
            label = batch["label"].to(device)
            progress_delta = batch["progress_delta"].to(device)
            u_i, u_j, delta_u = _load_batch(model, batch, device)
            loss = combined_loss(delta_u, label, progress_delta, u_i.square() + u_j.square(), args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            step += 1
            pbar.update(1)
            if step % args.eval_interval == 0 or step == args.max_steps:
                metrics = evaluate(model, val_loader, device, args)
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
                score = metric_value(metrics, args.best_metric)
                if score > best_score:
                    best_score = score
                    torch.save(ckpt, args.output_dir / "best.pt")
            if step >= args.max_steps:
                break
    pbar.close()
    with (args.output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")
    return history[-1] if history else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoboTwin process critic v2 with progress-delta auxiliary loss.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_critic_v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--progress-weight", type=float, default=0.5)
    parser.add_argument("--target-scale", type=float, default=2.0)
    parser.add_argument("--neutral-margin", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--task-buckets", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--best-metric", default="spearman")
    parser.add_argument("--drop-neutral", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
