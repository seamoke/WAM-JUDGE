from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, read_jsonl, stable_int
from robotwin_critic.metrics import macro_f1, process_predictions, spearman
from robotwin_critic.models import RobotWinProcessCritic
from robotwin_critic.build_spatial_feature_cache import spatial_state_feature
from robotwin_critic.train_process_critic import process_loss


@lru_cache(maxsize=512)
def _load_feature_cache(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


class SpatialCachedProcessPairDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, task_buckets: int = 4096, drop_neutral: bool = False):
        rows = read_jsonl(Path(jsonl_path))
        if drop_neutral:
            rows = [row for row in rows if int(row["label"]) != 0]
        self.rows = rows
        self.task_buckets = task_buckets

    def __len__(self) -> int:
        return len(self.rows)

    def state_dim(self) -> int:
        if not self.rows:
            raise ValueError("empty spatial cached dataset")
        cache = _load_feature_cache(self.rows[0]["feature_cache_path"])
        return int(cache.get("state_dim") or next(iter(cache["states"].values())).numel())

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        cache = _load_feature_cache(row["feature_cache_path"])
        states = cache["states"]
        frame_i = int(row["frame_i"])
        frame_j = int(row["frame_j"])
        final_frame = int(row["final_frame"])
        length = max(1, int(row.get("length", 1)) - 1)
        grid = int(cache.get("grid", 4))

        def get_state(frame: int) -> torch.Tensor:
            if frame not in states:
                states[frame] = spatial_state_feature(row["latents"], frame, grid).to(cache["text_emb"].dtype)
            return states[frame].float()

        return {
            "state_i": get_state(frame_i),
            "state_j": get_state(frame_j),
            "state_final": get_state(final_frame),
            "text_emb": cache["text_emb"].float(),
            "task_id": torch.tensor(stable_int(row["task_name"], self.task_buckets), dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "frame_i": torch.tensor(frame_i, dtype=torch.float32),
            "frame_j": torch.tensor(frame_j, dtype=torch.float32),
            "progress_delta": torch.tensor((frame_j - frame_i) / length, dtype=torch.float32),
        }


def _forward_delta(model: RobotWinProcessCritic, batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_i = batch["state_i"].to(device, non_blocking=True)
    state_j = batch["state_j"].to(device, non_blocking=True)
    final_state = batch["state_final"].to(device, non_blocking=True)
    text_emb = batch["text_emb"].to(device, non_blocking=True)
    task_id = batch["task_id"].to(device, non_blocking=True)
    u_i = model(state_i, final_state, text_emb, task_id)
    u_j = model(state_j, final_state, text_emb, task_id)
    return u_i, u_j, u_j - u_i


def combined_loss(delta_u: torch.Tensor, label: torch.Tensor, target_delta: torch.Tensor, score_reg: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    pair_loss = process_loss(delta_u, label, score_reg, args.reg_weight)
    regression = F.smooth_l1_loss(delta_u, target_delta * args.target_scale)
    return pair_loss + args.progress_weight * regression


@torch.no_grad()
def evaluate(model: RobotWinProcessCritic, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict:
    model.eval()
    losses, preds, labels, deltas, time_deltas, progress_deltas = [], [], [], [], [], []
    for batch in loader:
        label = batch["label"].to(device, non_blocking=True)
        progress_delta = batch["progress_delta"].to(device, non_blocking=True)
        u_i, u_j, delta_u = _forward_delta(model, batch, device)
        loss = combined_loss(delta_u, label, progress_delta, u_i.square() + u_j.square(), args)
        losses.append(loss.detach().cpu())
        preds.append(process_predictions(delta_u, args.neutral_margin).cpu())
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


def _make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    train_ds = SpatialCachedProcessPairDataset(args.train_jsonl, args.task_buckets, args.drop_neutral)
    val_ds = SpatialCachedProcessPairDataset(args.val_jsonl, args.task_buckets, args.drop_neutral)
    train_loader = _make_loader(train_ds, args, shuffle=True)
    val_loader = _make_loader(val_ds, args, shuffle=False)

    state_dim = args.state_dim or train_ds.state_dim()
    model = RobotWinProcessCritic(state_dim=state_dim, hidden_dim=args.hidden_dim, task_buckets=args.task_buckets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.json"
    history: list[dict] = []
    best_score = -1e9
    step = 0

    print(json.dumps({"state_dim": state_dim, "train_rows": len(train_ds), "val_rows": len(val_ds)}, ensure_ascii=False), flush=True)
    pbar = tqdm(total=args.max_steps, desc="train spatial process critic v2")
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            opt.zero_grad(set_to_none=True)
            label = batch["label"].to(device, non_blocking=True)
            progress_delta = batch["progress_delta"].to(device, non_blocking=True)
            u_i, u_j, delta_u = _forward_delta(model, batch, device)
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
                history_path.write_text(json.dumps(history, indent=2) + "\n")
                print(json.dumps(metrics, ensure_ascii=False), flush=True)
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "args": vars(args),
                    "state_dim": state_dim,
                    "metrics": metrics,
                    "model_class": "RobotWinProcessCritic",
                    "feature_type": "spatial_cached",
                }
                torch.save(ckpt, args.output_dir / "last.pt")
                score = float(metrics[args.best_metric])
                if score > best_score:
                    best_score = score
                    torch.save(ckpt, args.output_dir / "best.pt")
            if step >= args.max_steps:
                break
    pbar.close()
    return history[-1] if history else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoboTwin process critic v2 from spatial cached features.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_train_spatial_cached.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_pairs_val_spatial_cached.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_critic_spatial_v2_rank_only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--target-scale", type=float, default=2.0)
    parser.add_argument("--neutral-margin", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--task-buckets", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--best-metric", default="spearman")
    parser.add_argument("--drop-neutral", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
