from __future__ import annotations

import math

import torch


def process_predictions(delta_u: torch.Tensor, neutral_margin: float) -> torch.Tensor:
    pred = torch.zeros_like(delta_u, dtype=torch.long)
    pred[delta_u > neutral_margin] = 1
    pred[delta_u < -neutral_margin] = -1
    return pred


def macro_f1(pred: torch.Tensor, target: torch.Tensor, labels: tuple[int, ...] = (-1, 0, 1)) -> dict:
    out = {}
    f1s = []
    for label in labels:
        p = pred == label
        t = target == label
        tp = (p & t).sum().item()
        fp = (p & ~t).sum().item()
        fn = (~p & t).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out[f"f1_{label}"] = f1
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s)
    return out


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    xr = torch.argsort(torch.argsort(x.float())).float()
    yr = torch.argsort(torch.argsort(y.float())).float()
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = torch.sqrt((xr.square().sum() * yr.square().sum()).clamp_min(1e-12))
    val = (xr * yr).sum() / denom
    if torch.isnan(val):
        return 0.0
    return float(val.item())

