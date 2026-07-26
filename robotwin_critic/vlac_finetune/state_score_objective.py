"""Pairwise objective for a shared VLAC-backed state-score critic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StateScoreLossOutput:
    loss: torch.Tensor
    ranking_loss: torch.Tensor
    score_regularization: torch.Tensor
    delta: torch.Tensor


def state_score_pair_loss(
    score_i: torch.Tensor,
    score_j: torch.Tensor,
    labels: torch.Tensor,
    *,
    score_reg_weight: float = 1e-4,
) -> StateScoreLossOutput:
    """Train two shared-model forwards through their score difference.

    Labels are -1, 0, or +1. Positive means state_j should score above
    state_i, negative means the reverse, and neutral penalizes score drift.
    """

    score_i = score_i.float().reshape(-1)
    score_j = score_j.float().reshape(-1)
    labels = labels.to(device=score_i.device).reshape(-1)
    if score_i.shape != score_j.shape or score_i.shape != labels.shape:
        raise ValueError(
            "score_i, score_j, and labels must contain the same number of samples"
        )
    valid = (labels == -1) | (labels == 0) | (labels == 1)
    if not bool(torch.all(valid)):
        raise ValueError("labels must only contain -1, 0, or +1")

    delta = score_j - score_i
    positive = labels == 1
    negative = labels == -1
    neutral = labels == 0
    per_sample = torch.zeros_like(delta)
    per_sample[positive] = F.softplus(-delta[positive])
    per_sample[negative] = F.softplus(delta[negative])
    per_sample[neutral] = delta[neutral].square()
    ranking_loss = per_sample.mean()
    score_regularization = (score_i.square() + score_j.square()).mean()
    loss = ranking_loss + float(score_reg_weight) * score_regularization
    return StateScoreLossOutput(
        loss=loss,
        ranking_loss=ranking_loss,
        score_regularization=score_regularization,
        delta=delta,
    )


def classify_delta(delta: torch.Tensor, neutral_margin: float = 0.05) -> torch.Tensor:
    """Map score differences to {-1, 0, +1} for validation metrics."""

    delta = delta.float()
    margin = float(neutral_margin)
    if margin < 0:
        raise ValueError("neutral_margin must be non-negative")
    predictions = torch.zeros_like(delta, dtype=torch.long)
    predictions[delta > margin] = 1
    predictions[delta < -margin] = -1
    return predictions
