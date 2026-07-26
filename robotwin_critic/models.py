from __future__ import annotations

import torch
from torch import nn


class RobotWinProcessCritic(nn.Module):
    def __init__(
        self,
        state_dim: int = 288,
        text_dim: int = 4096,
        hidden_dim: int = 512,
        task_buckets: int = 4096,
        task_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_emb = nn.Embedding(task_buckets, task_dim)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
        )
        state_in = state_dim * 4
        self.state_proj = nn.Sequential(
            nn.LayerNorm(state_in),
            nn.Linear(state_in, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + task_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, final_state: torch.Tensor, text_emb: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        state_in = torch.cat([state, final_state, final_state - state, state * final_state], dim=-1)
        x_state = self.state_proj(state_in)
        x_text = self.text_proj(text_emb)
        x_task = self.task_emb(task_id)
        return self.head(torch.cat([x_state, x_text, x_task], dim=-1)).squeeze(-1)


class RobotWinConsistencyFilter(nn.Module):
    def __init__(
        self,
        state_dim: int = 288,
        action_dim: int = 48,
        text_dim: int = 4096,
        hidden_dim: int = 512,
        task_buckets: int = 4096,
        task_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_emb = nn.Embedding(task_buckets, task_dim)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim), nn.GELU())
        pair_dim = state_dim * 4 + action_dim
        self.pair_proj = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + task_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        future: torch.Tensor,
        action: torch.Tensor,
        text_emb: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        pair = torch.cat([state, future, future - state, state * future, action], dim=-1)
        x_pair = self.pair_proj(pair)
        x_text = self.text_proj(text_emb)
        x_task = self.task_emb(task_id)
        return self.head(torch.cat([x_pair, x_text, x_task], dim=-1)).squeeze(-1)

