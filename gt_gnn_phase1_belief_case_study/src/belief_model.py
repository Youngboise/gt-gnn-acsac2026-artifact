from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedBeliefGRU(nn.Module):
    """Action-conditioned temporal belief estimator.

    Input:
      x: [batch, window, feature_dim]

    Output logits:
      goal, skill, stealth, decoy_awareness
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_goal: int,
        num_skill: int,
        num_stealth: int,
        num_decoy_awareness: int,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.goal_head = nn.Linear(hidden_dim, num_goal)
        self.skill_head = nn.Linear(hidden_dim, num_skill)
        self.stealth_head = nn.Linear(hidden_dim, num_stealth)
        self.decoy_awareness_head = nn.Linear(hidden_dim, num_decoy_awareness)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_norm(x)
        encoded, hidden = self.encoder(x)
        h = encoded[:, -1, :]
        h = self.shared(self.dropout(h))
        return {
            "goal": self.goal_head(h),
            "skill": self.skill_head(h),
            "stealth": self.stealth_head(h),
            "decoy_awareness": self.decoy_awareness_head(h),
        }

    @torch.no_grad()
    def predict_belief(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.forward(x)
        probs = {k: F.softmax(v, dim=-1) for k, v in logits.items()}
        entropy = {k: -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(dim=-1) for k, p in probs.items()}
        return {"probs": probs, "entropy": entropy}
