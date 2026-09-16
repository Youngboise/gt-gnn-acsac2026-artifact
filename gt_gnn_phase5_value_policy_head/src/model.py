from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn


class ValuePolicyHead(nn.Module):
    """Decision-aware multi-task value/policy head.

    The regression head predicts R/I/D/U. The best-action head predicts a
    margin-aware good-action probability. The ranking head is trained with
    pairwise/listwise losses and is the preferred score for action selection.
    """

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        hidden_dims = list(hidden_dims)
        act_cls = nn.ReLU if activation.lower() == "relu" else nn.GELU

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()

        self.reg_head = nn.Linear(prev, 4)
        self.best_head = nn.Linear(prev, 1)
        self.rank_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        return {
            "reg_scaled": self.reg_head(h),
            "best_logit": self.best_head(h),
            "rank_score": self.rank_head(h).squeeze(-1),
        }
