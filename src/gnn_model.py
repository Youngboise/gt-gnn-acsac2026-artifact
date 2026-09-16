from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """Small pure-PyTorch GraphSAGE-style message passing layer.

    This avoids PyTorch Geometric so the starter code is easier to run on Windows.
    """

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src])
            deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        out = self.linear(torch.cat([h, agg], dim=-1))
        out = F.gelu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class BeliefAdapter(nn.Module):
    """FiLM adapter: global belief vector modulates every node embedding."""

    def __init__(self, belief_dim: int, hidden_dim: int, belief_hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim, belief_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(belief_hidden_dim, hidden_dim * 2),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        film = self.net(belief).squeeze(0)
        gamma, beta = film.chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(gamma).unsqueeze(0)) + beta.unsqueeze(0)
        return self.norm(h)


class MLPHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class BeliefConditionedDualHeadGNN(nn.Module):
    """Belief-conditioned dual-head GNN for Phase 2.

    Outputs:
      - threat_logits: nodes useful for reducing target reach
      - revelation_logits: nodes useful for revealing attacker state
      - candidate_logits: general deployability/usefulness auxiliary score
      - gate_logits: learned threat-vs-revelation mixture gate; sigmoid close to 1 means threat-heavy
      - fused_logits: gate-weighted fusion of threat and revelation logits
    """

    def __init__(
        self,
        node_feature_dim: int,
        belief_dim: int,
        hidden_dim: int = 96,
        belief_hidden_dim: int = 96,
        num_gnn_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pre_adapter = BeliefAdapter(belief_dim, hidden_dim, belief_hidden_dim, dropout)
        self.layers = nn.ModuleList([GraphSAGELayer(hidden_dim, dropout) for _ in range(num_gnn_layers)])
        self.post_adapter = BeliefAdapter(belief_dim, hidden_dim, belief_hidden_dim, dropout)
        self.threat_head = MLPHead(hidden_dim, dropout)
        self.revelation_head = MLPHead(hidden_dim, dropout)
        self.candidate_head = MLPHead(hidden_dim, dropout)
        self.gate_head = MLPHead(hidden_dim, dropout)

    def forward_graph(self, graph: Dict) -> Dict[str, torch.Tensor]:
        x = graph["x"]
        edge_index = graph["edge_index"]
        belief = graph["belief"]
        h = self.input_encoder(x)
        h = self.pre_adapter(h, belief)
        for layer in self.layers:
            h = layer(h, edge_index)
        h = self.post_adapter(h, belief)
        threat_logits = self.threat_head(h)
        revelation_logits = self.revelation_head(h)
        candidate_logits = self.candidate_head(h)
        gate_logits = self.gate_head(h)
        gate = torch.sigmoid(gate_logits)
        fused_logits = gate * threat_logits + (1.0 - gate) * revelation_logits
        return {
            "h": h,
            "threat_logits": threat_logits,
            "revelation_logits": revelation_logits,
            "candidate_logits": candidate_logits,
            "gate_logits": gate_logits,
            "fused_logits": fused_logits,
        }

    def forward(self, graphs: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        return [self.forward_graph(g) for g in graphs]
