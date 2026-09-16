from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .generate_gnn_dataset import FACTOR_KEYS, package_root_from_config
from .gnn_dataset import GraphSnapshotDataset, collate_graph_snapshots
from .gnn_model import BeliefConditionedDualHeadGNN
from .utils import ensure_parent, load_yaml, safe_auc_ap, set_seed, write_json


def move_graph_to_device(graph: Dict, device: torch.device) -> Dict:
    out = dict(graph)
    for key in ["x", "edge_index", "belief", "threat_y", "revelation_y", "candidate_y", "gate_y", "candidate_mask", "target_mask"]:
        if key in out and torch.is_tensor(out[key]):
            out[key] = out[key].to(device)
    return out


def build_model(cfg: Dict, metadata: Dict) -> BeliefConditionedDualHeadGNN:
    return BeliefConditionedDualHeadGNN(
        node_feature_dim=len(metadata["node_features"]),
        belief_dim=int(metadata["belief_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        belief_hidden_dim=int(cfg["model"]["belief_hidden_dim"]),
        num_gnn_layers=int(cfg["model"]["num_gnn_layers"]),
        dropout=float(cfg["model"]["dropout"]),
    )


def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.sum() < 1:
        mask = torch.ones_like(mask)
    loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.sum() < 1:
        mask = torch.ones_like(mask)
    pred = torch.sigmoid(logits)
    loss = (pred - y.float()).pow(2)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def compute_batch_loss(model, graphs: List[Dict], cfg: Dict) -> Dict[str, torch.Tensor]:
    outputs = model(graphs)
    losses = []
    parts = {"threat": [], "revelation": [], "candidate": [], "gate": []}
    for g, out in zip(graphs, outputs):
        mask = g["candidate_mask"]
        threat = masked_bce(out["threat_logits"], g["threat_y"], mask)
        revelation = masked_bce(out["revelation_logits"], g["revelation_y"], mask)
        candidate = masked_bce(out["candidate_logits"], g["candidate_y"], mask)
        gate = masked_mse_from_logits(out["gate_logits"], g["gate_y"], mask)
        total = (
            float(cfg["training"]["lambda_threat"]) * threat
            + float(cfg["training"]["lambda_revelation"]) * revelation
            + float(cfg["training"]["lambda_candidate"]) * candidate
            + float(cfg["training"]["lambda_gate"]) * gate
        )
        losses.append(total)
        parts["threat"].append(threat.detach())
        parts["revelation"].append(revelation.detach())
        parts["candidate"].append(candidate.detach())
        parts["gate"].append(gate.detach())
    return {
        "loss": torch.stack(losses).mean(),
        "threat_loss": torch.stack(parts["threat"]).mean(),
        "revelation_loss": torch.stack(parts["revelation"]).mean(),
        "candidate_loss": torch.stack(parts["candidate"]).mean(),
        "gate_loss": torch.stack(parts["gate"]).mean(),
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device, top_k: int = 3) -> Dict[str, float]:
    model.eval()
    y_threat, s_threat = [], []
    y_rev, s_rev = [], []
    y_cand, s_cand = [], []
    gate_true, gate_pred = [], []
    topk_hits = []
    fusion_topk_hits = []

    for batch in loader:
        graphs = [move_graph_to_device(g, device) for g in batch["graphs"]]
        outs = model(graphs)
        for g, out in zip(graphs, outs):
            mask = g["candidate_mask"].detach().cpu().numpy() > 0.5
            if mask.sum() == 0:
                continue
            threat_scores = torch.sigmoid(out["threat_logits"]).detach().cpu().numpy()
            rev_scores = torch.sigmoid(out["revelation_logits"]).detach().cpu().numpy()
            cand_scores = torch.sigmoid(out["candidate_logits"]).detach().cpu().numpy()
            fused_scores = torch.sigmoid(out["fused_logits"]).detach().cpu().numpy()
            gate_scores = torch.sigmoid(out["gate_logits"]).detach().cpu().numpy()

            yt = g["threat_y"].detach().cpu().numpy()
            yr = g["revelation_y"].detach().cpu().numpy()
            yc = g["candidate_y"].detach().cpu().numpy()
            gy = g["gate_y"].detach().cpu().numpy()

            y_threat.extend(yt[mask].tolist())
            s_threat.extend(threat_scores[mask].tolist())
            y_rev.extend(yr[mask].tolist())
            s_rev.extend(rev_scores[mask].tolist())
            y_cand.extend(yc[mask].tolist())
            s_cand.extend(cand_scores[mask].tolist())
            gate_true.extend(gy[mask].tolist())
            gate_pred.extend(gate_scores[mask].tolist())

            cand_indices = np.where(mask)[0]
            k = min(top_k, len(cand_indices))
            threat_ranked = cand_indices[np.argsort(-threat_scores[cand_indices])[:k]]
            fused_ranked = cand_indices[np.argsort(-fused_scores[cand_indices])[:k]]
            topk_hits.append(float(yt[threat_ranked].max() > 0.5))
            fusion_topk_hits.append(float(yc[fused_ranked].max() > 0.5))

    threat_auc, threat_ap = safe_auc_ap(y_threat, s_threat)
    rev_auc, rev_ap = safe_auc_ap(y_rev, s_rev)
    cand_auc, cand_ap = safe_auc_ap(y_cand, s_cand)
    return {
        "threat_auc": threat_auc,
        "threat_ap": threat_ap,
        "revelation_auc": rev_auc,
        "revelation_ap": rev_ap,
        "candidate_auc": cand_auc,
        "candidate_ap": cand_ap,
        "gate_mse": float(np.mean((np.asarray(gate_pred) - np.asarray(gate_true)) ** 2)) if gate_true else float("nan"),
        "threat_topk_hit": float(np.mean(topk_hits)) if topk_hits else float("nan"),
        "fused_topk_hit": float(np.mean(fusion_topk_hits)) if fusion_topk_hits else float("nan"),
    }


def train(config_path: str) -> Dict[str, float]:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pt_path = root / cfg["paths"]["gnn_dataset_pt"]
    train_ds = GraphSnapshotDataset(str(pt_path), split="train")
    val_ds = GraphSnapshotDataset(str(pt_path), split="val")
    metadata = train_ds.metadata
    model = build_model(cfg, metadata).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_graph_snapshots,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_graph_snapshots,
    )

    best_score = -float("inf")
    best_metrics: Dict[str, float] = {}
    history = []
    patience = int(cfg["training"]["early_stop_patience"])
    bad_epochs = 0
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            graphs = [move_graph_to_device(g, device) for g in batch["graphs"]]
            loss_dict = compute_batch_loss(model, graphs, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
            optimizer.step()
            epoch_losses.append(float(loss_dict["loss"].detach().cpu()))

        val_metrics = evaluate(model, val_loader, device, int(cfg["ranking"]["top_k"]))
        score = np.nanmean([
            val_metrics.get("threat_auc", float("nan")),
            val_metrics.get("revelation_auc", float("nan")),
            val_metrics.get("candidate_auc", float("nan")),
            val_metrics.get("fused_topk_hit", float("nan")),
        ])
        record = {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), **val_metrics, "selection_score": float(score)}
        history.append(record)
        print(
            f"Epoch {epoch:03d} | loss={record['train_loss']:.4f} | "
            f"threat_auc={val_metrics['threat_auc']:.4f} | rev_auc={val_metrics['revelation_auc']:.4f} | "
            f"cand_auc={val_metrics['candidate_auc']:.4f} | fused@k={val_metrics['fused_topk_hit']:.4f}"
        )

        if score > best_score:
            best_score = float(score)
            best_metrics = val_metrics
            bad_epochs = 0
            ckpt_path = root / cfg["paths"]["checkpoint"]
            ensure_parent(str(ckpt_path))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "metadata": metadata,
                    "best_val_metrics": best_metrics,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    write_json(history, str(root / cfg["paths"]["train_history"]))
    print(f"Best validation score: {best_score:.4f}")
    print(f"Saved checkpoint to {root / cfg['paths']['checkpoint']}")
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gnn_config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
