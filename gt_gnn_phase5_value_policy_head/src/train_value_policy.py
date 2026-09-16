from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .model import ValuePolicyHead
from .utils import get_device, load_config, repo_root, resolve_path, set_seed, write_json
from .value_dataset import ValuePolicyDataset


def _load_prepared_dataset(root: Path, cfg: Dict[str, object]) -> Dict[str, object]:
    dataset_path = resolve_path(root, cfg["paths"]["processed_dataset_pt"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}\n"
            "Run: python -m src.prepare_value_dataset --config config/value_config.yaml"
        )
    return torch.load(dataset_path, map_location="cpu", weights_only=False)


def _pairwise_rank_loss(scores: torch.Tensor, utilities: torch.Tensor, group_id: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    rank_cfg = cfg.get("ranking", {})
    if not bool(rank_cfg.get("enabled", True)):
        return scores.new_tensor(0.0)
    margin = float(rank_cfg.get("pairwise_margin", 0.015))
    max_pairs = int(rank_cfg.get("max_pairs_per_group", 64))
    losses = []
    utilities = utilities.reshape(-1)
    for gid in torch.unique(group_id):
        idx = torch.where(group_id == gid)[0]
        if idx.numel() < 2:
            continue
        u = utilities[idx]
        s = scores[idx]
        diff_u = u[:, None] - u[None, :]
        pos = torch.where(diff_u > margin)
        if pos[0].numel() == 0:
            continue
        if pos[0].numel() > max_pairs:
            # deterministic subsample: evenly spaced pairs avoids stochastic instability across machines
            keep = torch.linspace(0, pos[0].numel() - 1, steps=max_pairs, device=pos[0].device).long()
            i = pos[0][keep]
            j = pos[1][keep]
        else:
            i, j = pos
        losses.append(F.softplus(-(s[i] - s[j])).mean())
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def _listwise_rank_loss(scores: torch.Tensor, utilities: torch.Tensor, group_id: torch.Tensor, cfg: Dict[str, object]) -> torch.Tensor:
    rank_cfg = cfg.get("ranking", {})
    if not bool(rank_cfg.get("enabled", True)):
        return scores.new_tensor(0.0)
    tau = max(float(rank_cfg.get("listwise_temperature", 0.15)), 1e-4)
    losses = []
    utilities = utilities.reshape(-1)
    for gid in torch.unique(group_id):
        idx = torch.where(group_id == gid)[0]
        if idx.numel() < 2:
            continue
        u = utilities[idx]
        s = scores[idx]
        if torch.max(u) - torch.min(u) < 1e-6:
            continue
        p_true = torch.softmax(u / tau, dim=0).detach()
        log_p = torch.log_softmax(s / tau, dim=0)
        losses.append(-(p_true * log_p).sum())
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def _compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict[str, object], pos_weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    weights = cfg["loss_weights"]
    pred_reg = outputs["reg_scaled"]
    y_reg = batch["y_reg_scaled"]

    mse = (pred_reg - y_reg).pow(2).mean(dim=0)
    risk_loss, info_loss, delay_loss, utility_loss = mse[0], mse[1], mse[2], mse[3]

    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss = bce_fn(outputs["best_logit"], batch["y_best"])
    rank_score = outputs["rank_score"]
    rank_target = batch["y_rank_utility"].reshape(-1)
    pair_loss = _pairwise_rank_loss(rank_score, rank_target, batch["group_id"], cfg)
    list_loss = _listwise_rank_loss(rank_score, rank_target, batch["group_id"], cfg)
    rank_mse = (rank_score - rank_target).pow(2).mean()

    total = (
        float(weights["risk"]) * risk_loss
        + float(weights["info"]) * info_loss
        + float(weights["delay"]) * delay_loss
        + float(weights["utility"]) * utility_loss
        + float(weights["best_action"]) * best_loss
        + float(weights.get("pairwise_rank", 0.0)) * pair_loss
        + float(weights.get("listwise_rank", 0.0)) * list_loss
        + float(weights.get("rank_score", 0.0)) * rank_mse
    )
    return {
        "total": total,
        "risk": risk_loss.detach(),
        "info": info_loss.detach(),
        "delay": delay_loss.detach(),
        "utility": utility_loss.detach(),
        "best_action": best_loss.detach(),
        "pairwise_rank": pair_loss.detach(),
        "listwise_rank": list_loss.detach(),
        "rank_score": rank_mse.detach(),
    }


def _epoch(model, loader, cfg, device, optimizer=None, pos_weight=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "risk": 0.0, "info": 0.0, "delay": 0.0, "utility": 0.0, "best_action": 0.0, "pairwise_rank": 0.0, "listwise_rank": 0.0, "rank_score": 0.0}
    n = 0
    if pos_weight is None:
        pos_weight = torch.ones(1, device=device)

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["x"])
        losses = _compute_losses(outputs, batch, cfg, pos_weight)
        if training:
            losses["total"].backward()
            clip_norm = float(cfg["training"].get("gradient_clip_norm", 0.0))
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        bs = batch["x"].shape[0]
        n += bs
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def train(config_path: str) -> None:
    root = repo_root()
    cfg = load_config(config_path)
    set_seed(int(cfg["project"].get("seed", 42)))
    device = get_device(str(cfg["training"].get("device", "auto")))

    prepared = _load_prepared_dataset(root, cfg)
    train_ds = ValuePolicyDataset(prepared, "train")
    val_ds = ValuePolicyDataset(prepared, "val")

    # shuffle=False keeps graph action groups together so pairwise/listwise losses are meaningful.
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    input_dim = int(prepared["x"].shape[1])
    model = ValuePolicyHead(
        input_dim=input_dim,
        hidden_dims=cfg["model"].get("hidden_dims", [128, 64]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        activation=str(cfg["model"].get("activation", "relu")),
    ).to(device)

    y_best_train = prepared["y_best"][train_ds.indices].numpy().reshape(-1)
    positives = float(y_best_train.sum())
    negatives = float(len(y_best_train) - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    best_val = float("inf")
    best_epoch = -1
    patience = int(cfg["training"].get("patience", 20))
    history = []
    checkpoint_path = resolve_path(root, cfg["paths"]["model_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = _epoch(model, train_loader, cfg, device, optimizer=optimizer, pos_weight=pos_weight)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, cfg, device, optimizer=None, pos_weight=pos_weight)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        if val_metrics["total"] < best_val - 1e-8:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": cfg["model"].get("hidden_dims", [128, 64]),
                    "dropout": float(cfg["model"].get("dropout", 0.1)),
                    "activation": str(cfg["model"].get("activation", "relu")),
                    "feature_columns": prepared["feature_columns"],
                    "target_columns": prepared["target_columns"],
                    "target_scaler": prepared["target_scaler"],
                    "regression_targets": prepared["regression_targets"],
                    "selection": cfg.get("selection", {}),
                    "ranking": cfg.get("ranking", {}),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_total={train_metrics['total']:.4f} val_total={val_metrics['total']:.4f} "
                f"val_pair={val_metrics['pairwise_rank']:.4f} val_list={val_metrics['listwise_rank']:.4f}"
            )

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

    hist_path = resolve_path(root, cfg["paths"]["train_history_csv"])
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint": str(checkpoint_path),
        "input_dim": input_dim,
        "num_train_rows": len(train_ds),
        "num_val_rows": len(val_ds),
        "positive_good_action_weight": pos_weight_value,
        "notes": "Training uses pointwise value losses plus pairwise/listwise graph-level ranking losses.",
    }
    write_json(summary, resolve_path(root, "outputs/phase5_train_summary.json"))
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Saved training history to {hist_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
