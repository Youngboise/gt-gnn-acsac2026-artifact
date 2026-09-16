from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .belief_dataset import BeliefSequenceDataset, LabelMaps, collate_belief_batch
from .belief_model import FactorizedBeliefGRU
from .utils import ensure_parent, expected_calibration_error, load_yaml, set_seed, write_json


FACTOR_KEYS = ["goal", "skill", "stealth", "decoy_awareness"]


def build_loaders(cfg: Dict):
    label_maps = LabelMaps.from_config(cfg["labels"])
    kwargs = {
        "csv_path": cfg["paths"]["dataset_csv"],
        "features": cfg["features"],
        "label_maps": label_maps,
        "window_size": int(cfg["data"]["window_size"]),
    }
    train_ds = BeliefSequenceDataset(split="train", **kwargs)
    val_ds = BeliefSequenceDataset(split="val", **kwargs)
    test_ds = BeliefSequenceDataset(split="test", **kwargs)
    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_belief_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_belief_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_belief_batch)
    return train_loader, val_loader, test_loader


def build_model(cfg: Dict) -> FactorizedBeliefGRU:
    labels = cfg["labels"]
    model_cfg = cfg["model"]
    return FactorizedBeliefGRU(
        feature_dim=len(cfg["features"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        num_goal=len(labels["goal"]),
        num_skill=len(labels["skill"]),
        num_stealth=len(labels["stealth"]),
        num_decoy_awareness=len(labels["decoy_awareness"]),
    )


def compute_loss(logits: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], label_smoothing: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = {}
    total = 0.0
    for key in FACTOR_KEYS:
        loss = nn.functional.cross_entropy(logits[key], y[key], label_smoothing=label_smoothing)
        losses[key] = float(loss.detach().cpu())
        total = total + loss
    return total, losses


@torch.no_grad()
def evaluate(model, loader, device, label_smoothing: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = {k: 0 for k in FACTOR_KEYS}
    total = {k: 0 for k in FACTOR_KEYS}
    probs_store = {k: [] for k in FACTOR_KEYS}
    labels_store = {k: [] for k in FACTOR_KEYS}
    entropy_store = {k: [] for k in FACTOR_KEYS}

    for x, y, _ in loader:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        logits = model(x)
        loss, _ = compute_loss(logits, y, label_smoothing)
        total_loss += float(loss.cpu())
        n_batches += 1

        for key in FACTOR_KEYS:
            p = torch.softmax(logits[key], dim=-1)
            pred = p.argmax(dim=-1)
            correct[key] += int((pred == y[key]).sum().cpu())
            total[key] += int(y[key].numel())
            probs_store[key].append(p.cpu().numpy())
            labels_store[key].append(y[key].cpu().numpy())
            ent = -(p.clamp_min(1e-9) * p.clamp_min(1e-9).log()).sum(dim=-1)
            entropy_store[key].append(ent.cpu().numpy())

    metrics = {"loss": total_loss / max(1, n_batches)}
    acc_values = []
    for key in FACTOR_KEYS:
        probs = np.concatenate(probs_store[key], axis=0)
        labels = np.concatenate(labels_store[key], axis=0)
        entropies = np.concatenate(entropy_store[key], axis=0)
        acc = correct[key] / max(1, total[key])
        acc_values.append(acc)
        metrics[f"{key}_acc"] = float(acc)
        metrics[f"{key}_ece"] = expected_calibration_error(probs, labels)
        metrics[f"{key}_entropy"] = float(entropies.mean())
    metrics["macro_acc"] = float(np.mean(acc_values))
    metrics["macro_ece"] = float(np.mean([metrics[f"{k}_ece"] for k in FACTOR_KEYS]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/belief_config.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)

    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    label_smoothing = float(train_cfg["label_smoothing"])
    grad_clip_norm = float(train_cfg["grad_clip_norm"])
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stop_patience"])

    best_val = -1.0
    best_epoch = -1
    stale = 0
    history = []
    checkpoint_path = cfg["paths"]["checkpoint"]
    ensure_parent(checkpoint_path)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, loss_parts = compute_loss(logits, y, label_smoothing)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            n_batches += 1
            pbar.set_postfix(loss=train_loss / max(1, n_batches))

        val_metrics = evaluate(model, val_loader, device, label_smoothing)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, n_batches),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_acc={val_metrics['macro_acc']:.4f} "
            f"val_macro_ece={val_metrics['macro_ece']:.4f}"
        )

        if val_metrics["macro_acc"] > best_val:
            best_val = val_metrics["macro_acc"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "best_epoch": best_epoch,
                    "best_val_macro_acc": best_val,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    # Final test metrics from best checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, label_smoothing)
    history.append({"test_metrics": test_metrics, "best_epoch": best_epoch})
    write_json(history, cfg["paths"]["train_history"])
    print("Best validation macro accuracy:", round(best_val, 4))
    print("Test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
