from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from .model import ValuePolicyHead
from .utils import get_device, inverse_standardizer, load_config, repo_root, resolve_path, safe_auc, write_json
from .value_dataset import ValuePolicyDataset


def load_model(checkpoint_path: Path, device: torch.device) -> ValuePolicyHead:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ValuePolicyHead(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=ckpt.get("hidden_dims", [128, 64]),
        dropout=float(ckpt.get("dropout", 0.1)),
        activation=str(ckpt.get("activation", "relu")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _norm(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=v.index)
    return ((v - lo) / (hi - lo)).clip(0.0, 1.0)


def _add_policy_score(df: pd.DataFrame, group_col: str | None, cfg: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    sel = cfg.get("selection", {})
    weights = sel.get("weights", {})
    if not bool(sel.get("component_aware", True)):
        out["pred_policy_score"] = out.get("pred_utility", 0.0)
        return out
    if group_col is None or group_col not in out.columns:
        group_col = "row_index"
    parts = []
    for _, g in out.groupby(group_col, dropna=False, sort=False):
        r = _norm(g["pred_risk"]) if "pred_risk" in g else pd.Series(0.5, index=g.index)
        i = _norm(g["pred_info"]) if "pred_info" in g else pd.Series(0.5, index=g.index)
        d = _norm(g["pred_delay"]) if "pred_delay" in g else pd.Series(0.5, index=g.index)
        q = _norm(g["pred_rank_score"]) if "pred_rank_score" in g else pd.Series(0.5, index=g.index)
        b = pd.to_numeric(g.get("pred_best_action_prob", 0.0), errors="coerce").fillna(0.0)
        score = (
            float(weights.get("risk", 0.34)) * r
            + float(weights.get("info", 0.29)) * i
            + float(weights.get("delay", 0.20)) * d
            + float(weights.get("rank_score", 0.12)) * q
            + float(weights.get("best_prob", 0.05)) * b
        )
        parts.append(score)
    out["pred_policy_score"] = pd.concat(parts).sort_index()
    return out


def predict_split(model, prepared: Dict[str, object], split_name: str, cfg: Dict[str, object], device: torch.device) -> pd.DataFrame:
    ds = ValuePolicyDataset(prepared, split_name)
    loader = DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 128)), shuffle=False)

    reg_preds_scaled = []
    reg_true_raw = []
    best_true = []
    rank_true = []
    row_indices = []
    best_probs = []
    rank_scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)
            reg_preds_scaled.append(out["reg_scaled"].detach().cpu().numpy())
            reg_true_raw.append(batch["y_reg_raw"].detach().cpu().numpy())
            best_true.append(batch["y_best"].detach().cpu().numpy())
            rank_true.append(batch["y_rank_utility"].detach().cpu().numpy())
            row_indices.append(batch["row_index"].detach().cpu().numpy())
            best_probs.append(torch.sigmoid(out["best_logit"]).detach().cpu().numpy())
            rank_scores.append(out["rank_score"].detach().cpu().numpy())

    pred_scaled = np.concatenate(reg_preds_scaled, axis=0)
    true_raw = np.concatenate(reg_true_raw, axis=0)
    y_best = np.concatenate(best_true, axis=0).reshape(-1)
    y_rank = np.concatenate(rank_true, axis=0).reshape(-1)
    row_idx = np.concatenate(row_indices, axis=0).reshape(-1)
    p_best = np.concatenate(best_probs, axis=0).reshape(-1)
    rank_score = np.concatenate(rank_scores, axis=0).reshape(-1)

    pred_raw = inverse_standardizer(pred_scaled, prepared["target_scaler"])
    target_names = prepared["regression_targets"]

    info = prepared["row_info"].iloc[row_idx].reset_index(drop=True).copy()
    info["eval_split"] = split_name
    for j, name in enumerate(target_names):
        info[f"true_{name}"] = true_raw[:, j]
        info[f"pred_{name}"] = pred_raw[:, j]
        info[f"error_{name}"] = pred_raw[:, j] - true_raw[:, j]
    info["true_best_action"] = y_best
    info["true_rank_utility"] = y_rank
    info["pred_best_action_prob"] = p_best
    info["pred_best_action"] = (p_best >= 0.5).astype(int)
    info["pred_rank_score"] = rank_score
    info = _add_policy_score(info, prepared.get("group_column"), cfg)
    return info


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"mse": float(mean_squared_error(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))), "mae": float(mean_absolute_error(y_true, y_pred))}


def _spearman_by_group(pred_df: pd.DataFrame, score_col: str, group_col: str | None) -> float | None:
    if group_col is None or group_col not in pred_df.columns:
        return None
    vals = []
    for _, g in pred_df.groupby(group_col, dropna=False):
        if len(g) < 3:
            continue
        a = g["true_rank_utility"].rank().to_numpy()
        b = g[score_col].rank().to_numpy()
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals)) if vals else None


def evaluate_policy_quality(pred_df: pd.DataFrame, group_col: str | None, score_col: str = "pred_policy_score") -> Dict[str, float]:
    if group_col is None or group_col not in pred_df.columns:
        group_col = "row_index"
    selected = []
    for gid, g in pred_df.groupby(group_col, dropna=False):
        use_col = score_col if score_col in g.columns else "pred_utility"
        pred_row = g.loc[g[use_col].idxmax()]
        true_best_value = float(g["true_utility"].max())
        selected_value = float(pred_row["true_utility"])
        hit = int(float(pred_row["true_best_action"]) >= 0.5)
        selected.append({"group": gid, "selected_true_utility": selected_value, "oracle_true_utility": true_best_value, "utility_gap": true_best_value - selected_value, "hit_best_action": hit})
    s = pd.DataFrame(selected)
    return {
        "num_groups": int(len(s)),
        "selection_score_column": score_col,
        "policy_hit_rate": float(s["hit_best_action"].mean()) if len(s) else 0.0,
        "mean_selected_true_utility": float(s["selected_true_utility"].mean()) if len(s) else 0.0,
        "mean_oracle_true_utility": float(s["oracle_true_utility"].mean()) if len(s) else 0.0,
        "mean_utility_gap": float(s["utility_gap"].mean()) if len(s) else 0.0,
        "median_utility_gap": float(s["utility_gap"].median()) if len(s) else 0.0,
    }


def evaluate(config_path: str, checkpoint: str | None = None, split: str = "test") -> None:
    root = repo_root()
    cfg = load_config(config_path)
    device = get_device(str(cfg["training"].get("device", "auto")))

    dataset_path = resolve_path(root, cfg["paths"]["processed_dataset_pt"])
    prepared = torch.load(dataset_path, map_location="cpu", weights_only=False)

    checkpoint_path = resolve_path(root, checkpoint or cfg["paths"]["model_checkpoint"])
    model = load_model(checkpoint_path, device)

    splits: List[str]
    if split == "all":
        splits = [s for s in ["train", "val", "test"] if (np.asarray(prepared["split"]) == s).any()]
    else:
        splits = [split]

    pred_frames = [predict_split(model, prepared, s, cfg, device) for s in splits]
    pred_df = pd.concat(pred_frames, ignore_index=True)

    metrics: Dict[str, object] = {"checkpoint": str(checkpoint_path), "splits": splits, "num_rows": int(len(pred_df))}
    for target in prepared["regression_targets"]:
        metrics[target] = regression_metrics(pred_df[f"true_{target}"].to_numpy(), pred_df[f"pred_{target}"].to_numpy())

    y_true = pred_df["true_best_action"].to_numpy().astype(int)
    p = pred_df["pred_best_action_prob"].to_numpy()
    y_hat = pred_df["pred_best_action"].to_numpy().astype(int)
    metrics["best_action"] = {
        "accuracy": float(accuracy_score(y_true, y_hat)) if len(np.unique(y_true)) > 1 else float((y_true == y_hat).mean()),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "auc": safe_auc(y_true, p),
        "average_precision": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else None,
    }
    score_col = cfg.get("selection", {}).get("score_column", "pred_policy_score")
    metrics["policy_quality"] = evaluate_policy_quality(pred_df, prepared.get("group_column"), score_col=score_col)
    metrics["ranking"] = {
        "mean_group_spearman_pred_policy_score": _spearman_by_group(pred_df, "pred_policy_score", prepared.get("group_column")),
        "mean_group_spearman_pred_rank_score": _spearman_by_group(pred_df, "pred_rank_score", prepared.get("group_column")),
        "pred_utility_std": float(pred_df["pred_utility"].std()),
        "pred_policy_score_std": float(pred_df["pred_policy_score"].std()),
    }

    out_csv = resolve_path(root, cfg["paths"]["predictions_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    out_json = resolve_path(root, cfg["paths"]["evaluation_summary_json"])
    write_json(metrics, out_json)

    print(f"Saved predictions to {out_csv}")
    print(f"Saved evaluation summary to {out_json}")
    print("Key metrics:")
    print(f"  utility RMSE: {metrics['utility']['rmse']:.4f}")
    print(f"  utility MAE : {metrics['utility']['mae']:.4f}")
    print(f"  best-action accuracy: {metrics['best_action']['accuracy']:.4f}")
    print(f"  policy hit rate: {metrics['policy_quality']['policy_hit_rate']:.4f}")
    print(f"  mean utility gap: {metrics['policy_quality']['mean_utility_gap']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    args = parser.parse_args()
    evaluate(args.config, checkpoint=args.checkpoint, split=args.split)


if __name__ == "__main__":
    main()
