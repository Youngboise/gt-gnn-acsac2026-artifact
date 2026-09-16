from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

from .utils import load_config, repo_root, resolve_path, write_json


def _metric(y: pd.Series, p: pd.Series) -> dict[str, float]:
    yv = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy()
    pv = pd.to_numeric(p, errors="coerce").fillna(0.0).to_numpy()
    return {
        "mae": float(mean_absolute_error(yv, pv)),
        "rmse": float(np.sqrt(mean_squared_error(yv, pv))),
    }


def summarize(pred_csv: Path, out_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(pred_csv)
    if "eval_split" in df.columns:
        df = df[df["eval_split"] == "test"].copy()
    elif "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if df.empty:
        raise ValueError("No test rows found in Phase 5 predictions. Run evaluate_value_policy.py --split all or --split test.")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"num_test_rows": int(len(df))}
    for target in ["risk", "info", "delay", "utility"]:
        tc = f"true_{target}"
        pc = f"pred_{target}"
        if tc in df.columns and pc in df.columns:
            metrics[target] = _metric(df[tc], df[pc])
    if {"true_best_action", "pred_best_action_prob"}.issubset(df.columns):
        y = df["true_best_action"].astype(int).to_numpy()
        p = pd.to_numeric(df["pred_best_action_prob"], errors="coerce").fillna(0.0).to_numpy()
        metrics["best_action"] = {
            "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
            "average_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
        }
    group_col = "graph_id" if "graph_id" in df.columns else None
    selected_rows = []
    if group_col:
        score_col = None
        for cand in ["calibrated_pred_policy_score", "pred_policy_score", "pred_utility"]:
            if cand in df.columns:
                score_col = cand
                break
        for gid, g in df.groupby(group_col):
            pred_row = g.loc[g[score_col].idxmax()] if score_col else g.iloc[0]
            oracle_row = g.loc[g["true_utility"].idxmax()] if "true_utility" in g.columns else g.iloc[0]
            selected_rows.append({
                "graph_id": gid,
                "selected_action_id": pred_row.get("action_id"),
                "oracle_action_id": oracle_row.get("action_id"),
                "selected_true_utility": float(pred_row.get("true_utility", np.nan)),
                "oracle_true_utility": float(oracle_row.get("true_utility", np.nan)),
                "utility_gap": float(oracle_row.get("true_utility", 0.0) - pred_row.get("true_utility", 0.0)),
                "hit_oracle": int(str(pred_row.get("action_id")) == str(oracle_row.get("action_id"))),
            })
        selected = pd.DataFrame(selected_rows)
        selected.to_csv(out_dir / "phase5_test_selected_actions.csv", index=False)
        metrics["policy_quality"] = {
            "num_graphs": int(len(selected)),
            "selection_score_column": score_col,
            "policy_hit_rate": float(selected["hit_oracle"].mean()) if len(selected) else 0.0,
            "mean_utility_gap": float(selected["utility_gap"].mean()) if len(selected) else 0.0,
            "mean_selected_true_utility": float(selected["selected_true_utility"].mean()) if len(selected) else 0.0,
            "mean_oracle_true_utility": float(selected["oracle_true_utility"].mean()) if len(selected) else 0.0,
        }
    df.to_csv(out_dir / "phase5_predictions_test_only.csv", index=False)
    write_json(metrics, out_dir / "phase5_test_only_summary.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--out_dir", default="outputs/test_only")
    args = parser.parse_args()
    root = repo_root()
    cfg = load_config(args.config)
    pred_csv = resolve_path(root, args.predictions or cfg["paths"]["predictions_csv"])
    out_dir = resolve_path(root, args.out_dir)
    metrics = summarize(pred_csv, out_dir)
    print(metrics)


if __name__ == "__main__":
    main()
