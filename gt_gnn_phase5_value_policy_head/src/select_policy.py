from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import load_config, read_json, repo_root, resolve_path, write_json


def select_policy(config_path: str, predictions: str | None = None) -> None:
    root = repo_root()
    cfg = load_config(config_path)
    pred_path = resolve_path(root, predictions or cfg["paths"]["predictions_csv"])
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: {pred_path}\n"
            "Run evaluate first: python -m src.evaluate_value_policy --config config/value_config.yaml --split all"
        )

    df = pd.read_csv(pred_path)
    group_col = None
    for c in cfg["data"].get("group_columns", []):
        if c in df.columns:
            group_col = c
            break
    if group_col is None:
        group_col = "row_index"

    score_col = cfg["selection"].get("score_column", "pred_utility")
    if score_col not in df.columns:
        raise ValueError(f"Configured score column '{score_col}' is missing from {pred_path}")

    rows = []
    for gid, g in df.groupby(group_col, dropna=False):
        chosen = g.loc[g[score_col].idxmax()].copy()
        oracle_idx = g["true_utility"].idxmax() if "true_utility" in g.columns else None
        oracle_utility = float(g.loc[oracle_idx, "true_utility"]) if oracle_idx is not None else None
        chosen_utility = float(chosen.get("true_utility", float("nan")))
        rows.append({
            "group_column": group_col,
            "group_id": gid,
            "selected_row_index": int(chosen.get("row_index", -1)),
            "selected_score": float(chosen[score_col]),
            "selected_pred_utility": float(chosen.get("pred_utility", chosen[score_col])),
            "selected_true_utility": chosen_utility,
            "oracle_true_utility": oracle_utility,
            "utility_gap_to_oracle": None if oracle_utility is None else oracle_utility - chosen_utility,
            "hit_true_best_action": int(float(chosen.get("true_best_action", 0.0)) >= 0.5),
            "pred_best_action_prob": float(chosen.get("pred_best_action_prob", 0.0)),
            "eval_split": chosen.get("eval_split", "unknown"),
            "action_id": chosen.get("action_id", ""),
            "action_key": chosen.get("action_key", ""),
            "candidate_node": chosen.get("candidate_node", chosen.get("candidate_node_id", chosen.get("node_id", ""))),
            "snapshot_id": chosen.get("snapshot_id", ""),
            "graph_id": chosen.get("graph_id", ""),
        })
    selected = pd.DataFrame(rows)

    out_csv = resolve_path(root, cfg["paths"]["selected_actions_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_csv, index=False)

    summary = {
        "predictions_csv": str(pred_path),
        "selected_actions_csv": str(out_csv),
        "group_column": group_col,
        "score_column": score_col,
        "num_groups": int(len(selected)),
        "policy_hit_rate": float(selected["hit_true_best_action"].mean()) if len(selected) else 0.0,
        "mean_selected_true_utility": float(selected["selected_true_utility"].mean()) if "selected_true_utility" in selected else None,
        "mean_oracle_true_utility": float(selected["oracle_true_utility"].mean()) if "oracle_true_utility" in selected else None,
        "mean_utility_gap_to_oracle": float(selected["utility_gap_to_oracle"].mean()) if "utility_gap_to_oracle" in selected else None,
        "median_utility_gap_to_oracle": float(selected["utility_gap_to_oracle"].median()) if "utility_gap_to_oracle" in selected else None,
    }
    out_json = resolve_path(root, cfg["paths"]["policy_summary_json"])
    write_json(summary, out_json)

    print(f"Saved selected actions to {out_csv}")
    print(f"Saved policy summary to {out_json}")
    print(f"Policy hit rate: {summary['policy_hit_rate']:.4f}")
    print(f"Mean utility gap: {summary['mean_utility_gap_to_oracle']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()
    select_policy(args.config, args.predictions)


if __name__ == "__main__":
    main()
