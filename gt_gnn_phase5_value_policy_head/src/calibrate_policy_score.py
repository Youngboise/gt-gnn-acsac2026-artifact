from __future__ import annotations

"""Validation-tuned decision-score calibration for Phase 5.

The learned heads predict R/I/D/U and auxiliary ranking signals, but the final
closed-loop policy needs a single deployable action score.  This module tunes a
nonnegative linear combiner on the validation split only, applies it to all
splits, and writes a calibrated prediction table for Phase 6.

It does not use test labels for tuning.  Test labels are retained only for later
paper evaluation, exactly as in the original Phase 6 evaluator.
"""

import argparse
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import load_config, repo_root, resolve_path, write_json


def _norm_within_group(df: pd.DataFrame, group_col: str, col: str, default: float = 0.5) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    v = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    lo = v.groupby(df[group_col]).transform("min")
    hi = v.groupby(df[group_col]).transform("max")
    denom = (hi - lo).replace(0.0, np.nan)
    return ((v - lo) / denom).fillna(default).clip(0.0, 1.0)


def _merge_phase4_predictions(phase4: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ["graph_id", "action_id", "node_id"] if k in phase4.columns and k in pred.columns]
    if len(keys) < 2:
        raise ValueError(f"Cannot merge Phase 4 and Phase 5 predictions; keys={keys}")
    # Drop duplicated Phase 5 columns that are exact labels already carried by Phase 4.
    pred_keep = pred.copy()
    for c in list(pred_keep.columns):
        if c in phase4.columns and c not in keys and c not in {"eval_split"}:
            pred_keep = pred_keep.drop(columns=[c])
    merged = phase4.merge(pred_keep, on=keys, how="left", validate="one_to_one")
    missing_pred = merged[[c for c in ["pred_utility", "pred_risk", "pred_info", "pred_delay"] if c in merged.columns]].isna().any(axis=1).mean()
    if missing_pred > 0.0:
        print(f"[calibrate_policy_score] warning: {missing_pred:.2%} rows have at least one missing prediction")
    return merged


def _utility_col(df: pd.DataFrame) -> str:
    for c in ["utility_amc_norm", "utility_amc_raw", "true_utility", "utility_amc", "utility"]:
        if c in df.columns:
            return c
    raise ValueError("No utility target column found for calibration.")


def _cost_col(df: pd.DataFrame) -> str | None:
    for c in ["operational_cost", "cost"]:
        if c in df.columns:
            return c
    return None


def _candidate_weight_vectors(feature_names: list[str], seed: int, n_random: int) -> list[dict[str, float]]:
    """Construct a compact deterministic search set of weight vectors."""
    # Hand-crafted anchors correspond to risk/info/delay/balanced policies.
    anchors: list[dict[str, float]] = []
    def w(**kw):
        d = {f: 0.0 for f in feature_names}
        d.update({k: float(v) for k, v in kw.items() if k in d})
        s = sum(max(v, 0.0) for v in d.values())
        return {k: (max(v, 0.0) / s if s > 0 else 1.0 / len(d)) for k, v in d.items()}

    anchors.extend([
        w(pred_utility=1.0),
        w(pred_policy_score=1.0),
        w(pred_risk=1.0),
        w(pred_info=1.0),
        w(pred_delay=1.0),
        w(pred_utility=0.35, pred_risk=0.25, pred_info=0.20, pred_delay=0.15, pred_best_action_prob=0.05),
        w(pred_utility=0.25, pred_risk=0.30, pred_info=0.25, pred_delay=0.15, pred_best_action_prob=0.05),
        w(pred_utility=0.20, pred_risk=0.25, pred_info=0.30, pred_delay=0.15, pred_best_action_prob=0.10),
        w(pred_utility=0.20, pred_risk=0.22, pred_info=0.20, pred_delay=0.28, pred_best_action_prob=0.10),
        w(pred_risk=0.32, pred_info=0.28, pred_delay=0.25, pred_best_action_prob=0.10, fused_score=0.05),
        w(pred_risk=0.25, pred_info=0.25, pred_delay=0.25, pred_rank_score=0.15, pred_best_action_prob=0.10),
        w(pred_utility=0.20, risk_proxy=0.18, info_proxy=0.18, delay_proxy=0.18, pred_risk=0.12, pred_info=0.12, pred_delay=0.12),
    ])

    # A few simplex grid points over the most important four components.
    core = [f for f in ["pred_utility", "pred_risk", "pred_info", "pred_delay", "pred_best_action_prob"] if f in feature_names]
    if len(core) >= 3:
        levels = [0.0, 0.15, 0.30, 0.45, 0.60]
        for vals in itertools.product(levels, repeat=len(core)):
            if 0.85 <= sum(vals) <= 1.15 and sum(vals) > 0:
                anchors.append(w(**dict(zip(core, vals))))

    rng = np.random.default_rng(seed)
    alpha = np.ones(len(feature_names), dtype=float)
    for _ in range(int(n_random)):
        vals = rng.dirichlet(alpha)
        anchors.append({f: float(v) for f, v in zip(feature_names, vals)})

    # Deduplicate rounded weights.
    dedup: dict[tuple[float, ...], dict[str, float]] = {}
    for d in anchors:
        key = tuple(round(d.get(f, 0.0), 4) for f in feature_names)
        dedup[key] = d
    return list(dedup.values())


def _score_from_weights(feature_matrix: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=feature_matrix.index, dtype=float)
    for c, w in weights.items():
        if c in feature_matrix.columns and abs(w) > 0:
            score = score + float(w) * feature_matrix[c]
    return score


def _evaluate_weights(df: pd.DataFrame, score: pd.Series, group_col: str, utility_col: str, cost_col: str | None) -> dict[str, float]:
    tmp = df[[group_col, utility_col]].copy()
    tmp["_score"] = score
    if cost_col and cost_col in df.columns:
        tmp["_cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0).astype(float)
    else:
        tmp["_cost"] = 0.0
    idx = tmp.groupby(group_col)["_score"].idxmax()
    selected = tmp.loc[idx]
    oracle = tmp.groupby(group_col)[utility_col].max()
    return {
        "mean_selected_utility": float(selected[utility_col].mean()),
        "mean_oracle_utility": float(oracle.mean()),
        "mean_gap": float((oracle.loc[selected[group_col]].to_numpy() - selected[utility_col].to_numpy()).mean()),
        "mean_cost": float(selected["_cost"].mean()),
        "objective": float(selected[utility_col].mean() - 0.035 * selected["_cost"].mean()),
    }


def calibrate(config_path: str, phase4_csv: str | None = None, predictions_csv: str | None = None, output_csv: str | None = None, n_random: int = 96) -> None:
    root = repo_root()
    cfg = load_config(config_path)
    seed = int(cfg.get("project", {}).get("seed", 42))
    phase4_path = resolve_path(root, phase4_csv or "../gt_gnn_phase4_amc_evaluator/data/phase5_value_training_data.csv")
    pred_path = resolve_path(root, predictions_csv or cfg["paths"].get("predictions_csv", "outputs/phase5_predictions.csv"))
    out_path = resolve_path(root, output_csv or cfg["paths"].get("calibrated_predictions_csv", "outputs/phase5_predictions_calibrated.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    phase4 = pd.read_csv(phase4_path)
    pred = pd.read_csv(pred_path)
    merged = _merge_phase4_predictions(phase4, pred)
    if "split" not in merged.columns:
        raise ValueError("Calibration requires a split column with train/val/test.")
    group_col = "graph_id"
    utility_col = _utility_col(merged)
    cost_col = _cost_col(merged)

    feature_candidates = [
        "pred_utility", "pred_risk", "pred_info", "pred_delay", "pred_rank_score",
        "pred_best_action_prob", "pred_policy_score", "fused_score", "risk_proxy", "info_proxy", "delay_proxy",
        "threat_score", "revelation_score",
    ]
    feature_names = [c for c in feature_candidates if c in merged.columns]
    if not feature_names:
        raise ValueError("No candidate score features found for calibration.")

    features = pd.DataFrame(index=merged.index)
    for c in feature_names:
        features[c] = _norm_within_group(merged, group_col, c, default=0.5)

    val_mask = merged["split"].astype(str).str.lower().eq("val")
    if val_mask.sum() == 0:
        raise ValueError("No validation rows available. Run Phase 5 evaluation with --split all and use a dataset with val split.")

    weights_list = _candidate_weight_vectors(feature_names, seed=seed + 1357, n_random=n_random)
    best: dict[str, Any] | None = None
    records = []
    val_df = merged.loc[val_mask].copy()
    val_features = features.loc[val_mask]
    for k, weights in enumerate(weights_list):
        score = _score_from_weights(val_features, weights)
        metrics = _evaluate_weights(val_df, score, group_col, utility_col, cost_col)
        rec = {"candidate_id": k, **metrics, **{f"w_{name}": weights.get(name, 0.0) for name in feature_names}}
        records.append(rec)
        if best is None or metrics["objective"] > best["objective"]:
            best = {"candidate_id": k, "weights": weights, **metrics}

    assert best is not None
    merged["calibrated_pred_policy_score"] = _score_from_weights(features, best["weights"])
    # Maintain compatibility with old code while preserving the raw model score.
    if "pred_policy_score" in merged.columns:
        merged["raw_pred_policy_score"] = merged["pred_policy_score"]
    merged["pred_policy_score"] = merged["calibrated_pred_policy_score"]
    merged.to_csv(out_path, index=False)

    search_df = pd.DataFrame(records).sort_values("objective", ascending=False)
    search_csv = out_path.with_name("phase5_policy_score_calibration_search.csv")
    search_df.to_csv(search_csv, index=False)
    summary = {
        "phase4_csv": str(phase4_path),
        "predictions_csv": str(pred_path),
        "output_csv": str(out_path),
        "validation_rows": int(val_mask.sum()),
        "validation_graphs": int(val_df[group_col].nunique()),
        "utility_column": utility_col,
        "cost_column": cost_col,
        "feature_names": feature_names,
        "best": best,
        "search_csv": str(search_csv),
        "notes": [
            "Weights are tuned only on the validation split.",
            "The calibrated_pred_policy_score is deployable because it combines predicted heads and pre-action proxy scores, not test outcomes.",
        ],
    }
    write_json(summary, out_path.with_name("phase5_policy_score_calibration.json"))
    print("Best validation-calibrated score:")
    print({k: v for k, v in best.items() if k != "weights"})
    print("Weights:")
    print(best["weights"])
    print(f"Wrote calibrated predictions: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    parser.add_argument("--phase4_csv", default=None)
    parser.add_argument("--predictions_csv", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--n_random", type=int, default=96)
    args = parser.parse_args()
    calibrate(args.config, args.phase4_csv, args.predictions_csv, args.output_csv, args.n_random)


if __name__ == "__main__":
    main()
