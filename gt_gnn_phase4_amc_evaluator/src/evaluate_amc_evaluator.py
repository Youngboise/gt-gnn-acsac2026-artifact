from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .amc_core import evaluate_graph_actions
from .phase3_io import load_phase3_strategic_graphs
from .utils import load_yaml, package_root_from_config, pearsonr, read_json, safe_float, set_seed, write_json, write_jsonl


def _merge_counterfactual_labels(df: pd.DataFrame, label_csv: Path, use_as_targets: bool) -> pd.DataFrame:
    if not use_as_targets or not label_csv.exists() or df.empty:
        return df
    labels = pd.read_csv(label_csv)
    keys = ["graph_id", "action_id", "node_id"]
    missing = set(keys) - set(labels.columns)
    if missing:
        raise ValueError(f"Counterfactual label CSV is missing columns: {sorted(missing)}")
    merged = df.merge(labels, on=keys, how="left", suffixes=("", "_cf_label"))
    required = ["target_reach_reduction_cf", "expected_info_gain_cf", "delay_gain_cf", "utility_counterfactual_norm"]
    missing_labels = merged[required].isna().any(axis=1).sum() if all(c in merged.columns for c in required) else len(merged)
    if missing_labels:
        raise ValueError(f"Missing counterfactual labels for {int(missing_labels)} action rows. Regenerate labels after Phase 3.")
    rename_pairs = {
        "baseline_target_prob_cf": "baseline_target_prob",
        "action_target_prob_cf": "action_target_prob",
        "target_reach_reduction_cf": "target_reach_reduction",
        "action_decoy_absorption_prob_cf": "action_decoy_absorption_prob",
        "expected_info_gain_cf": "expected_info_gain",
        "delay_gain_cf": "delay_gain",
        "utility_counterfactual_raw": "utility_amc_raw",
        "utility_counterfactual_norm": "utility_amc_norm",
        "is_best_action_cf": "is_best_action",
    }
    for src, dst in rename_pairs.items():
        if src in merged.columns:
            merged[dst] = merged[src]
    merged["label_source"] = "counterfactual_rollout"
    return merged


def _rank_and_export(rows: List[Dict[str, Any]], normalize_per_graph: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "utility_amc_raw" in df.columns:
        df["utility_amc_raw"] = df["utility_amc_raw"].astype(float)
    else:
        df["utility_amc_raw"] = df["utility_amc"].astype(float)
    if normalize_per_graph:
        norm_values = []
        for _, group in df.groupby("graph_id", sort=False):
            vals = group["utility_amc_raw"].astype(float)
            lo, hi = float(vals.min()), float(vals.max())
            if hi - lo < 1e-12:
                norm = np.ones(len(group), dtype=np.float64) * 0.5
            else:
                norm = (vals.to_numpy() - lo) / (hi - lo)
            norm_values.extend(norm.tolist())
        df["utility_amc_norm"] = norm_values
    else:
        vals = df["utility_amc_raw"].astype(float)
        lo, hi = float(vals.min()), float(vals.max())
        df["utility_amc_norm"] = 0.5 if hi - lo < 1e-12 else (vals - lo) / (hi - lo)

    df["amc_rank"] = df.groupby("graph_id")["utility_amc_raw"].rank(method="first", ascending=False).astype(int)
    df["is_best_action"] = (df["amc_rank"] == 1).astype(int)
    best_utility = df.groupby("graph_id")["utility_amc_raw"].transform("max")
    df["utility_gap_to_best"] = best_utility - df["utility_amc_raw"]
    return df


def _summary(df: pd.DataFrame, phase3_path: Path, params_path: Path) -> Dict[str, Any]:
    if df.empty:
        return {"num_actions": 0}
    out: Dict[str, Any] = {
        "phase": "Phase 4 - Game/AMC evaluator output",
        "source_phase3_graphs": str(phase3_path),
        "params_json": str(params_path),
        "num_actions": int(len(df)),
        "num_graphs": int(df["graph_id"].nunique()),
        "mean_relative_risk_reduction": float(df["relative_risk_reduction"].mean()),
        "mean_expected_info_gain": float(df["expected_info_gain"].mean()),
        "mean_delay_gain": float(df["delay_gain"].mean()),
        "mean_utility_amc": float(df["utility_amc_raw"].mean()),
        "corr_phase3_fused_vs_amc_utility": pearsonr(df["fused_score"].tolist(), df["utility_amc_raw"].tolist()),
        "corr_risk_proxy_vs_relative_risk_reduction": pearsonr(df["risk_proxy"].tolist(), df["relative_risk_reduction"].tolist()),
        "corr_info_proxy_vs_expected_info_gain": pearsonr(df["info_proxy"].tolist(), df["expected_info_gain"].tolist()),
        "corr_delay_proxy_vs_delay_gain": pearsonr(df["delay_proxy"].tolist(), df["delay_gain"].tolist()),
    }
    split_summary: Dict[str, Any] = {}
    for split, g in df.groupby("split"):
        split_summary[str(split)] = {
            "num_actions": int(len(g)),
            "num_graphs": int(g["graph_id"].nunique()),
            "mean_relative_risk_reduction": float(g["relative_risk_reduction"].mean()),
            "mean_expected_info_gain": float(g["expected_info_gain"].mean()),
            "mean_delay_gain": float(g["delay_gain"].mean()),
            "mean_utility_amc": float(g["utility_amc_raw"].mean()),
        }
    out["by_split"] = split_summary
    best = df[df["is_best_action"] == 1]
    out["best_action_summary"] = {
        "num_best_actions": int(len(best)),
        "mean_best_relative_risk_reduction": float(best["relative_risk_reduction"].mean()) if len(best) else 0.0,
        "mean_best_info_gain": float(best["expected_info_gain"].mean()) if len(best) else 0.0,
        "mean_best_delay_gain": float(best["delay_gain"].mean()) if len(best) else 0.0,
        "mean_best_utility": float(best["utility_amc_raw"].mean()) if len(best) else 0.0,
    }
    return out


def evaluate(config_path: str, params_path_arg: str | None = None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    set_seed(int(cfg.get("seed", 17)))
    start = time.time()

    phase3_path = root / cfg["paths"]["phase3_strategic_graphs_jsonl"]
    params_path = root / (params_path_arg or cfg["paths"]["fitted_params_json"])
    if not params_path.exists():
        raise FileNotFoundError(f"Fitted parameter file not found: {params_path}. Run fit_amc_evaluator first.")
    params = read_json(params_path)
    graphs = load_phase3_strategic_graphs(phase3_path)
    labels = cfg.get("labels", {})
    utility_cfg = cfg.get("utility", {})

    rows: List[Dict[str, Any]] = []
    for i, graph in enumerate(graphs):
        rows.extend(evaluate_graph_actions(graph, params, labels, utility_cfg))
        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1:,}/{len(graphs):,} strategic graphs...")

    df = _rank_and_export(rows, normalize_per_graph=bool(cfg.get("export", {}).get("normalize_utility_per_graph", True)))
    label_csv = root / cfg.get("paths", {}).get("intervention_labels_csv", "data/intervention_rollout_labels.csv")
    df = _merge_counterfactual_labels(df, label_csv, bool(cfg.get("export", {}).get("use_intervention_labels_as_phase5_targets", True)))
    if "utility_amc_raw" in df.columns:
        df = _rank_and_export(df.to_dict(orient="records"), normalize_per_graph=bool(cfg.get("export", {}).get("normalize_utility_per_graph", True)))
    action_csv = root / cfg["paths"]["action_outcomes_csv"]
    action_jsonl = root / cfg["paths"]["action_outcomes_jsonl"]
    best_csv = root / cfg["paths"]["best_actions_csv"]
    phase5_csv = root / cfg["paths"]["phase5_training_csv"]
    summary_json = root / cfg["paths"]["evaluation_summary_json"]

    action_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(action_csv, index=False)
    write_jsonl(df.to_dict(orient="records"), action_jsonl)
    best_df = df[df["is_best_action"] == 1].copy() if not df.empty else df.copy()
    best_df.to_csv(best_csv, index=False)

    # Phase 5 training table keeps input features and target values in one flat file.
    keep_cols = [
        "graph_id", "episode_id", "t", "split", "action_id", "node_id", "rank_phase3",
        "action_type", "decoy_type_hint", "exposure_level_hint", "action_is_high_interaction", "action_is_low_exposure",
        "action_is_credential_breadcrumb", "action_is_telemetry_trap", "action_is_path_perturbation", "action_is_rotate_or_retain", "exposure_level_value",
        "risk_proxy", "info_proxy", "delay_proxy", "threat_score", "revelation_score", "fused_score", "candidate_score",
        "belief_entropy", "operational_cost", "churn_proxy",
        "baseline_target_prob", "action_target_prob", "relative_risk_reduction", "target_reach_reduction",
        "action_decoy_absorption_prob", "expected_info_gain", "delay_gain", "utility_amc_raw", "utility_amc_norm",
        "amc_rank", "is_best_action", "utility_gap_to_best", "label_source",
    ]
    phase5_cols = [c for c in keep_cols if c in df.columns]
    df[phase5_cols].to_csv(phase5_csv, index=False)

    summary = _summary(df, phase3_path, params_path)
    summary["output_files"] = {
        "action_outcomes_csv": str(action_csv),
        "action_outcomes_jsonl": str(action_jsonl),
        "best_actions_csv": str(best_csv),
        "phase5_training_csv": str(phase5_csv),
    }
    summary["elapsed_seconds"] = round(time.time() - start, 4)
    summary["counterfactual_labels_csv"] = str(label_csv)
    summary["using_counterfactual_labels_for_phase5"] = bool(cfg.get("export", {}).get("use_intervention_labels_as_phase5_targets", True))
    summary["notes"] = "AMC outputs are calibrated with counterfactual rollout labels; Phase 5 training targets are overridden by those labels when configured."
    write_json(summary, summary_json)

    print(f"Loaded strategic graphs: {len(graphs):,} from {phase3_path}")
    print(f"Evaluated candidate actions: {len(df):,}")
    print(f"Wrote action outcomes to: {action_csv}")
    print(f"Wrote Phase 5 training table to: {phase5_csv}")
    print(f"Wrote summary to: {summary_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/amc_config.yaml")
    parser.add_argument("--params", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.params)


if __name__ == "__main__":
    main()
