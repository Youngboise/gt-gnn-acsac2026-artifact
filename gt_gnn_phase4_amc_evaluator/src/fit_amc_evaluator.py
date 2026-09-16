from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .amc_core import evaluate_graph_actions
from .phase3_io import filter_graphs, load_phase3_strategic_graphs
from .utils import clamp, load_yaml, package_root_from_config, pearsonr, read_json, safe_float, set_seed, write_json


TUNABLE_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature": (0.45, 1.30),
    "transition_base_scale": (0.70, 2.10),
    "reliability_scale": (0.10, 0.85),
    "movement_cost_scale": (0.25, 1.45),
    "target_bias_scale": (0.30, 1.60),
    "target_proximity_scale": (0.20, 1.25),
    "skill_movement_scale": (0.05, 0.55),
    "edge_deception_penalty_scale": (0.15, 1.15),
    "action_avoidance_scale": (0.10, 1.35),
    "action_bait_scale": (0.05, 0.70),
    "action_decoy_strength": (0.16, 0.72),
    "revelation_decoy_strength": (0.05, 0.45),
    "awareness_decoy_avoidance": (0.20, 0.90),
    "stealth_decoy_avoidance": (0.05, 0.55),
    "noisy_decoy_exposure": (0.05, 0.40),
    "information_gain_scale": (0.45, 1.20),
    "delay_conversion_scale": (1.0, 6.0),
}


def load_intervention_label_lookup(path: Path) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"graph_id", "action_id", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Intervention label file is missing columns: {sorted(missing)}")
    lookup: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (str(row["graph_id"]), str(row["action_id"]), str(row["node_id"]))
        lookup[key] = {str(k): safe_float(v) for k, v in row.items() if k not in {"graph_id", "action_id", "node_id", "split", "label_source"}}
    return lookup


def _label_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row.get("graph_id", "")), str(row.get("action_id", "")), str(row.get("node_id", "")))


def sample_params(defaults: Dict[str, Any], trial: int, rng: random.Random) -> Dict[str, float]:
    params = {k: safe_float(v) for k, v in defaults.items()}
    if trial == 0:
        return params
    for key, (lo, hi) in TUNABLE_RANGES.items():
        base = safe_float(params.get(key), (lo + hi) / 2)
        # Mix local perturbation with occasional wide search.
        if rng.random() < 0.70:
            span = hi - lo
            val = rng.gauss(base, 0.20 * span)
        else:
            val = rng.uniform(lo, hi)
        params[key] = clamp(val, lo, hi)
    return params


def _target_values(row: Dict[str, Any], label_lookup: Dict[Tuple[str, str, str], Dict[str, float]] | None = None, require_labels: bool = False) -> Dict[str, float]:
    label_lookup = label_lookup or {}
    label = label_lookup.get(_label_key(row))
    if label is not None:
        return {
            "risk": safe_float(label.get("target_reach_reduction_cf", label.get("target_reach_reduction", 0.0))),
            "info": safe_float(label.get("expected_info_gain_cf", label.get("expected_info_gain", 0.0))),
            "delay": safe_float(label.get("delay_gain_cf", label.get("delay_gain", 0.0))),
        }
    if require_labels:
        raise KeyError(f"Missing counterfactual intervention label for action key={_label_key(row)}")
    # Debug-only fallback. ACSAC runs should set fitting.require_intervention_labels=true.
    risk_t = safe_float(row.get("risk_proxy"), 0.0)
    info_t = safe_float(row.get("info_proxy"), 0.0) * safe_float(row.get("belief_entropy"), 1.0)
    delay_t = safe_float(row.get("delay_proxy"), 0.0)
    return {"risk": risk_t, "info": info_t, "delay": delay_t}


def objective_for_params(
    graphs: List[Dict[str, Any]],
    params: Dict[str, float],
    labels: Dict[str, List[str]],
    utility_cfg: Dict[str, float],
    objective_weights: Dict[str, float],
    max_actions: Optional[int] = None,
    intervention_label_lookup: Dict[Tuple[str, str, str], Dict[str, float]] | None = None,
    require_intervention_labels: bool = False,
) -> Dict[str, float]:
    risk_err: List[float] = []
    info_err: List[float] = []
    delay_err: List[float] = []
    risk_pred: List[float] = []
    risk_targ: List[float] = []
    info_pred: List[float] = []
    info_targ: List[float] = []
    delay_pred: List[float] = []
    delay_targ: List[float] = []

    for graph in graphs:
        rows = evaluate_graph_actions(graph, params, labels, utility_cfg, max_actions=max_actions)
        for row in rows:
            target = _target_values(row, intervention_label_lookup, require_intervention_labels)
            rp = safe_float(row.get("relative_risk_reduction"), 0.0)
            ip = safe_float(row.get("expected_info_gain"), 0.0)
            dp = safe_float(row.get("delay_gain"), 0.0)
            rt, it, dt = target["risk"], target["info"], target["delay"]
            risk_err.append((rp - rt) ** 2)
            info_err.append((ip - it) ** 2)
            delay_err.append((dp - dt) ** 2)
            risk_pred.append(rp); risk_targ.append(rt)
            info_pred.append(ip); info_targ.append(it)
            delay_pred.append(dp); delay_targ.append(dt)

    if not risk_err:
        return {"loss": float("inf"), "n": 0}

    risk_mse = float(np.mean(risk_err))
    info_mse = float(np.mean(info_err))
    delay_mse = float(np.mean(delay_err))
    loss = (
        safe_float(objective_weights.get("risk"), 1.0) * risk_mse
        + safe_float(objective_weights.get("info"), 0.65) * info_mse
        + safe_float(objective_weights.get("delay"), 0.65) * delay_mse
    )
    return {
        "loss": float(loss),
        "risk_mse": risk_mse,
        "info_mse": info_mse,
        "delay_mse": delay_mse,
        "risk_corr": pearsonr(risk_pred, risk_targ),
        "info_corr": pearsonr(info_pred, info_targ),
        "delay_corr": pearsonr(delay_pred, delay_targ),
        "n": int(len(risk_err)),
    }


def fit(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    set_seed(int(cfg.get("seed", 17)))
    rng = random.Random(int(cfg.get("seed", 17)))
    start = time.time()

    phase3_path = root / cfg["paths"]["phase3_strategic_graphs_jsonl"]
    graphs_all = load_phase3_strategic_graphs(phase3_path)
    fit_cfg = cfg.get("fitting", {})
    train_graphs = filter_graphs(graphs_all, split=str(fit_cfg.get("split", "train")), max_graphs=fit_cfg.get("max_train_graphs"))
    if not train_graphs:
        print("No train split graphs found; falling back to all graphs for fitting.")
        train_graphs = filter_graphs(graphs_all, split="all", max_graphs=fit_cfg.get("max_train_graphs"))
    val_graphs = filter_graphs(graphs_all, split=str(fit_cfg.get("validation_split", "val")), max_graphs=fit_cfg.get("max_validation_graphs"))
    if not val_graphs:
        val_graphs = train_graphs[: min(len(train_graphs), int(fit_cfg.get("max_validation_graphs") or 100))]

    labels = cfg.get("labels", {})
    utility_cfg = cfg.get("utility", {})
    objective_weights = fit_cfg.get("objective_weights", {})
    max_actions = fit_cfg.get("candidate_actions_per_graph", None)
    require_intervention_labels = bool(fit_cfg.get("require_intervention_labels", True))
    label_path = root / cfg.get("paths", {}).get("intervention_labels_csv", "data/intervention_rollout_labels.csv")
    intervention_label_lookup = load_intervention_label_lookup(label_path)
    if require_intervention_labels and not intervention_label_lookup:
        raise FileNotFoundError(
            f"Counterfactual intervention labels are required but not found at {label_path}. "
            "Run: python -m src.simulate_intervention_labels --config config/amc_config.yaml"
        )
    defaults = cfg.get("amc", {})
    trials = int(fit_cfg.get("random_search_trials", 90))

    history: List[Dict[str, Any]] = []
    best_params: Optional[Dict[str, float]] = None
    best_record: Optional[Dict[str, Any]] = None

    print(f"Loaded strategic graphs: {len(graphs_all):,} from {phase3_path}")
    print(f"Fitting on graphs: {len(train_graphs):,}; validation graphs: {len(val_graphs):,}; trials: {trials}")

    for trial in range(trials):
        params = sample_params(defaults, trial, rng)
        train_obj = objective_for_params(train_graphs, params, labels, utility_cfg, objective_weights, max_actions=max_actions, intervention_label_lookup=intervention_label_lookup, require_intervention_labels=require_intervention_labels)
        val_obj = objective_for_params(val_graphs, params, labels, utility_cfg, objective_weights, max_actions=max_actions, intervention_label_lookup=intervention_label_lookup, require_intervention_labels=require_intervention_labels)
        record = {
            "trial": trial,
            "train_loss": train_obj["loss"],
            "val_loss": val_obj["loss"],
            "train_risk_mse": train_obj.get("risk_mse", 0.0),
            "train_info_mse": train_obj.get("info_mse", 0.0),
            "train_delay_mse": train_obj.get("delay_mse", 0.0),
            "val_risk_mse": val_obj.get("risk_mse", 0.0),
            "val_info_mse": val_obj.get("info_mse", 0.0),
            "val_delay_mse": val_obj.get("delay_mse", 0.0),
            "train_n": train_obj.get("n", 0),
            "val_n": val_obj.get("n", 0),
        }
        for key in TUNABLE_RANGES:
            record[f"param_{key}"] = params[key]
        history.append(record)
        if best_record is None or record["val_loss"] < best_record["val_loss"]:
            best_record = record
            best_params = copy.deepcopy(params)
        if (trial + 1) % max(1, trials // 10) == 0 or trial == 0:
            print(f"trial {trial + 1:03d}/{trials}: train_loss={record['train_loss']:.5f} val_loss={record['val_loss']:.5f}")

    assert best_params is not None and best_record is not None
    params_path = root / cfg["paths"]["fitted_params_json"]
    hist_path = root / cfg["paths"]["fit_history_csv"]
    summary_path = root / cfg["paths"]["fit_summary_json"]
    pd.DataFrame(history).to_csv(hist_path, index=False)
    write_json(best_params, params_path)
    summary = {
        "phase": "Phase 4 - Game/AMC evaluator fitting",
        "source_phase3_graphs": str(phase3_path),
        "num_graphs_total": len(graphs_all),
        "num_train_graphs": len(train_graphs),
        "num_validation_graphs": len(val_graphs),
        "trials": trials,
        "best_trial": best_record,
        "best_params_json": str(params_path),
        "fit_history_csv": str(hist_path),
        "elapsed_seconds": round(time.time() - start, 4),
        "intervention_labels_csv": str(label_path),
        "num_intervention_labels": len(intervention_label_lookup),
        "require_intervention_labels": require_intervention_labels,
        "notes": "Fitting uses counterfactual rollout/intervention labels when require_intervention_labels=true; Phase 3 proxy fallback is debug-only.",
    }
    write_json(summary, summary_path)
    print(f"Best trial: {best_record['trial']} val_loss={best_record['val_loss']:.6f}")
    print(f"Wrote fitted params to: {params_path}")
    print(f"Wrote fit history to: {hist_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/amc_config.yaml")
    args = parser.parse_args()
    fit(args.config)


if __name__ == "__main__":
    main()
