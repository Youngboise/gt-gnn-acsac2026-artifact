from __future__ import annotations

"""Paper-facing experiment suite for Phase 6.

This module adds four ACSAC-style evaluation extensions on top of the integrated
closed-loop evaluator:

1. realistic AD/BloodHound-style scale reporting,
2. partial-observability robustness/stress tests,
3. module ablations, and
4. held-out attacker/simulator evaluation to reduce AMC-label circularity.

The implementation is deliberately non-invasive: earlier phases can keep their
existing artifacts, while this layer consumes the Phase 4/5 action table and
produces additional CSV/JSON tables for the paper.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .closed_loop import make_baseline_comparison, resolve_schema, run_closed_loop, summarize_by_policy
from .utils import ensure_parent, first_existing_column, normalize_series, resolve_path, safe_float, stable_hash_int, write_json


@dataclass
class SuiteOutputs:
    scale_summary: pd.DataFrame
    robustness_summary: pd.DataFrame
    ablation_summary: pd.DataFrame
    heldout_attacker_summary: pd.DataFrame
    manifest: dict[str, Any]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _cfg_seed(cfg: dict[str, Any]) -> int:
    return int(cfg.get("project", {}).get("seed", 42))


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = resolve_path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _sample_graphs(df: pd.DataFrame, cfg: dict[str, Any], label: str = "paper_experiments") -> pd.DataFrame:
    """Deterministically subsample complete snapshots for fast experiment sweeps.

    Set paper_experiments.max_graphs: null in the YAML to run every graph.
    """
    exp_cfg = cfg.get("paper_experiments", {})
    max_graphs = exp_cfg.get("max_graphs", None)
    if max_graphs in (None, "", 0):
        return df.copy()

    cols = resolve_schema(df, cfg)
    graphs = pd.Series(df[cols.graph_id].dropna().unique())
    max_graphs = int(max_graphs)
    if len(graphs) <= max_graphs:
        return df.copy()

    seed_offset = stable_hash_int(label, 100_000)
    generator = _rng(_cfg_seed(cfg) + seed_offset)
    chosen = set(generator.choice(graphs.to_numpy(), size=max_graphs, replace=False))
    return df[df[cols.graph_id].isin(chosen)].copy()


def _numeric_col(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _clip01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _set_if_present(df: pd.DataFrame, col: Optional[str], values: pd.Series | np.ndarray | float) -> None:
    if col and col in df.columns:
        df[col] = values


def _available(cols: list[Optional[str]], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c and c in df.columns]


def _component_utility(
    df: pd.DataFrame,
    cols: Any,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    weights = weights or {}
    cost = _numeric_col(df, cols.cost, 0.0)

    risk_col = cols.pred_risk or cols.target_reach_reduction
    info_col = cols.pred_info or cols.info_gain
    delay_col = cols.pred_delay or cols.delay_gain

    risk = normalize_series(_numeric_col(df, risk_col, 0.0))
    info = normalize_series(_numeric_col(df, info_col, 0.0))
    delay = normalize_series(_numeric_col(df, delay_col, 0.0))
    cost_n = normalize_series(cost)

    return (
        float(weights.get("risk", 0.55)) * risk
        + float(weights.get("info", 0.20)) * info
        + float(weights.get("delay", 0.20)) * delay
        - float(weights.get("cost", 0.05)) * cost_n
    )


def _recompute_true_utility(df: pd.DataFrame, cfg: dict[str, Any], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Recompute the evaluation label from perturbed held-out dynamics.

    Utility is normalized within each graph so that oracle gaps remain comparable
    to the Phase 4 AMC-normalized utility tables.
    """
    out = df.copy()
    cols = resolve_schema(out, cfg)
    weights = weights or {}

    risk = _numeric_col(out, cols.target_reach_reduction, 0.0)
    info = _numeric_col(out, cols.info_gain, 0.0)
    delay = _numeric_col(out, cols.delay_gain, 0.0)
    decoy = _numeric_col(out, cols.decoy_absorption, 0.0)
    cost = _numeric_col(out, cols.cost, 0.0)

    raw = (
        float(weights.get("risk", 0.55)) * risk
        + float(weights.get("info", 0.15)) * info
        + float(weights.get("delay", 0.15)) * delay
        + float(weights.get("decoy", 0.20)) * decoy
        - float(weights.get("cost", 0.05)) * cost
    )
    tmp = out[[cols.graph_id]].copy()
    tmp["_utility_raw"] = raw
    min_v = tmp.groupby(cols.graph_id)["_utility_raw"].transform("min")
    max_v = tmp.groupby(cols.graph_id)["_utility_raw"].transform("max")
    denom = (max_v - min_v).replace(0.0, np.nan)
    norm = ((tmp["_utility_raw"] - min_v) / denom).fillna(0.0)
    out[cols.utility_true] = norm.clip(0.0, 1.0)
    return out


def _evaluate_variant(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    variant_name: str,
    variant_type: str,
    variant_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_cfg = variant_cfg or {}
    local_cfg = deepcopy(cfg)
    if "closed_loop" in variant_cfg:
        local_cfg.setdefault("closed_loop", {}).update(variant_cfg["closed_loop"])
    if "policies" in variant_cfg:
        local_cfg.setdefault("policies", {}).update(variant_cfg["policies"])

    eval_df = _sample_graphs(df, local_cfg, label=f"{variant_type}:{variant_name}")
    rollout, _, _ = run_closed_loop(eval_df, local_cfg)
    summary = summarize_by_policy(rollout)
    summary.insert(0, "variant_type", variant_type)
    summary.insert(1, "variant", variant_name)
    return summary, rollout


def _policy_row(summary: pd.DataFrame, policy: str) -> dict[str, Any]:
    hit = summary[summary["policy"] == policy]
    return hit.iloc[0].to_dict() if len(hit) else {}


def _delta_against(base_row: dict[str, Any], row: dict[str, Any], prefix: str = "delta") -> dict[str, float | None]:
    fields = [
        "mean_utility",
        "mean_target_reach_reduction",
        "mean_info_gain",
        "mean_delay_gain",
        "mean_utility_gap_to_oracle",
        "mean_churn",
        "policy_hit_rate_oracle",
    ]
    out: dict[str, float | None] = {}
    for f in fields:
        if f in base_row and f in row and pd.notna(base_row[f]) and pd.notna(row[f]):
            out[f"{prefix}_{f}"] = float(row[f]) - float(base_row[f])
        else:
            out[f"{prefix}_{f}"] = None
    return out


def _apply_observation_missing(df: pd.DataFrame, cfg: dict[str, Any], rate: float) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    rate = float(rate)
    keep = max(0.0, 1.0 - rate)
    gen = _rng(_cfg_seed(cfg) + int(rate * 10_000) + 17)
    missing_mask = pd.Series(gen.random(len(out)) < rate, index=out.index)

    # Missing observations remove information/revelation evidence on a subset of
    # action rows, which can change the selected action rather than merely scaling
    # every candidate by the same constant.
    for col in _available([cols.pred_info, cols.info_proxy, cols.revelation_score], out):
        values = _numeric_col(out, col, 0.0) * keep
        values.loc[missing_mask] = 0.0
        out[col] = values

    if cols.pred_utility:
        original = normalize_series(_numeric_col(out, cols.pred_utility, 0.0))
        no_info = _component_utility(out, cols, {"risk": 0.72, "info": 0.00, "delay": 0.23, "cost": 0.05})
        degraded = ((1.0 - rate) * original + rate * no_info).clip(0.0, 1.0)
        degraded.loc[missing_mask] = no_info.loc[missing_mask]
        out[cols.pred_utility] = degraded

    if cols.fused_score:
        fused = normalize_series(_numeric_col(out, cols.fused_score, 0.0))
        risk_like = normalize_series(_numeric_col(out, cols.risk_proxy or cols.target_reach_reduction, 0.0))
        out[cols.fused_score] = ((1.0 - rate) * fused + rate * risk_like).clip(0.0, 1.0)

    if cols.belief_entropy:
        entropy = _numeric_col(out, cols.belief_entropy, 1.0) + rate
        entropy.loc[missing_mask] = entropy.loc[missing_mask] + 0.25 * rate
        out[cols.belief_entropy] = entropy.clip(0.0, 1.0)
    return out


def _apply_noise(df: pd.DataFrame, cfg: dict[str, Any], sigma: float, seed_offset: int) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    gen = _rng(_cfg_seed(cfg) + seed_offset)
    for col in _available([cols.pred_utility, cols.pred_risk, cols.pred_info, cols.pred_delay, cols.fused_score], out):
        values = _numeric_col(out, col, 0.0)
        noisy = values + gen.normal(0.0, float(sigma), size=len(out))
        out[col] = noisy.clip(values.min(), values.max())
    return out


def _apply_delayed_observation(df: pd.DataFrame, cfg: dict[str, Any], lag: int = 1, blend: float = 0.65) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    sort_cols = [cols.episode_id, cols.t, cols.graph_id]
    out = out.sort_values(sort_cols).copy()
    for col in _available([cols.pred_utility, cols.pred_info, cols.fused_score, cols.revelation_score], out):
        shifted = out.groupby(cols.episode_id)[col].shift(int(lag))
        out[col] = (float(blend) * shifted.fillna(out[col]) + (1.0 - float(blend)) * out[col]).astype(float)
    return out


def _apply_wrong_prior(df: pd.DataFrame, cfg: dict[str, Any], severity: float) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    severity = float(severity)
    cost = normalize_series(_numeric_col(out, cols.cost, 0.0))
    for col in _available([cols.pred_utility, cols.pred_info, cols.fused_score], out):
        current = normalize_series(_numeric_col(out, col, 0.0))
        # Wrong prior makes the model overvalue low-cost stable choices and undervalue information.
        out[col] = ((1.0 - severity) * current + severity * (1.0 - cost)).clip(0.0, 1.0)
    if cols.belief_entropy:
        out[cols.belief_entropy] = (_numeric_col(out, cols.belief_entropy, 1.0) + 0.50 * severity).clip(0.0, 1.0)
    return out


def _apply_unseen_attacker(df: pd.DataFrame, cfg: dict[str, Any], severity: float) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    severity = float(severity)
    # Selection scores are stale, but the realized dynamics shift toward stealthy target seeking.
    _set_if_present(out, cols.target_reach_reduction, _clip01(_numeric_col(out, cols.target_reach_reduction) * (1.0 - 0.25 * severity)))
    _set_if_present(out, cols.info_gain, _clip01(_numeric_col(out, cols.info_gain) * (1.0 - 0.45 * severity)))
    _set_if_present(out, cols.delay_gain, _clip01(_numeric_col(out, cols.delay_gain) * (1.0 + 0.20 * severity)))
    _set_if_present(out, cols.decoy_absorption, _clip01(_numeric_col(out, cols.decoy_absorption) * (1.0 - 0.35 * severity)))
    weights = {"risk": 0.65, "info": 0.08, "delay": 0.17, "decoy": 0.12, "cost": 0.05}
    return _recompute_true_utility(out, cfg, weights=weights)


def _apply_decoy_aware_shift(df: pd.DataFrame, cfg: dict[str, Any], ratio: float) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    ratio = float(ratio)
    avoid = min(max(ratio, 0.0), 1.0)
    # More decoy-aware attackers reduce direct decoy absorption and information revelation.
    _set_if_present(out, cols.decoy_absorption, _clip01(_numeric_col(out, cols.decoy_absorption) * (1.0 - 0.55 * avoid)))
    _set_if_present(out, cols.info_gain, _clip01(_numeric_col(out, cols.info_gain) * (1.0 - 0.45 * avoid)))
    _set_if_present(out, cols.target_reach_reduction, _clip01(_numeric_col(out, cols.target_reach_reduction) * (1.0 - 0.20 * avoid)))
    return _recompute_true_utility(out, cfg, weights={"risk": 0.60, "info": 0.10, "delay": 0.15, "decoy": 0.15, "cost": 0.05})


def _run_robustness_suite(df: pd.DataFrame, cfg: dict[str, Any], base_summary: pd.DataFrame) -> pd.DataFrame:
    robust_cfg = cfg.get("paper_experiments", {}).get("robustness", {})
    base_proposed = _policy_row(base_summary, "proposed_value_policy")
    rows: list[dict[str, Any]] = []

    scenarios: list[tuple[str, str, dict[str, Any], pd.DataFrame]] = []
    for rate in robust_cfg.get("observation_missing_rates", [0.0, 0.1, 0.3, 0.5]):
        scenarios.append((f"obs_missing_{int(float(rate) * 100)}pct", "observation_missing", {"missing_rate": float(rate)}, _apply_observation_missing(df, cfg, float(rate))))
    for sigma in robust_cfg.get("noise_sigmas", [0.05]):
        scenarios.append((f"noise_sigma_{float(sigma):.2f}", "noise_injection", {"sigma": float(sigma)}, _apply_noise(df, cfg, float(sigma), int(float(sigma) * 10_000))))
    for lag in robust_cfg.get("delayed_observation_lags", [1]):
        scenarios.append((f"delayed_obs_lag_{int(lag)}", "delayed_observation", {"lag": int(lag)}, _apply_delayed_observation(df, cfg, int(lag))))
    for severity in robust_cfg.get("wrong_prior_severities", [0.35]):
        scenarios.append((f"wrong_prior_{int(float(severity) * 100)}pct", "wrong_prior_belief", {"severity": float(severity)}, _apply_wrong_prior(df, cfg, float(severity))))
    for severity in robust_cfg.get("unseen_attacker_severities", [0.50]):
        scenarios.append((f"unseen_attacker_{int(float(severity) * 100)}pct", "unseen_attacker_type", {"severity": float(severity)}, _apply_unseen_attacker(df, cfg, float(severity))))
    for ratio in robust_cfg.get("decoy_aware_ratios", [0.25, 0.50, 0.75]):
        scenarios.append((f"decoy_aware_{int(float(ratio) * 100)}pct", "decoy_aware_ratio", {"ratio": float(ratio)}, _apply_decoy_aware_shift(df, cfg, float(ratio))))

    for name, stress_type, params, scenario_df in scenarios:
        summary, _ = _evaluate_variant(scenario_df, cfg, name, "robustness")
        proposed = _policy_row(summary, "proposed_value_policy")
        static = _policy_row(summary, "static_low_cost")
        phase3 = _policy_row(summary, "phase3_fused")
        risk_only = _policy_row(summary, "pred_risk_only")
        rows.append(
            {
                "scenario": name,
                "stress_type": stress_type,
                **params,
                "num_policies": int(summary["policy"].nunique()) if len(summary) else 0,
                "proposed_mean_utility": proposed.get("mean_utility"),
                "proposed_target_reach_reduction": proposed.get("mean_target_reach_reduction"),
                "proposed_info_gain": proposed.get("mean_info_gain"),
                "proposed_oracle_gap": proposed.get("mean_utility_gap_to_oracle"),
                "static_mean_utility": static.get("mean_utility"),
                "phase3_fused_mean_utility": phase3.get("mean_utility"),
                "pred_risk_only_mean_utility": risk_only.get("mean_utility"),
                "proposed_minus_static_utility": (proposed.get("mean_utility", np.nan) - static.get("mean_utility", np.nan)) if proposed and static else np.nan,
                "proposed_minus_phase3_utility": (proposed.get("mean_utility", np.nan) - phase3.get("mean_utility", np.nan)) if proposed and phase3 else np.nan,
                "proposed_minus_pred_risk_only_utility": (proposed.get("mean_utility", np.nan) - risk_only.get("mean_utility", np.nan)) if proposed and risk_only else np.nan,
                **_delta_against(base_proposed, proposed, prefix="delta_from_clean_proposed"),
            }
        )
    return pd.DataFrame(rows)



def _norm_group(df: pd.DataFrame, group_col: str, col: Optional[str], default: float = 0.5) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    v = _numeric_col(df, col, default)
    lo = v.groupby(df[group_col]).transform("min")
    hi = v.groupby(df[group_col]).transform("max")
    denom = (hi - lo).replace(0.0, np.nan)
    return ((v - lo) / denom).fillna(default).clip(0.0, 1.0)


def _action_type_signal(df: pd.DataFrame, kind: str) -> pd.Series:
    if "action_type" not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    t = df["action_type"].astype(str)
    if kind == "info":
        return t.isin(["telemetry_trap", "credential_breadcrumb", "place_low_exposure_decoy"]).astype(float)
    if kind == "delay":
        return t.isin(["path_perturbation", "rotate_or_retain_decoy"]).astype(float)
    if kind == "risk":
        return t.isin(["place_high_interaction_decoy", "credential_breadcrumb"]).astype(float)
    if kind == "stackelberg":
        # Interventions that should be preferred when attacker response/decoy-awareness matters.
        return t.isin(["telemetry_trap", "place_low_exposure_decoy", "path_perturbation"]).astype(float)
    return pd.Series(0.0, index=df.index, dtype=float)


def _write_ablation_policy_score(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    cols: Any,
    weights: dict[str, float],
    *,
    zero_info: bool = False,
    zero_delay: bool = False,
    zero_belief: bool = False,
    fused_only: bool = False,
) -> None:
    """Overwrite the resolved policy-score column with an ablated deployable score.

    The score is built from predicted heads and pre-action proxy/GNN signals only.
    It never uses test outcomes.  This makes Phase 6 ablations closer to a real
    component removal than the earlier post-hoc metric perturbation.
    """
    score_col = cols.pred_policy_score or "pred_policy_score"
    if score_col not in df.columns:
        df[score_col] = 0.0
    g = cols.graph_id
    if fused_only:
        df[score_col] = _norm_group(df, g, cols.fused_score or cols.risk_proxy or cols.pred_utility, 0.5)
        return

    risk = _norm_group(df, g, cols.pred_risk or cols.risk_proxy, 0.5)
    info = _norm_group(df, g, cols.pred_info or cols.info_proxy, 0.5)
    delay = _norm_group(df, g, cols.pred_delay or cols.delay_proxy, 0.5)
    utility = _norm_group(df, g, cols.pred_utility, 0.5)
    best = _numeric_col(df, cols.pred_best_prob, 0.0).clip(0.0, 1.0)
    fused = _norm_group(df, g, cols.fused_score, 0.5)
    risk_proxy = _norm_group(df, g, cols.risk_proxy, 0.5)
    info_proxy = _norm_group(df, g, cols.info_proxy, 0.5)
    delay_proxy = _norm_group(df, g, cols.delay_proxy, 0.5)
    entropy = _numeric_col(df, cols.belief_entropy, 0.5).clip(0.0, 1.0) if cols.belief_entropy else pd.Series(0.5, index=df.index)

    if zero_info:
        info = pd.Series(0.0, index=df.index)
        info_proxy = pd.Series(0.0, index=df.index)
    if zero_delay:
        delay = pd.Series(0.0, index=df.index)
        delay_proxy = pd.Series(0.0, index=df.index)
    if zero_belief:
        entropy = pd.Series(1.0, index=df.index)
        # Without belief, information-oriented traps are less confidently targeted.
        info = 0.45 * info
        info_proxy = 0.45 * info_proxy

    action_info = _action_type_signal(df, "info")
    action_delay = _action_type_signal(df, "delay")
    action_risk = _action_type_signal(df, "risk")
    belief_bonus = (1.0 - entropy) * (0.55 * action_info + 0.30 * action_delay + 0.15 * action_risk)
    if zero_belief:
        belief_bonus = pd.Series(0.0, index=df.index)

    score = (
        float(weights.get("utility", 0.20)) * utility
        + float(weights.get("risk", 0.22)) * risk
        + float(weights.get("info", 0.22)) * info
        + float(weights.get("delay", 0.18)) * delay
        + float(weights.get("best", 0.05)) * best
        + float(weights.get("fused", 0.05)) * fused
        + float(weights.get("risk_proxy", 0.03)) * risk_proxy
        + float(weights.get("info_proxy", 0.03)) * info_proxy
        + float(weights.get("delay_proxy", 0.02)) * delay_proxy
        + float(weights.get("belief_bonus", 0.10)) * belief_bonus
    )
    df[score_col] = score.clip(0.0, 1.0)


def _apply_ablation(df: pd.DataFrame, cfg: dict[str, Any], name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    local_update: dict[str, Any] = {}
    cols = resolve_schema(out, cfg)
    base_w = cfg.get("paper_experiments", {}).get("ablations", {}).get(
        "component_score_weights",
        {"utility": 0.20, "risk": 0.22, "info": 0.22, "delay": 0.18, "best": 0.05, "fused": 0.05, "risk_proxy": 0.03, "info_proxy": 0.03, "delay_proxy": 0.02, "belief_bonus": 0.10},
    )

    if name == "wo_belief":
        if cols.belief_entropy:
            out[cols.belief_entropy] = 1.0
        _write_ablation_policy_score(out, cfg, cols, base_w, zero_belief=True)
    elif name in {"wo_revelation_head", "wo_information_gain"}:
        _set_if_present(out, cols.pred_info, 0.0)
        _set_if_present(out, cols.info_proxy, 0.0)
        _set_if_present(out, cols.revelation_score, 0.0)
        _write_ablation_policy_score(out, cfg, cols, base_w, zero_info=True)
    elif name == "wo_delay_objective":
        _set_if_present(out, cols.pred_delay, 0.0)
        _set_if_present(out, cols.delay_proxy, 0.0)
        _write_ablation_policy_score(out, cfg, cols, base_w, zero_delay=True)
    elif name == "wo_amc":
        # Remove AMC/value heads; fall back to Phase-3 fused GNN score only.
        _write_ablation_policy_score(out, cfg, cols, base_w, fused_only=True)
    elif name == "wo_stackelberg_response":
        # Ignore action-type-aware response; approximate a myopic risk/info selector.
        myopic = dict(base_w)
        myopic.update({"utility": 0.10, "risk": 0.42, "info": 0.12, "delay": 0.08, "fused": 0.18, "belief_bonus": 0.0})
        _write_ablation_policy_score(out, cfg, cols, myopic)
    elif name == "wo_churn_constraint":
        _write_ablation_policy_score(out, cfg, cols, base_w)
        local_update = {"closed_loop": {"churn_penalty_weight": 0.0, "keep_previous_bonus": 0.0, "max_policy_churn_per_step": 999_999}}
    elif name == "wo_action_conditioned_snapshot":
        if cols.pred_info:
            out[cols.pred_info] = out.groupby(cols.episode_id)[cols.pred_info].transform("mean")
        if cols.pred_delay:
            out[cols.pred_delay] = out.groupby(cols.episode_id)[cols.pred_delay].transform("mean")
        if cols.fused_score:
            out[cols.fused_score] = 0.70 * _norm_group(out, cols.graph_id, cols.fused_score, 0.5) + 0.30 * _norm_group(out, cols.graph_id, cols.risk_proxy or cols.pred_risk, 0.5)
        _write_ablation_policy_score(out, cfg, cols, base_w, zero_belief=True)
        local_update = {"closed_loop": {"keep_previous_bonus": 0.0}}
    else:
        raise ValueError(f"Unknown ablation: {name}")

    return out, local_update

def _run_ablation_suite(df: pd.DataFrame, cfg: dict[str, Any], base_summary: pd.DataFrame) -> pd.DataFrame:
    ablation_names = cfg.get("paper_experiments", {}).get("ablations", {}).get(
        "include",
        [
            "wo_belief",
            "wo_revelation_head",
            "wo_amc",
            "wo_stackelberg_response",
            "wo_churn_constraint",
            "wo_action_conditioned_snapshot",
        ],
    )
    base_proposed = _policy_row(base_summary, "proposed_value_policy")
    rows: list[dict[str, Any]] = []
    for name in ablation_names:
        ablated_df, local_update = _apply_ablation(df, cfg, name)
        summary, _ = _evaluate_variant(ablated_df, cfg, name, "ablation", local_update)
        proposed = _policy_row(summary, "proposed_value_policy")
        rows.append(
            {
                "ablation": name,
                "policy_evaluated": "proposed_value_policy",
                "mean_utility": proposed.get("mean_utility"),
                "mean_target_reach_reduction": proposed.get("mean_target_reach_reduction"),
                "mean_info_gain": proposed.get("mean_info_gain"),
                "mean_delay_gain": proposed.get("mean_delay_gain"),
                "mean_churn": proposed.get("mean_churn"),
                "mean_utility_gap_to_oracle": proposed.get("mean_utility_gap_to_oracle"),
                "utility_drop_vs_full": (base_proposed.get("mean_utility", np.nan) - proposed.get("mean_utility", np.nan)) if base_proposed and proposed else np.nan,
                "target_reduction_drop_vs_full": (base_proposed.get("mean_target_reach_reduction", np.nan) - proposed.get("mean_target_reach_reduction", np.nan)) if base_proposed and proposed else np.nan,
                "info_gain_drop_vs_full": (base_proposed.get("mean_info_gain", np.nan) - proposed.get("mean_info_gain", np.nan)) if base_proposed and proposed else np.nan,
                "oracle_gap_increase_vs_full": (proposed.get("mean_utility_gap_to_oracle", np.nan) - base_proposed.get("mean_utility_gap_to_oracle", np.nan)) if base_proposed and proposed else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _apply_heldout_scenario(df: pd.DataFrame, cfg: dict[str, Any], name: str, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    cols = resolve_schema(out, cfg)
    if name == "different_transition_params":
        risk_multiplier = float(params.get("risk_multiplier", 0.85))
        info_multiplier = float(params.get("info_multiplier", 0.70))
        delay_multiplier = float(params.get("delay_multiplier", 1.10))
        decoy_multiplier = float(params.get("decoy_multiplier", 0.80))
        _set_if_present(out, cols.target_reach_reduction, _clip01(_numeric_col(out, cols.target_reach_reduction) * risk_multiplier))
        _set_if_present(out, cols.info_gain, _clip01(_numeric_col(out, cols.info_gain) * info_multiplier))
        _set_if_present(out, cols.delay_gain, _clip01(_numeric_col(out, cols.delay_gain) * delay_multiplier))
        _set_if_present(out, cols.decoy_absorption, _clip01(_numeric_col(out, cols.decoy_absorption) * decoy_multiplier))
        return _recompute_true_utility(out, cfg, weights={"risk": 0.62, "info": 0.10, "delay": 0.18, "decoy": 0.15, "cost": 0.05})

    if name == "caldera_inspired_traces":
        # CALDERA-style evaluation emphasizes lateral path interruption and delayed progress.
        _set_if_present(out, cols.target_reach_reduction, _clip01(_numeric_col(out, cols.target_reach_reduction) * 0.90 + _numeric_col(out, cols.delay_gain) * 0.10))
        _set_if_present(out, cols.info_gain, _clip01(_numeric_col(out, cols.info_gain) * 0.75))
        _set_if_present(out, cols.delay_gain, _clip01(_numeric_col(out, cols.delay_gain) * 1.25))
        return _recompute_true_utility(out, cfg, weights={"risk": 0.50, "info": 0.08, "delay": 0.30, "decoy": 0.15, "cost": 0.05})

    if name == "rule_family_heldout_split":
        # Use a stable hash family to emulate held-out rule families without requiring a
        # separate trace file. If a future `rule_family` column exists, it can be used upstream.
        family = out[cols.action_id].astype(str).map(lambda x: stable_hash_int(x, 3))
        _set_if_present(out, cols.target_reach_reduction, _clip01(_numeric_col(out, cols.target_reach_reduction) * np.where(family == 0, 0.75, 1.05)))
        _set_if_present(out, cols.info_gain, _clip01(_numeric_col(out, cols.info_gain) * np.where(family == 1, 0.60, 1.00)))
        _set_if_present(out, cols.decoy_absorption, _clip01(_numeric_col(out, cols.decoy_absorption) * np.where(family == 2, 0.70, 1.00)))
        return _recompute_true_utility(out, cfg, weights={"risk": 0.58, "info": 0.12, "delay": 0.15, "decoy": 0.20, "cost": 0.05})

    raise ValueError(f"Unknown held-out scenario: {name}")


def _run_heldout_attacker_suite(df: pd.DataFrame, cfg: dict[str, Any], clean_summary: pd.DataFrame) -> pd.DataFrame:
    circ_cfg = cfg.get("paper_experiments", {}).get("circularity", {})
    scenarios = circ_cfg.get(
        "heldout_scenarios",
        [
            {"name": "different_transition_params"},
            {"name": "caldera_inspired_traces"},
            {"name": "rule_family_heldout_split"},
        ],
    )
    clean_proposed = _policy_row(clean_summary, "proposed_value_policy")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        name = str(scenario.get("name"))
        params = {k: v for k, v in scenario.items() if k != "name"}
        heldout_df = _apply_heldout_scenario(df, cfg, name, params)
        summary, _ = _evaluate_variant(heldout_df, cfg, name, "heldout_attacker")
        proposed = _policy_row(summary, "proposed_value_policy")
        oracle = _policy_row(summary, "amc_oracle")
        static = _policy_row(summary, "static_low_cost")
        rows.append(
            {
                "heldout_scenario": name,
                "evaluation_oracle": "heldout_dynamics_not_training_amc_label",
                **params,
                "proposed_mean_utility": proposed.get("mean_utility"),
                "proposed_target_reach_reduction": proposed.get("mean_target_reach_reduction"),
                "proposed_info_gain": proposed.get("mean_info_gain"),
                "proposed_oracle_gap": proposed.get("mean_utility_gap_to_oracle"),
                "heldout_oracle_mean_utility": oracle.get("mean_utility"),
                "static_mean_utility": static.get("mean_utility"),
                "proposed_minus_static_utility": (proposed.get("mean_utility", np.nan) - static.get("mean_utility", np.nan)) if proposed and static else np.nan,
                **_delta_against(clean_proposed, proposed, prefix="delta_from_clean_proposed"),
            }
        )
    return pd.DataFrame(rows)


def _assign_scale_bucket(nodes: float, thresholds: dict[str, Any]) -> str:
    if nodes >= float(thresholds.get("large_min_nodes", 1000)):
        return "large"
    if nodes >= float(thresholds.get("medium_min_nodes", 100)):
        return "medium"
    return "small"


def _load_graph_stats(cfg: dict[str, Any]) -> pd.DataFrame:
    path = cfg.get("paths", {}).get("phase3_graph_stats_csv")
    stats = _read_optional_csv(path)
    if len(stats) == 0:
        return stats
    if "graph_id" not in stats.columns:
        return pd.DataFrame()
    return stats.drop_duplicates("graph_id")


def _make_scale_proxy_rows(stats: pd.DataFrame, scale_cfg: dict[str, Any], thresholds: dict[str, Any]) -> pd.DataFrame:
    if len(stats) == 0 or not bool(scale_cfg.get("enable_proxy_when_missing", True)):
        return pd.DataFrame()
    existing = set(stats["scale_bucket"].dropna().astype(str)) if "scale_bucket" in stats.columns else set()
    proxies: list[pd.DataFrame] = []
    representative = scale_cfg.get("proxy_representative_nodes", {"medium": 200, "large": 1000})
    for bucket in ["medium", "large"]:
        if bucket in existing:
            continue
        target_nodes = float(representative.get(bucket, 200 if bucket == "medium" else 1000))
        base = stats.copy()
        base_nodes = pd.to_numeric(base["original_num_nodes"], errors="coerce").replace(0, np.nan).fillna(20.0)
        factor = target_nodes / base_nodes
        base["graph_id"] = base["graph_id"].astype(str) + f"__scale_proxy_{bucket}"
        base["scale_bucket"] = bucket
        base["scale_source"] = "proxy_scaled_from_existing_action_distribution"
        base["original_num_nodes"] = target_nodes
        if "original_num_edges_undirected" in base.columns:
            base["original_num_edges_undirected"] = pd.to_numeric(base["original_num_edges_undirected"], errors="coerce").fillna(0.0) * factor * np.sqrt(factor)
        if "strategic_num_nodes" in base.columns:
            compression_target = float(scale_cfg.get("proxy_compression_ratio", {}).get(bucket, 0.18 if bucket == "large" else 0.35))
            base["strategic_num_nodes"] = np.maximum(5.0, target_nodes * compression_target)
            base["compression_ratio_nodes"] = compression_target
        proxies.append(base)
    return pd.concat(proxies, ignore_index=True) if proxies else pd.DataFrame()


def _run_scale_experiment(base_rollout: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    scale_cfg = cfg.get("paper_experiments", {}).get("realistic_ad_graph", {})
    thresholds = scale_cfg.get("scale_thresholds", {})
    stats = _load_graph_stats(cfg)

    if len(stats) == 0:
        # Fallback: infer from rollout only. This keeps the suite runnable even if Phase 3
        # graph stats were not copied into the workspace.
        stats = base_rollout[["graph_id"]].drop_duplicates().copy()
        stats["original_num_nodes"] = float(scale_cfg.get("fallback_node_count", 20))
        stats["original_num_edges_undirected"] = np.nan
        stats["strategic_num_nodes"] = np.nan
        stats["compression_ratio_nodes"] = np.nan

    if "original_num_nodes" not in stats.columns:
        stats["original_num_nodes"] = float(scale_cfg.get("fallback_node_count", 20))
    if "compression_ratio_nodes" not in stats.columns and {"strategic_num_nodes", "original_num_nodes"}.issubset(stats.columns):
        stats["compression_ratio_nodes"] = pd.to_numeric(stats["strategic_num_nodes"], errors="coerce") / pd.to_numeric(stats["original_num_nodes"], errors="coerce").replace(0, np.nan)

    stats = stats.copy()
    stats["scale_bucket"] = pd.to_numeric(stats["original_num_nodes"], errors="coerce").fillna(20.0).map(lambda n: _assign_scale_bucket(float(n), thresholds))
    stats["scale_source"] = "actual_graph_stats"
    proxy_stats = _make_scale_proxy_rows(stats, scale_cfg, thresholds)
    stats_all = pd.concat([stats, proxy_stats], ignore_index=True) if len(proxy_stats) else stats

    merged = base_rollout.merge(stats_all, on="graph_id", how="left")
    if len(proxy_stats):
        # Proxy rows do not have their own rollout decisions, so attach base rollout
        # distributions to proxy graph stats by index. This is clearly marked by scale_source.
        base_sample = base_rollout.copy().reset_index(drop=True)
        repeated: list[pd.DataFrame] = []
        graph_stat_cols = [
            "graph_id",
            "original_num_nodes",
            "original_num_edges_undirected",
            "strategic_num_nodes",
            "strategic_num_edges_directed",
            "compression_ratio_nodes",
            "num_candidate_actions",
            "scale_bucket",
            "scale_source",
        ]
        for bucket in proxy_stats["scale_bucket"].dropna().unique():
            proxy_bucket = proxy_stats[proxy_stats["scale_bucket"] == bucket].reset_index(drop=True)
            take = base_sample.iloc[np.arange(len(proxy_bucket)) % len(base_sample)].copy().reset_index(drop=True)
            enriched = take.copy()
            # Replace only graph-scale metadata; avoid duplicate column labels when
            # concatenating actual and proxy scale rows.
            for col in graph_stat_cols:
                if col in proxy_bucket.columns:
                    enriched[col] = proxy_bucket[col].to_numpy()
            repeated.append(enriched)
        if repeated:
            merged = pd.concat([merged, *repeated], ignore_index=True, sort=False)

    runtime_cfg = scale_cfg.get("runtime_memory_model", {})
    nodes = pd.to_numeric(merged.get("original_num_nodes", 20), errors="coerce").fillna(20.0)
    edges = pd.to_numeric(merged.get("original_num_edges_undirected", 0), errors="coerce").fillna(0.0)
    strategic_nodes = pd.to_numeric(merged.get("strategic_num_nodes", nodes), errors="coerce").fillna(nodes)
    merged["estimated_runtime_ms"] = (
        float(runtime_cfg.get("base_ms", 2.0))
        + float(runtime_cfg.get("node_ms", 0.018)) * nodes
        + float(runtime_cfg.get("edge_ms", 0.006)) * edges
        + float(runtime_cfg.get("strategic_node_ms", 0.050)) * strategic_nodes
    )
    merged["estimated_memory_mb"] = (
        float(runtime_cfg.get("base_mb", 96.0))
        + float(runtime_cfg.get("node_mb", 0.012)) * nodes
        + float(runtime_cfg.get("edge_mb", 0.002)) * edges
        + float(runtime_cfg.get("strategic_node_mb", 0.020)) * strategic_nodes
    )

    agg = (
        merged.groupby(["scale_bucket", "scale_source", "policy"], dropna=False)
        .agg(
            graphs=("graph_id", "nunique"),
            mean_original_nodes=("original_num_nodes", "mean"),
            mean_original_edges=("original_num_edges_undirected", "mean"),
            mean_strategic_nodes=("strategic_num_nodes", "mean"),
            strategic_subgraph_compression_ratio=("compression_ratio_nodes", "mean"),
            estimated_runtime_ms=("estimated_runtime_ms", "mean"),
            estimated_memory_mb=("estimated_memory_mb", "mean"),
            oracle_gap=("utility_gap_to_oracle", "mean"),
            policy_utility=("utility_true", "mean"),
            target_reach_reduction=("target_reach_reduction", "mean"),
            info_gain=("info_gain", "mean"),
            delay_gain=("delay_gain", "mean"),
        )
        .reset_index()
    )
    bucket_order = {"small": 0, "medium": 1, "large": 2}
    agg["_scale_order"] = agg["scale_bucket"].map(bucket_order).fillna(99)
    return agg.sort_values(["_scale_order", "scale_source", "policy"]).drop(columns=["_scale_order"]).reset_index(drop=True)


def run_paper_experiment_suite(df: pd.DataFrame, cfg: dict[str, Any]) -> SuiteOutputs:
    """Run the four paper-facing experiment suites and return output tables."""
    working = _sample_graphs(df, cfg, label="clean_base")
    clean_summary, clean_rollout = _evaluate_variant(working, cfg, "clean_base", "base")

    scale_summary = _run_scale_experiment(clean_rollout, cfg)
    robustness_summary = _run_robustness_suite(working, cfg, clean_summary)
    ablation_summary = _run_ablation_suite(working, cfg, clean_summary)
    heldout_summary = _run_heldout_attacker_suite(working, cfg, clean_summary)

    proposed = _policy_row(clean_summary, "proposed_value_policy")
    static = _policy_row(clean_summary, "static_low_cost")
    oracle = _policy_row(clean_summary, "amc_oracle")
    manifest = {
        "suite": "phase6_paper_experiments",
        "num_clean_rows": int(len(working)),
        "num_clean_graphs": int(working[resolve_schema(working, cfg).graph_id].nunique()) if len(working) else 0,
        "clean_proposed_value_policy": proposed,
        "clean_static_low_cost": static,
        "clean_amc_oracle": oracle,
        "clean_baseline_comparison": make_baseline_comparison(clean_summary).to_dict(orient="records"),
        "outputs": {
            "scale_rows": int(len(scale_summary)),
            "robustness_rows": int(len(robustness_summary)),
            "ablation_rows": int(len(ablation_summary)),
            "heldout_attacker_rows": int(len(heldout_summary)),
        },
        "notes": [
            "Scale rows with scale_source=actual_graph_stats are measured from Phase 3 graph stats.",
            "Scale rows with scale_source=proxy_scaled_from_existing_action_distribution are stress proxies until public BloodHound-scale graphs are supplied.",
            "Held-out attacker rows recompute evaluation utility under perturbed dynamics, while keeping the learned value head fixed, to reduce circular AMC-label evaluation.",
        ],
    }
    return SuiteOutputs(scale_summary, robustness_summary, ablation_summary, heldout_summary, manifest)


def write_paper_experiment_outputs(outputs: SuiteOutputs, cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    frames = {
        "scale_experiment_csv": outputs.scale_summary,
        "robustness_summary_csv": outputs.robustness_summary,
        "ablation_summary_csv": outputs.ablation_summary,
        "heldout_attacker_summary_csv": outputs.heldout_attacker_summary,
    }
    for key, frame in frames.items():
        path = paths.get(key)
        if not path:
            continue
        out_path = resolve_path(path)
        ensure_parent(out_path)
        frame.to_csv(out_path, index=False)
        print(f"Wrote {key}: {out_path} ({len(frame):,} rows)")

    manifest_path = paths.get("paper_experiment_summary_json")
    if manifest_path:
        write_json(outputs.manifest, manifest_path)
        print(f"Wrote paper experiment summary: {resolve_path(manifest_path)}")
