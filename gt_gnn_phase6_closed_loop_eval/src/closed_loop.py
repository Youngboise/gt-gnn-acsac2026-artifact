from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .utils import first_existing_column, safe_float, stable_hash_int


@dataclass
class SchemaColumns:
    graph_id: str
    episode_id: str
    t: str
    action_id: str
    node_id: str
    split: Optional[str]
    baseline_target_prob: str
    action_target_prob: str
    target_reach_reduction: str
    decoy_absorption: str
    info_gain: str
    delay_gain: str
    utility_true: str
    cost: Optional[str]
    churn_proxy: Optional[str]
    is_best_action: Optional[str]
    pred_utility: Optional[str]
    pred_risk: Optional[str]
    pred_info: Optional[str]
    pred_delay: Optional[str]
    pred_best_prob: Optional[str]
    pred_rank_score: Optional[str]
    pred_policy_score: Optional[str]
    fused_score: Optional[str]
    threat_score: Optional[str]
    revelation_score: Optional[str]
    risk_proxy: Optional[str]
    info_proxy: Optional[str]
    delay_proxy: Optional[str]
    belief_entropy: Optional[str]
    centrality_score: Optional[str]


def resolve_schema(df: pd.DataFrame, cfg: dict[str, Any]) -> SchemaColumns:
    schema = cfg["schema"]
    ids = schema["id_columns"]
    outcomes = schema["outcome_columns"]
    scores = schema["score_columns"]

    graph_id = first_existing_column(df, ids["graph_id"], True, "graph_id")
    episode_id = first_existing_column(df, ids["episode_id"], False, "episode_id") or graph_id
    t = first_existing_column(df, ids["t"], False, "t") or graph_id
    action_id = first_existing_column(df, ids["action_id"], True, "action_id")
    node_id = first_existing_column(df, ids["node_id"], False, "node_id") or action_id
    split = first_existing_column(df, ids.get("split", []), False, "split")

    return SchemaColumns(
        graph_id=graph_id,
        episode_id=episode_id,
        t=t,
        action_id=action_id,
        node_id=node_id,
        split=split,
        baseline_target_prob=first_existing_column(df, outcomes["baseline_target_prob"], True, "baseline_target_prob"),
        action_target_prob=first_existing_column(df, outcomes["action_target_prob"], True, "action_target_prob"),
        target_reach_reduction=first_existing_column(df, outcomes["target_reach_reduction"], True, "target_reach_reduction"),
        decoy_absorption=first_existing_column(df, outcomes["decoy_absorption"], True, "decoy_absorption"),
        info_gain=first_existing_column(df, outcomes["info_gain"], True, "info_gain"),
        delay_gain=first_existing_column(df, outcomes["delay_gain"], True, "delay_gain"),
        utility_true=first_existing_column(df, outcomes["utility_true"], True, "utility_true"),
        cost=first_existing_column(df, outcomes["cost"], False, "operational_cost"),
        churn_proxy=first_existing_column(df, outcomes["churn_proxy"], False, "churn_proxy"),
        is_best_action=first_existing_column(df, outcomes["is_best_action"], False, "is_best_action"),
        pred_utility=first_existing_column(df, scores["pred_utility"], False, "pred_utility"),
        pred_risk=first_existing_column(df, scores["pred_risk"], False, "pred_risk"),
        pred_info=first_existing_column(df, scores["pred_info"], False, "pred_info"),
        pred_delay=first_existing_column(df, scores["pred_delay"], False, "pred_delay"),
        pred_best_prob=first_existing_column(df, scores["pred_best_prob"], False, "pred_best_prob"),
        pred_rank_score=first_existing_column(df, scores.get("pred_rank_score", []), False, "pred_rank_score"),
        pred_policy_score=first_existing_column(df, scores.get("pred_policy_score", []), False, "pred_policy_score"),
        fused_score=first_existing_column(df, scores["fused_score"], False, "fused_score"),
        threat_score=first_existing_column(df, scores["threat_score"], False, "threat_score"),
        revelation_score=first_existing_column(df, scores["revelation_score"], False, "revelation_score"),
        risk_proxy=first_existing_column(df, scores["risk_proxy"], False, "risk_proxy"),
        info_proxy=first_existing_column(df, scores["info_proxy"], False, "info_proxy"),
        delay_proxy=first_existing_column(df, scores["delay_proxy"], False, "delay_proxy"),
        belief_entropy=first_existing_column(df, scores["belief_entropy"], False, "belief_entropy"),
        centrality_score=first_existing_column(df, scores.get("centrality_score", []), False, "centrality_score"),
    )


def merge_phase4_phase5(phase4: pd.DataFrame, phase5: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Merge AMC outcomes and learned predictions.

    Phase 5 predictions often already contain Phase 4 outcome columns. If so, this function
    simply returns the Phase 5 dataframe. Otherwise, it merges by available identifiers.
    """
    outcome_candidates = cfg["schema"]["outcome_columns"]["utility_true"]
    pred_candidates = cfg["schema"]["score_columns"]["pred_utility"]

    has_outcome = any(c in phase5.columns for c in outcome_candidates)
    has_pred = any(c in phase5.columns for c in pred_candidates)
    if has_outcome and has_pred:
        return phase5.copy()

    id_cfg = cfg["schema"]["id_columns"]
    merge_keys = []
    for key in ["graph_id", "action_id", "node_id"]:
        col4 = first_existing_column(phase4, id_cfg[key], False, f"phase4 {key}")
        col5 = first_existing_column(phase5, id_cfg[key], False, f"phase5 {key}")
        if col4 and col5 and col4 == col5:
            merge_keys.append(col4)

    if len(merge_keys) < 2:
        raise ValueError(
            "Could not infer merge keys between Phase 4 outcomes and Phase 5 predictions. "
            f"Phase4 columns={list(phase4.columns)} Phase5 columns={list(phase5.columns)}"
        )

    merged = phase4.merge(phase5, on=merge_keys, how="left", suffixes=("", "_phase5"))
    return merged


def _numeric(row: pd.Series, col: Optional[str], default: float = 0.0) -> float:
    if col is None:
        return default
    return safe_float(row.get(col, default), default)


def _group_norm_value(group: pd.DataFrame, row: pd.Series, col: Optional[str], default: float = 0.0) -> float:
    if col is None or col not in group.columns:
        return float(default)
    values = pd.to_numeric(group[col], errors="coerce").fillna(default).astype(float)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return 0.5
    return float((safe_float(row.get(col, default), default) - lo) / (hi - lo))


def _proposed_component_score(group: pd.DataFrame, row: pd.Series, cols: SchemaColumns, params: dict[str, Any]) -> float:
    weights = params.get("proposed_score_weights", {})
    # Prefer the Phase 5 policy score when present; otherwise recompute component-aware score here.
    if cols.pred_policy_score is not None:
        policy_score = _group_norm_value(group, row, cols.pred_policy_score, 0.5)
    else:
        policy_score = 0.0
    risk = _group_norm_value(group, row, cols.pred_risk, 0.5)
    info = _group_norm_value(group, row, cols.pred_info, 0.5)
    delay = _group_norm_value(group, row, cols.pred_delay, 0.5)
    rank = _group_norm_value(group, row, cols.pred_rank_score, policy_score)
    best = _numeric(row, cols.pred_best_prob, 0.0)
    if cols.pred_policy_score is not None:
        # Blend validation-calibrated Phase 5 score with decomposed heads.
        # A calibrated score is tuned on validation groups only; the decomposed
        # fallback prevents one noisy head from dominating deployment decisions.
        decomposed = (
            float(weights.get("risk", 0.34)) * risk
            + float(weights.get("info", 0.29)) * info
            + float(weights.get("delay", 0.20)) * delay
            + float(weights.get("rank_score", 0.12)) * rank
            + float(weights.get("best_prob", 0.05)) * best
        )
        alpha = float(params.get("calibrated_score_weight", 0.72))
        alpha = max(0.0, min(1.0, alpha))
        return alpha * policy_score + (1.0 - alpha) * decomposed
    return (
        float(weights.get("risk", 0.34)) * risk
        + float(weights.get("info", 0.29)) * info
        + float(weights.get("delay", 0.20)) * delay
        + float(weights.get("rank_score", 0.12)) * rank
        + float(weights.get("best_prob", 0.05)) * best
    )


def _policy_score(
    group: pd.DataFrame,
    row: pd.Series,
    policy: str,
    cols: SchemaColumns,
    prev_node: Optional[str],
    params: dict[str, Any],
) -> float:
    cost_penalty = float(params.get("cost_penalty_weight", 0.0))
    churn_penalty = float(params.get("churn_penalty_weight", 0.0))
    keep_bonus = float(params.get("keep_previous_bonus", 0.0))

    cost = _numeric(row, cols.cost, 0.0)
    node = str(row.get(cols.node_id, row.get(cols.action_id)))
    changed = 1.0 if prev_node is not None and node != prev_node else 0.0
    keep = 1.0 if prev_node is not None and node == prev_node else 0.0
    constraint_adjust = -cost_penalty * cost - churn_penalty * changed + keep_bonus * keep

    if policy == "proposed_value_policy":
        return _proposed_component_score(group, row, cols, params) + constraint_adjust
    if policy == "amc_oracle":
        return _numeric(row, cols.utility_true, 0.0) + constraint_adjust

    # Deployable baselines: predicted heads or pre-action proxy scores.
    if policy == "pred_risk_only":
        return _group_norm_value(group, row, cols.pred_risk, _numeric(row, cols.risk_proxy, 0.0)) + constraint_adjust
    if policy == "pred_info_only":
        return _group_norm_value(group, row, cols.pred_info, _numeric(row, cols.info_proxy, 0.0)) + constraint_adjust
    if policy == "pred_delay_only":
        return _group_norm_value(group, row, cols.pred_delay, _numeric(row, cols.delay_proxy, 0.0)) + constraint_adjust
    if policy == "proxy_risk_only":
        return _numeric(row, cols.risk_proxy, 0.0) + constraint_adjust
    if policy == "proxy_info_only":
        return _numeric(row, cols.info_proxy, 0.0) + constraint_adjust
    if policy == "proxy_delay_only":
        return _numeric(row, cols.delay_proxy, 0.0) + constraint_adjust

    # Oracle component upper bounds: not deployable, useful only as ceilings.
    if policy in {"true_risk_only", "risk_only"}:
        return _numeric(row, cols.target_reach_reduction, 0.0) + constraint_adjust
    if policy in {"true_info_only", "info_only"}:
        return _numeric(row, cols.info_gain, 0.0) + constraint_adjust
    if policy in {"true_delay_only", "delay_only"}:
        return _numeric(row, cols.delay_gain, 0.0) + constraint_adjust

    if policy == "random_placement":
        key = f"{row.get(cols.graph_id)}::{row.get(cols.action_id)}::{row.get(cols.node_id)}"
        return (stable_hash_int(key, 1_000_000) / 1_000_000.0) + constraint_adjust
    if policy == "centrality_based":
        return _numeric(row, cols.centrality_score, _numeric(row, cols.delay_proxy, _numeric(row, cols.fused_score, 0.0))) + constraint_adjust
    if policy == "phase3_fused":
        base = _numeric(row, cols.fused_score, _numeric(row, cols.threat_score, 0.0))
        return base + constraint_adjust
    if policy == "static_low_cost":
        phase3 = _numeric(row, cols.fused_score, 0.0)
        return 0.10 * phase3 - cost - 0.50 * changed + 0.10 * keep
    raise ValueError(f"Unknown policy: {policy}")


def _filter_feasible(group: pd.DataFrame, cols: SchemaColumns, params: dict[str, Any]) -> pd.DataFrame:
    if not params.get("use_action_cost_constraints", True) or cols.cost is None:
        return group
    max_cost = float(params.get("max_action_cost", 1.0))
    feasible = group[pd.to_numeric(group[cols.cost], errors="coerce").fillna(0.0) <= max_cost].copy()
    if len(feasible) == 0:
        return group
    return feasible


def _select_row(
    group: pd.DataFrame,
    policy: str,
    cols: SchemaColumns,
    prev_node: Optional[str],
    params: dict[str, Any],
) -> pd.Series | None:
    if policy == "no_deception":
        return None
    feasible = _filter_feasible(group, cols, params)
    scores = feasible.apply(lambda r: _policy_score(feasible, r, policy, cols, prev_node, params), axis=1)
    return feasible.loc[scores.idxmax()]


def _oracle_row(group: pd.DataFrame, cols: SchemaColumns) -> pd.Series:
    return group.loc[pd.to_numeric(group[cols.utility_true], errors="coerce").fillna(-1e9).idxmax()]


def run_closed_loop(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = resolve_schema(df, cfg)
    policies = cfg["policies"]["include"]
    params = cfg["closed_loop"]
    entropy_cfg = params.get("entropy_update", {})

    working = df.copy()
    for c in [
        cols.baseline_target_prob,
        cols.action_target_prob,
        cols.target_reach_reduction,
        cols.decoy_absorption,
        cols.info_gain,
        cols.delay_gain,
        cols.utility_true,
    ]:
        working[c] = pd.to_numeric(working[c], errors="coerce").fillna(0.0)
    if cols.cost:
        working[cols.cost] = pd.to_numeric(working[cols.cost], errors="coerce").fillna(0.0)

    if cols.split is None:
        working["_split"] = "all"
        cols.split = "_split"

    eval_split = params.get("evaluation_split", None)
    if eval_split not in (None, "", "all"):
        before = len(working)
        working = working[working[cols.split].astype(str) == str(eval_split)].copy()
        if working.empty:
            raise ValueError(f"No rows left after evaluation_split={eval_split!r}; before={before}")

    if cols.belief_entropy is None:
        # Use a stable synthetic entropy prior if not carried from earlier phases.
        working["_belief_entropy"] = 1.0
        cols.belief_entropy = "_belief_entropy"

    # Sort snapshots by episode and time. If t is non-numeric, stable lexical order is used.
    sort_cols = [cols.episode_id, cols.t, cols.graph_id]
    working = working.sort_values(sort_cols).reset_index(drop=True)

    rollout_rows: list[dict[str, Any]] = []
    recalibration_rows: list[dict[str, Any]] = []

    for policy in policies:
        prev_node_by_episode: dict[str, Optional[str]] = {}
        entropy_by_episode: dict[str, float] = {}

        for graph_id, group in working.groupby(cols.graph_id, sort=False):
            first = group.iloc[0]
            episode = str(first[cols.episode_id])
            t_val = first[cols.t]
            split = first[cols.split]
            prev_node = prev_node_by_episode.get(episode)

            if episode not in entropy_by_episode:
                entropy_by_episode[episode] = safe_float(first.get(cols.belief_entropy, 1.0), 1.0)
            entropy_before = entropy_by_episode[episode]

            oracle = _oracle_row(group, cols)
            selected = _select_row(group, policy, cols, prev_node, params)

            if selected is None:
                selected_action_id = "NO_DECEPTION"
                selected_node = "NO_DECEPTION"
                baseline_target = safe_float(first[cols.baseline_target_prob], 0.0)
                action_target = baseline_target
                risk_reduction = 0.0
                decoy_absorption = 0.0
                info_gain = 0.0
                delay_gain = 0.0
                utility_true = 0.0
                cost = 0.0
                pred_utility = np.nan
                pred_risk = np.nan
                pred_info = np.nan
                pred_delay = np.nan
                selected_action_type = "none"
            else:
                selected_action_id = str(selected[cols.action_id])
                selected_node = str(selected[cols.node_id])
                baseline_target = safe_float(selected[cols.baseline_target_prob], 0.0)
                action_target = safe_float(selected[cols.action_target_prob], 0.0)
                risk_reduction = safe_float(selected[cols.target_reach_reduction], 0.0)
                decoy_absorption = safe_float(selected[cols.decoy_absorption], 0.0)
                info_gain = safe_float(selected[cols.info_gain], 0.0)
                delay_gain = safe_float(selected[cols.delay_gain], 0.0)
                utility_true = safe_float(selected[cols.utility_true], 0.0)
                cost = _numeric(selected, cols.cost, 0.0)
                pred_utility = _numeric(selected, cols.pred_utility, np.nan)
                pred_risk = _numeric(selected, cols.pred_risk, np.nan)
                pred_info = _numeric(selected, cols.pred_info, np.nan)
                pred_delay = _numeric(selected, cols.pred_delay, np.nan)
                selected_action_type = str(selected.get("action_type", "unknown"))

            oracle_action = str(oracle[cols.action_id])
            oracle_node = str(oracle[cols.node_id])
            oracle_utility = safe_float(oracle[cols.utility_true], 0.0)
            oracle_target_reduction = safe_float(oracle[cols.target_reach_reduction], 0.0)

            changed = 0 if prev_node is None or selected_node == "NO_DECEPTION" else int(selected_node != prev_node)
            if selected_node != "NO_DECEPTION":
                prev_node_by_episode[episode] = selected_node

            # Closed-loop observation model: estimate whether the selected intervention generated
            # a post-action signal, then update posterior entropy. This is still simulated, but it
            # is event-conditioned rather than a fixed entropy decrement.
            triggered_decoy_estimate = max(0.0, min(1.0, 0.60 * decoy_absorption + 0.40 * info_gain))
            if selected_action_type in {"telemetry_trap", "credential_breadcrumb", "place_low_exposure_decoy"}:
                triggered_decoy_estimate = max(triggered_decoy_estimate, min(1.0, info_gain + 0.15))
            post_action_event_strength = max(0.0, min(1.0, info_gain + 0.5 * triggered_decoy_estimate + 0.15 * decoy_absorption))
            entropy_reduction = max(
                0.0,
                float(entropy_cfg.get("info_gain_scale", 1.0)) * info_gain
                + float(entropy_cfg.get("decoy_absorption_scale", 0.15)) * decoy_absorption
                + float(entropy_cfg.get("trigger_scale", 0.25)) * triggered_decoy_estimate,
            )
            entropy_after = max(float(entropy_cfg.get("min_entropy", 0.0)), entropy_before - entropy_reduction)
            posterior_confidence_before = max(0.0, min(1.0, 1.0 - entropy_before))
            posterior_confidence_after = max(0.0, min(1.0, 1.0 - entropy_after))
            entropy_by_episode[episode] = entropy_after

            hit_oracle = int(selected_action_id == oracle_action)
            utility_gap = max(0.0, oracle_utility - utility_true)

            rollout_rows.append(
                {
                    "policy": policy,
                    "graph_id": graph_id,
                    "episode_id": episode,
                    "t": t_val,
                    "split": split,
                    "selected_action_id": selected_action_id,
                    "selected_node_id": selected_node,
                    "selected_action_type": selected_action_type,
                    "oracle_action_id": oracle_action,
                    "oracle_node_id": oracle_node,
                    "baseline_target_prob": baseline_target,
                    "action_target_prob": action_target,
                    "target_reach_reduction": risk_reduction,
                    "decoy_absorption": decoy_absorption,
                    "info_gain": info_gain,
                    "delay_gain": delay_gain,
                    "utility_true": utility_true,
                    "oracle_utility": oracle_utility,
                    "oracle_target_reduction": oracle_target_reduction,
                    "utility_gap_to_oracle": utility_gap,
                    "policy_hit_oracle": hit_oracle,
                    "operational_cost": cost,
                    "churn": changed,
                    "belief_entropy_before": entropy_before,
                    "posterior_confidence_before": posterior_confidence_before,
                    "triggered_decoy_estimate": triggered_decoy_estimate,
                    "post_action_event_strength": post_action_event_strength,
                    "entropy_reduction": entropy_reduction,
                    "belief_entropy_after": entropy_after,
                    "posterior_confidence_after": posterior_confidence_after,
                    "pred_utility": pred_utility,
                    "pred_risk": pred_risk,
                    "pred_info": pred_info,
                    "pred_delay": pred_delay,
                }
            )

            if policy == "proposed_value_policy":
                recalibration_rows.append(
                    {
                        "graph_id": graph_id,
                        "episode_id": episode,
                        "t": t_val,
                        "selected_action_id": selected_action_id,
                        "selected_action_type": selected_action_type,
                        "triggered_decoy_estimate": triggered_decoy_estimate,
                        "post_action_event_strength": post_action_event_strength,
                        "belief_entropy_before": entropy_before,
                        "belief_entropy_after": entropy_after,
                        "posterior_confidence_before": posterior_confidence_before,
                        "posterior_confidence_after": posterior_confidence_after,
                        "utility_true": utility_true,
                        "pred_utility": pred_utility,
                        "utility_error_true_minus_pred": utility_true - pred_utility if pd.notna(pred_utility) else np.nan,
                        "risk_true": risk_reduction,
                        "pred_risk": pred_risk,
                        "risk_error_true_minus_pred": risk_reduction - pred_risk if pd.notna(pred_risk) else np.nan,
                        "info_true": info_gain,
                        "pred_info": pred_info,
                        "info_error_true_minus_pred": info_gain - pred_info if pd.notna(pred_info) else np.nan,
                        "delay_true": delay_gain,
                        "pred_delay": pred_delay,
                        "delay_error_true_minus_pred": delay_gain - pred_delay if pd.notna(pred_delay) else np.nan,
                    }
                )

    rollout = pd.DataFrame(rollout_rows)
    recalib = pd.DataFrame(recalibration_rows)
    episode_summary = summarize_by_episode(rollout)
    return rollout, episode_summary, recalib


def summarize_by_episode(rollout: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rollout.groupby(["policy", "episode_id"], dropna=False)
        .agg(
            snapshots=("graph_id", "count"),
            target_reach_reduction=("target_reach_reduction", "mean"),
            action_target_prob=("action_target_prob", "mean"),
            decoy_absorption=("decoy_absorption", "mean"),
            info_gain=("info_gain", "sum"),
            entropy_reduction=("entropy_reduction", "sum"),
            delay_gain=("delay_gain", "mean"),
            utility_true=("utility_true", "mean"),
            oracle_utility=("oracle_utility", "mean"),
            utility_gap_to_oracle=("utility_gap_to_oracle", "mean"),
            policy_hit_oracle=("policy_hit_oracle", "mean"),
            operational_cost=("operational_cost", "sum"),
            churn=("churn", "sum"),
        )
        .reset_index()
    )
    return agg


def summarize_by_policy(rollout: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rollout.groupby("policy", dropna=False)
        .agg(
            snapshots=("graph_id", "count"),
            episodes=("episode_id", "nunique"),
            mean_target_reach_reduction=("target_reach_reduction", "mean"),
            mean_action_target_prob=("action_target_prob", "mean"),
            mean_decoy_absorption=("decoy_absorption", "mean"),
            total_info_gain=("info_gain", "sum"),
            mean_info_gain=("info_gain", "mean"),
            total_entropy_reduction=("entropy_reduction", "sum"),
            mean_delay_gain=("delay_gain", "mean"),
            mean_utility=("utility_true", "mean"),
            mean_oracle_utility=("oracle_utility", "mean"),
            mean_utility_gap_to_oracle=("utility_gap_to_oracle", "mean"),
            policy_hit_rate_oracle=("policy_hit_oracle", "mean"),
            total_operational_cost=("operational_cost", "sum"),
            mean_operational_cost=("operational_cost", "mean"),
            total_churn=("churn", "sum"),
            mean_churn=("churn", "mean"),
        )
        .reset_index()
    )
    agg = agg.sort_values("mean_utility", ascending=False).reset_index(drop=True)
    return agg


def make_baseline_comparison(policy_summary: pd.DataFrame, proposed_policy: str = "proposed_value_policy") -> pd.DataFrame:
    if proposed_policy not in set(policy_summary["policy"]):
        return policy_summary.copy()
    proposed = policy_summary[policy_summary["policy"] == proposed_policy].iloc[0]
    rows = []
    for _, row in policy_summary.iterrows():
        rows.append(
            {
                "baseline_policy": row["policy"],
                "proposed_policy": proposed_policy,
                "delta_mean_utility": proposed["mean_utility"] - row["mean_utility"],
                "delta_target_reach_reduction": proposed["mean_target_reach_reduction"] - row["mean_target_reach_reduction"],
                "delta_action_target_prob": proposed["mean_action_target_prob"] - row["mean_action_target_prob"],
                "delta_info_gain": proposed["mean_info_gain"] - row["mean_info_gain"],
                "delta_delay_gain": proposed["mean_delay_gain"] - row["mean_delay_gain"],
                "delta_cost": proposed["mean_operational_cost"] - row["mean_operational_cost"],
                "delta_churn": proposed["mean_churn"] - row["mean_churn"],
                "delta_utility_gap_to_oracle": proposed["mean_utility_gap_to_oracle"] - row["mean_utility_gap_to_oracle"],
            }
        )
    return pd.DataFrame(rows)
