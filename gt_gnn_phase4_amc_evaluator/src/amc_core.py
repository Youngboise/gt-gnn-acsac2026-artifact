from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import clamp, clamp01, normalized_entropy, safe_float, softmax


@dataclass
class BeliefProfile:
    goal_dc_access: float = 0.25
    goal_cred_collection: float = 0.25
    goal_service_disruption: float = 0.25
    goal_exploration: float = 0.25
    skill_low: float = 0.33
    skill_medium: float = 0.34
    skill_high: float = 0.33
    stealth_noisy: float = 0.33
    stealth_balanced: float = 0.34
    stealth_stealthy: float = 0.33
    decoy_unaware: float = 0.34
    decoy_suspecting: float = 0.33
    decoy_aware: float = 0.33
    mean_entropy: float = 1.0

    @property
    def target_bias(self) -> float:
        return clamp01(0.75 * self.goal_dc_access + 0.45 * self.goal_cred_collection + 0.25 * self.goal_service_disruption + 0.20 * self.goal_exploration)

    @property
    def skill_efficiency(self) -> float:
        return clamp01(0.20 * self.skill_low + 0.55 * self.skill_medium + 0.95 * self.skill_high)

    @property
    def stealth(self) -> float:
        return clamp01(0.15 * self.stealth_noisy + 0.55 * self.stealth_balanced + 0.95 * self.stealth_stealthy)

    @property
    def noisy_exposure(self) -> float:
        return clamp01(0.95 * self.stealth_noisy + 0.35 * self.stealth_balanced + 0.10 * self.stealth_stealthy)

    @property
    def decoy_awareness(self) -> float:
        return clamp01(0.05 * self.decoy_unaware + 0.55 * self.decoy_suspecting + 0.95 * self.decoy_aware)


def belief_profile_from_graph(graph: Dict[str, Any], labels: Dict[str, List[str]]) -> BeliefProfile:
    belief = list(graph.get("belief_vector") or [])
    offset = 0
    chunks: Dict[str, Dict[str, float]] = {}
    for factor in ["goal", "skill", "stealth", "decoy_awareness"]:
        names = list(labels.get(factor, []))
        n = len(names)
        probs = belief[offset : offset + n]
        offset += n
        if len(probs) != n or n == 0:
            probs = [1.0 / max(1, n)] * n
        s = sum(float(p) for p in probs)
        if s <= 0:
            probs = [1.0 / max(1, n)] * n
            s = 1.0
        chunks[factor] = {name: float(p) / s for name, p in zip(names, probs)}

    ent_obj = graph.get("belief_entropy") or {}
    if isinstance(ent_obj, dict):
        mean_entropy = safe_float(ent_obj.get("mean"), normalized_entropy(belief))
    else:
        mean_entropy = normalized_entropy(belief)

    return BeliefProfile(
        goal_dc_access=chunks.get("goal", {}).get("dc_access", 0.25),
        goal_cred_collection=chunks.get("goal", {}).get("cred_collection", 0.25),
        goal_service_disruption=chunks.get("goal", {}).get("service_disruption", 0.25),
        goal_exploration=chunks.get("goal", {}).get("exploration", 0.25),
        skill_low=chunks.get("skill", {}).get("low", 0.33),
        skill_medium=chunks.get("skill", {}).get("medium", 0.34),
        skill_high=chunks.get("skill", {}).get("high", 0.33),
        stealth_noisy=chunks.get("stealth", {}).get("noisy", 0.33),
        stealth_balanced=chunks.get("stealth", {}).get("balanced", 0.34),
        stealth_stealthy=chunks.get("stealth", {}).get("stealthy", 0.33),
        decoy_unaware=chunks.get("decoy_awareness", {}).get("unaware", 0.34),
        decoy_suspecting=chunks.get("decoy_awareness", {}).get("suspecting", 0.33),
        decoy_aware=chunks.get("decoy_awareness", {}).get("aware", 0.33),
        mean_entropy=mean_entropy,
    )


def _node_by_index(graph: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(n["strategic_index"]): n for n in graph.get("nodes", [])}


def _node_by_id(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(n["node_id"]): n for n in graph.get("nodes", [])}


def _node_feature(node: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return safe_float((node.get("features") or {}).get(key), default)


def _node_score(node: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return safe_float((node.get("scores") or {}).get(key), default)


def _is_target(node: Dict[str, Any]) -> bool:
    return safe_float(node.get("target_mask"), 0.0) > 0.5 or "target" in set(node.get("tags") or [])


def _edge_feature(edge: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return safe_float((edge.get("features") or {}).get(key), default)


def _candidate_node_id(action: Optional[Dict[str, Any]]) -> Optional[str]:
    if action is None:
        return None
    return str(action.get("node_id"))


def _decoy_absorption_at_node(node: Dict[str, Any], action_node_id: Optional[str], belief: BeliefProfile, params: Dict[str, float]) -> float:
    is_action = 1.0 if action_node_id and str(node.get("node_id")) == action_node_id else 0.0
    active_decoy = _node_feature(node, "active_decoy")
    recent_trigger = _node_feature(node, "recent_trigger")
    revelation = _node_score(node, "revelation_score")
    base = safe_float(params.get("decoy_base"), 0.025)
    raw = (
        base
        + safe_float(params.get("action_decoy_strength"), 0.38) * is_action
        + safe_float(params.get("active_decoy_strength"), 0.16) * active_decoy
        + safe_float(params.get("trigger_decoy_strength"), 0.16) * recent_trigger
        + safe_float(params.get("revelation_decoy_strength"), 0.20) * revelation
        + safe_float(params.get("noisy_decoy_exposure"), 0.18) * belief.noisy_exposure * (0.35 + 0.65 * revelation)
    )
    avoidance = (
        safe_float(params.get("awareness_decoy_avoidance"), 0.62) * belief.decoy_awareness
        + safe_float(params.get("stealth_decoy_avoidance"), 0.28) * belief.stealth
    )
    # Awareness reduces decoy absorption, but does not make it zero because decoys can still reveal avoidance.
    p = raw * clamp(1.0 - 0.55 * avoidance, 0.18, 1.0)
    return clamp01(p)


def _abort_probability(node: Dict[str, Any], params: Dict[str, float]) -> float:
    alert = _node_feature(node, "alert_intensity")
    criticality = _node_feature(node, "criticality")
    base = safe_float(params.get("abort_base"), 0.025)
    scale = safe_float(params.get("detection_abort_scale"), 0.045)
    return clamp01(base + scale * (0.75 * alert + 0.25 * criticality))


def _transition_score(src_node: Dict[str, Any], dst_node: Dict[str, Any], edge: Dict[str, Any], action_node_id: Optional[str], belief: BeliefProfile, params: Dict[str, float]) -> float:
    dst_target = 1.0 if _is_target(dst_node) else 0.0
    transition_base = _edge_feature(edge, "transition_base", 0.25)
    reliability = _edge_feature(edge, "reliability", 0.65)
    movement_cost = _edge_feature(edge, "movement_cost", 0.5)
    edge_deception = _edge_feature(edge, "deception_penalty", 0.0)

    dst_exposure = _node_feature(dst_node, "exposure")
    dst_vuln = _node_feature(dst_node, "vulnerability")
    dst_alert = _node_feature(dst_node, "alert_intensity")
    dst_distance = _node_feature(dst_node, "distance_to_target")
    dst_threat = _node_score(dst_node, "threat_score")
    dst_revelation = _node_score(dst_node, "revelation_score")
    dst_is_action = 1.0 if action_node_id and str(dst_node.get("node_id")) == action_node_id else 0.0

    score = 0.0
    score += safe_float(params.get("transition_base_scale"), 1.35) * transition_base
    score += safe_float(params.get("reliability_scale"), 0.45) * reliability
    score -= safe_float(params.get("movement_cost_scale"), 0.85) * movement_cost
    score += safe_float(params.get("exposure_scale"), 0.35) * dst_exposure
    score += safe_float(params.get("vulnerability_scale"), 0.35) * dst_vuln
    score += safe_float(params.get("alert_scale"), 0.20) * dst_alert
    score += safe_float(params.get("target_proximity_scale"), 0.65) * dst_distance
    score += safe_float(params.get("skill_movement_scale"), 0.25) * belief.skill_efficiency
    score += 0.25 * dst_threat

    if dst_target > 0:
        score += safe_float(params.get("target_bias_scale"), 0.95) * belief.target_bias

    # More aware/stealthy attackers penalize edges that look deceptive, especially toward a newly selected decoy.
    score -= safe_float(params.get("edge_deception_penalty_scale"), 0.60) * edge_deception * (0.35 + 0.65 * belief.decoy_awareness)
    score -= safe_float(params.get("action_avoidance_scale"), 0.70) * dst_is_action * (0.55 * belief.decoy_awareness + 0.45 * belief.stealth)

    # Unaware or noisy attackers can be pulled into high-revelation decoys.
    baitability = clamp01((1.0 - belief.decoy_awareness) * 0.65 + belief.noisy_exposure * 0.35)
    score += safe_float(params.get("action_bait_scale"), 0.25) * dst_is_action * baitability * dst_revelation
    return float(score)


def _build_outgoing_edges(graph: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for e in graph.get("edges", []):
        out.setdefault(int(e["source"]), []).append(e)
    return out


def _initial_distribution(graph: Dict[str, Any], transient_indices: List[int], node_map: Dict[int, Dict[str, Any]]) -> np.ndarray:
    weights: List[float] = []
    entry_ids = set((graph.get("seed_nodes") or {}).get("entries") or [])
    for idx in transient_indices:
        node = node_map[idx]
        tags = set(node.get("tags") or [])
        is_entry = 1.0 if str(node.get("node_id")) in entry_ids or "entries" in tags else 0.0
        w = 0.05 + 1.10 * is_entry + 0.55 * _node_feature(node, "exposure") + 0.35 * _node_feature(node, "alert_intensity") + 0.20 * _node_feature(node, "vulnerability")
        weights.append(max(1e-9, w))
    arr = np.asarray(weights, dtype=np.float64)
    arr = arr / max(float(arr.sum()), 1e-12)
    return arr


def evaluate_amc_for_action(graph: Dict[str, Any], action: Optional[Dict[str, Any]], params: Dict[str, float], labels: Dict[str, List[str]]) -> Dict[str, float]:
    node_map = _node_by_index(graph)
    if not node_map:
        return {
            "target_absorption_prob": 0.0,
            "decoy_absorption_prob": 0.0,
            "abort_absorption_prob": 1.0,
            "expected_steps_to_absorption": 0.0,
            "num_transient_states": 0,
        }
    belief = belief_profile_from_graph(graph, labels)
    action_node_id = _candidate_node_id(action)
    transient_indices = [idx for idx, node in sorted(node_map.items()) if not _is_target(node)]
    if not transient_indices:
        return {
            "target_absorption_prob": 1.0,
            "decoy_absorption_prob": 0.0,
            "abort_absorption_prob": 0.0,
            "expected_steps_to_absorption": 0.0,
            "num_transient_states": 0,
        }
    state_id = {idx: k for k, idx in enumerate(transient_indices)}
    m = len(transient_indices)
    Q = np.zeros((m, m), dtype=np.float64)
    R = np.zeros((m, 3), dtype=np.float64)  # target, decoy, abort
    outgoing = _build_outgoing_edges(graph)
    temp = safe_float(params.get("temperature"), 0.75)

    for src_idx in transient_indices:
        row = state_id[src_idx]
        src_node = node_map[src_idx]
        p_decoy = _decoy_absorption_at_node(src_node, action_node_id, belief, params)
        p_abort = _abort_probability(src_node, params)
        # Keep movement possible even at highly deceptive nodes.
        p_decoy = min(p_decoy, 0.82)
        p_abort = min(p_abort, 0.45)
        if p_decoy + p_abort > 0.92:
            scale = 0.92 / max(p_decoy + p_abort, 1e-12)
            p_decoy *= scale
            p_abort *= scale
        p_move = max(0.0, 1.0 - p_decoy - p_abort)
        R[row, 1] += p_decoy
        R[row, 2] += p_abort

        edge_list = outgoing.get(src_idx, [])
        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for edge in edge_list:
            dst_idx = int(edge["target"])
            dst_node = node_map.get(dst_idx)
            if dst_node is None:
                continue
            scored.append((_transition_score(src_node, dst_node, edge, action_node_id, belief, params), edge, dst_node))

        if not scored:
            R[row, 2] += p_move
            continue
        probs = softmax([s for s, _, _ in scored], temperature=temp)
        for prob, (_, edge, dst_node) in zip(probs, scored):
            mass = p_move * float(prob)
            dst_idx = int(edge["target"])
            if _is_target(dst_node):
                R[row, 0] += mass
            elif dst_idx in state_id:
                Q[row, state_id[dst_idx]] += mass
            else:
                R[row, 2] += mass

        # Numerical guard: any residual mass becomes abort.
        residual = 1.0 - float(Q[row, :].sum() + R[row, :].sum())
        if residual > 1e-9:
            R[row, 2] += residual
        elif residual < -1e-8:
            total = float(Q[row, :].sum() + R[row, :].sum())
            Q[row, :] /= total
            R[row, :] /= total

    I = np.eye(m, dtype=np.float64)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    B = N @ R
    steps = N @ np.ones((m, 1), dtype=np.float64)
    init = _initial_distribution(graph, transient_indices, node_map).reshape(1, -1)
    agg_absorb = (init @ B).reshape(-1)
    agg_steps = float((init @ steps).reshape(-1)[0])
    visit_counts = (init @ N).reshape(-1)
    action_visit = 0.0
    if action_node_id is not None:
        for original_idx, st_idx in state_id.items():
            if str(node_map[original_idx].get("node_id")) == action_node_id:
                action_visit = float(visit_counts[st_idx])
                break
    return {
        "target_absorption_prob": clamp01(float(agg_absorb[0])),
        "decoy_absorption_prob": clamp01(float(agg_absorb[1])),
        "abort_absorption_prob": clamp01(float(agg_absorb[2])),
        "expected_steps_to_absorption": max(0.0, agg_steps),
        "expected_visit_count_action_node": max(0.0, action_visit),
        "num_transient_states": int(m),
    }


def evaluate_action_with_baseline(graph: Dict[str, Any], action: Dict[str, Any], params: Dict[str, float], labels: Dict[str, List[str]], utility_cfg: Dict[str, float]) -> Dict[str, Any]:
    baseline = evaluate_amc_for_action(graph, None, params, labels)
    actual = evaluate_amc_for_action(graph, action, params, labels)

    baseline_target = baseline["target_absorption_prob"]
    action_target = actual["target_absorption_prob"]
    risk_reduction = max(0.0, baseline_target - action_target)
    relative_risk_reduction = clamp01(risk_reduction / max(baseline_target, 1e-6))

    baseline_steps = baseline["expected_steps_to_absorption"]
    action_steps = actual["expected_steps_to_absorption"]
    raw_delay = max(0.0, action_steps - baseline_steps)
    delay_norm = clamp01(raw_delay / (raw_delay + safe_float(params.get("delay_conversion_scale"), 3.0)))
    # Decoy absorption can terminate earlier while still creating defensive delay/diversion value.
    delay_gain = clamp01(delay_norm + 0.35 * safe_float(action.get("delay_proxy"), 0.0) * actual["decoy_absorption_prob"])

    ent = safe_float(action.get("belief_entropy"), safe_float((graph.get("belief_entropy") or {}).get("mean"), 0.0))
    expected_info_gain = clamp01(
        safe_float(params.get("information_gain_scale"), 0.85)
        * ent
        * safe_float(action.get("info_proxy"), 0.0)
        * (0.25 + 0.75 * actual["decoy_absorption_prob"])
    )

    operational_cost = safe_float(action.get("operational_cost"), 0.0)
    churn_proxy = safe_float(action.get("churn_proxy"), 0.0)
    utility = (
        safe_float(utility_cfg.get("lambda_risk"), 1.0) * relative_risk_reduction
        + safe_float(utility_cfg.get("lambda_info"), 0.70) * expected_info_gain
        + safe_float(utility_cfg.get("lambda_delay"), 0.55) * delay_gain
        - safe_float(utility_cfg.get("lambda_cost"), 0.35) * operational_cost
        - safe_float(utility_cfg.get("lambda_churn"), 0.15) * churn_proxy
    )

    out = {
        "graph_id": graph.get("graph_id"),
        "episode_id": graph.get("episode_id"),
        "t": graph.get("t"),
        "split": graph.get("split"),
        "action_id": action.get("action_id"),
        "node_id": action.get("node_id"),
        "rank_phase3": action.get("rank"),
        "action_type": action.get("action_type", "place_or_retain_decoy"),
        "decoy_type_hint": action.get("decoy_type_hint"),
        "exposure_level_hint": action.get("exposure_level_hint"),
        "action_is_high_interaction": 1.0 if action.get("action_type") == "place_high_interaction_decoy" else 0.0,
        "action_is_low_exposure": 1.0 if action.get("action_type") == "place_low_exposure_decoy" else 0.0,
        "action_is_credential_breadcrumb": 1.0 if action.get("action_type") == "credential_breadcrumb" else 0.0,
        "action_is_telemetry_trap": 1.0 if action.get("action_type") == "telemetry_trap" else 0.0,
        "action_is_path_perturbation": 1.0 if action.get("action_type") == "path_perturbation" else 0.0,
        "action_is_rotate_or_retain": 1.0 if action.get("action_type") == "rotate_or_retain_decoy" else 0.0,
        "exposure_level_value": 1.0 if action.get("exposure_level_hint") == "high" else (0.5 if action.get("exposure_level_hint") == "medium" else 0.0),
        "baseline_target_prob": baseline_target,
        "action_target_prob": action_target,
        "target_reach_reduction": risk_reduction,
        "relative_risk_reduction": relative_risk_reduction,
        "baseline_decoy_absorption_prob": baseline["decoy_absorption_prob"],
        "action_decoy_absorption_prob": actual["decoy_absorption_prob"],
        "action_abort_absorption_prob": actual["abort_absorption_prob"],
        "baseline_expected_steps": baseline_steps,
        "action_expected_steps": action_steps,
        "raw_delay_delta": raw_delay,
        "delay_gain": delay_gain,
        "expected_info_gain": expected_info_gain,
        "expected_visit_count_action_node": actual.get("expected_visit_count_action_node", 0.0),
        "utility_amc": utility,
        "operational_cost": operational_cost,
        "churn_proxy": churn_proxy,
        "risk_proxy": safe_float(action.get("risk_proxy"), 0.0),
        "info_proxy": safe_float(action.get("info_proxy"), 0.0),
        "delay_proxy": safe_float(action.get("delay_proxy"), 0.0),
        "threat_score": safe_float(action.get("threat_score"), 0.0),
        "revelation_score": safe_float(action.get("revelation_score"), 0.0),
        "fused_score": safe_float(action.get("fused_score"), 0.0),
        "candidate_score": safe_float(action.get("candidate_score"), 0.0),
        "belief_entropy": ent,
        "num_transient_states": actual.get("num_transient_states", 0),
    }
    return out


def evaluate_graph_actions(graph: Dict[str, Any], params: Dict[str, float], labels: Dict[str, List[str]], utility_cfg: Dict[str, float], max_actions: Optional[int] = None) -> List[Dict[str, Any]]:
    actions = list(graph.get("candidate_actions") or [])
    if max_actions is not None:
        actions = actions[: int(max_actions)]
    return [evaluate_action_with_baseline(graph, a, params, labels, utility_cfg) for a in actions]
