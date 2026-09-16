from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .phase3_io import load_phase3_strategic_graphs
from .utils import clamp01, load_yaml, package_root_from_config, safe_float, set_seed, write_json


def _node_maps(graph: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    nodes = {str(n["node_id"]): n for n in graph.get("nodes", [])}
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.get("edges", []):
        u = str(e.get("source_node_id"))
        v = str(e.get("target_node_id"))
        if u in nodes and v in nodes and v not in adj[u]:
            adj[u].append(v)
    # Ensure every node has at least a self-loop fallback.
    for node_id in nodes:
        adj.setdefault(node_id, [])
    return nodes, adj


def _feature(node: dict[str, Any], name: str, default: float = 0.0) -> float:
    return safe_float(node.get("features", {}).get(name, default), default)


def _score(node: dict[str, Any], name: str, default: float = 0.0) -> float:
    return safe_float(node.get("scores", {}).get(name, default), default)


def _target_nodes(nodes: dict[str, dict[str, Any]]) -> list[str]:
    targets = [nid for nid, n in nodes.items() if safe_float(n.get("target_mask", 0.0)) > 0.5 or "target" in n.get("tags", [])]
    if targets:
        return targets
    return sorted(nodes, key=lambda nid: _feature(nodes[nid], "criticality", 0.0), reverse=True)[:1]


def _entry_distribution(nodes: dict[str, dict[str, Any]], targets: set[str]) -> tuple[list[str], np.ndarray]:
    candidates = [nid for nid in nodes if nid not in targets]
    if not candidates:
        candidates = list(nodes)
    weights = []
    for nid in candidates:
        n = nodes[nid]
        weights.append(0.20 + 0.50 * _feature(n, "exposure") + 0.35 * _feature(n, "alert_intensity") + 0.15 * (1.0 - _feature(n, "distance_to_target", 0.5)))
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    return candidates, w


def _next_distribution(
    cur: str,
    nodes: dict[str, dict[str, Any]],
    adj: dict[str, list[str]],
    decoy_node: str | None,
    decoy_aware: float,
    action_mod: dict[str, float],
    rng: np.random.Generator,
) -> str:
    nbrs = adj.get(cur, []) or list(nodes)
    vals = []
    for nb in nbrs:
        n = nodes[nb]
        target_pull = _feature(n, "distance_to_target", 0.5)
        movement = 0.35 * target_pull + 0.25 * _feature(n, "vulnerability") + 0.20 * _feature(n, "admin_bridge") + 0.15 * _feature(n, "lateral_bridge") + 0.10 * _score(n, "threat_score")
        if decoy_node is not None and nb == decoy_node:
            # Action-type aware attacker response: low-exposure traps are less avoided; high-interaction decoys bait more but are easier to avoid.
            movement += action_mod.get("bait", 0.0) * (1.0 - decoy_aware) - action_mod.get("avoidance", 0.0) * decoy_aware
        vals.append(movement)
    logits = np.asarray(vals, dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits / 0.55)
    probs = probs / max(float(probs.sum()), 1e-12)
    return str(rng.choice(nbrs, p=probs))


def _action_modifiers(action: dict[str, Any] | None) -> dict[str, float]:
    if action is None:
        return {"bait": 0.0, "avoidance": 0.0, "trigger": 0.0, "info": 0.0, "delay": 0.0, "cost": 0.0}
    atype = str(action.get("action_type", "place_or_retain_decoy"))
    exposure_hint = str(action.get("exposure_level_hint", "medium"))
    # bait attracts unaware/noisy attackers; avoidance is how much decoy-aware attackers avoid it.
    table = {
        "place_high_interaction_decoy": {"bait": 0.58, "avoidance": 0.44, "trigger": 0.35, "info": 0.22, "delay": 0.06, "cost": 0.10},
        "place_low_exposure_decoy": {"bait": 0.20, "avoidance": 0.12, "trigger": 0.20, "info": 0.35, "delay": 0.08, "cost": 0.02},
        "credential_breadcrumb": {"bait": 0.34, "avoidance": 0.18, "trigger": 0.28, "info": 0.42, "delay": 0.04, "cost": 0.04},
        "telemetry_trap": {"bait": 0.10, "avoidance": 0.04, "trigger": 0.18, "info": 0.55, "delay": 0.02, "cost": 0.01},
        "path_perturbation": {"bait": 0.12, "avoidance": 0.08, "trigger": 0.08, "info": 0.12, "delay": 0.45, "cost": 0.06},
        "rotate_or_retain_decoy": {"bait": 0.16, "avoidance": 0.10, "trigger": 0.20, "info": 0.22, "delay": 0.14, "cost": -0.02},
        "place_or_retain_decoy": {"bait": 0.32, "avoidance": 0.22, "trigger": 0.24, "info": 0.24, "delay": 0.08, "cost": 0.03},
    }
    out = dict(table.get(atype, table["place_or_retain_decoy"]))
    if exposure_hint == "high":
        out["bait"] += 0.10; out["avoidance"] += 0.08; out["trigger"] += 0.06; out["cost"] += 0.04
    elif exposure_hint == "low":
        out["bait"] -= 0.06; out["avoidance"] -= 0.05; out["info"] += 0.06; out["cost"] -= 0.02
    return {k: clamp01(v) if k != "cost" else float(v) for k, v in out.items()}


def _rollout_once(
    graph: dict[str, Any],
    action: dict[str, Any] | None,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    nodes, adj = _node_maps(graph)
    if not nodes:
        return {"target": 0.0, "decoy": 0.0, "steps": float(cfg.get("horizon", 8)), "triggered": 0.0}
    targets = set(_target_nodes(nodes))
    entries, entry_p = _entry_distribution(nodes, targets)
    cur = str(rng.choice(entries, p=entry_p))
    decoy_node = str(action["node_id"]) if action is not None else None
    action_mod = _action_modifiers(action)
    entropy = safe_float(graph.get("belief_entropy", {}).get("mean", action.get("belief_entropy", 0.5) if action else 0.5), 0.5)
    # Higher entropy means defender is less sure; attacker awareness is sampled broadly.
    decoy_aware = clamp01(0.30 + 0.35 * (1.0 - entropy) + rng.normal(0.0, 0.08))
    horizon = int(cfg.get("horizon", 8))
    triggered = 0.0
    for step in range(horizon + 1):
        if cur in targets:
            return {"target": 1.0, "decoy": 0.0, "steps": float(step), "triggered": triggered}
        if decoy_node is not None and cur == decoy_node:
            decoy_n = nodes.get(cur, {})
            trigger_p = clamp01(0.18 + action_mod.get("trigger", 0.0) + 0.32 * _feature(decoy_n, "exposure") + 0.34 * _score(decoy_n, "revelation_score") - 0.30 * decoy_aware + 0.12 * action_mod.get("info", 0.0))
            if rng.random() < trigger_p:
                return {"target": 0.0, "decoy": 1.0, "steps": float(step), "triggered": 1.0}
            triggered = max(triggered, 0.5)
        if step == horizon:
            break
        cur = _next_distribution(cur, nodes, adj, decoy_node, decoy_aware, action_mod, rng)
    return {"target": 0.0, "decoy": 0.0, "steps": float(horizon), "triggered": triggered}


def _simulate_action(graph: dict[str, Any], action: dict[str, Any], cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    n = int(cfg.get("rollouts_per_action", 96))
    base = [_rollout_once(graph, None, cfg, rng) for _ in range(n)]
    acted = [_rollout_once(graph, action, cfg, rng) for _ in range(n)]
    base_target = float(np.mean([r["target"] for r in base]))
    action_target = float(np.mean([r["target"] for r in acted]))
    decoy_abs = float(np.mean([r["decoy"] for r in acted]))
    base_steps = float(np.mean([r["steps"] for r in base]))
    action_steps = float(np.mean([r["steps"] for r in acted]))
    entropy = safe_float(action.get("belief_entropy", graph.get("belief_entropy", {}).get("mean", 0.5)), 0.5)
    action_mod = _action_modifiers(action)
    trigger = float(np.mean([r["triggered"] for r in acted]))
    risk = clamp01(base_target - action_target)
    delay = clamp01((action_steps - base_steps) / max(float(cfg.get("horizon", 8)), 1.0) + action_mod.get("delay", 0.0) * safe_float(action.get("delay_proxy", 0.0)))
    info = clamp01(entropy * (0.58 * decoy_abs + 0.42 * trigger) + 0.12 * safe_float(action.get("revelation_score", 0.0)) + action_mod.get("info", 0.0) * (0.35 + 0.65 * entropy))
    cost = clamp01(safe_float(action.get("operational_cost", 0.0), 0.0) + action_mod.get("cost", 0.0))
    utility_raw = (
        float(cfg.get("lambda_risk", 1.0)) * risk
        + float(cfg.get("lambda_info", 0.7)) * info
        + float(cfg.get("lambda_delay", 0.55)) * delay
        + float(cfg.get("lambda_decoy", 0.25)) * decoy_abs
        - float(cfg.get("lambda_cost", 0.20)) * cost
    )
    return {
        "graph_id": graph["graph_id"],
        "episode_id": graph.get("episode_id"),
        "t": graph.get("t"),
        "split": graph.get("split"),
        "action_id": action["action_id"],
        "node_id": action["node_id"],
        "action_type": action.get("action_type"),
        "decoy_type_hint": action.get("decoy_type_hint"),
        "exposure_level_hint": action.get("exposure_level_hint"),
        "baseline_target_prob_cf": base_target,
        "action_target_prob_cf": action_target,
        "target_reach_reduction_cf": risk,
        "action_decoy_absorption_prob_cf": decoy_abs,
        "expected_info_gain_cf": info,
        "delay_gain_cf": delay,
        "utility_counterfactual_raw": utility_raw,
        "label_source": "monte_carlo_counterfactual_rollout",
    }


def simulate(config_path: str) -> dict[str, Any]:
    cfg_all = load_yaml(config_path)
    root = package_root_from_config(config_path)
    set_seed(int(cfg_all.get("seed", 17)))
    sim_cfg = cfg_all.get("intervention_simulation", {})
    graphs = load_phase3_strategic_graphs(root / cfg_all["paths"]["phase3_strategic_graphs_jsonl"])
    max_graphs = sim_cfg.get("max_graphs", None)
    if max_graphs not in (None, "", 0):
        graphs = graphs[: int(max_graphs)]
    rng = np.random.default_rng(int(sim_cfg.get("seed", cfg_all.get("seed", 17))))
    rows: list[dict[str, Any]] = []
    for i, graph in enumerate(graphs):
        for action in graph.get("candidate_actions", []):
            rows.append(_simulate_action(graph, action, sim_cfg, rng))
        if (i + 1) % 100 == 0:
            print(f"Simulated {i + 1:,}/{len(graphs):,} strategic graphs...")
    df = pd.DataFrame(rows)
    if len(df):
        # Normalize counterfactual utility within each graph for ranking labels.
        lo = df.groupby("graph_id")["utility_counterfactual_raw"].transform("min")
        hi = df.groupby("graph_id")["utility_counterfactual_raw"].transform("max")
        denom = (hi - lo).replace(0.0, np.nan)
        df["utility_counterfactual_norm"] = ((df["utility_counterfactual_raw"] - lo) / denom).fillna(0.0).clip(0.0, 1.0)
        df["counterfactual_rank"] = df.groupby("graph_id")["utility_counterfactual_raw"].rank(method="first", ascending=False).astype(int)
        df["is_best_action_cf"] = (df["counterfactual_rank"] == 1).astype(int)
    out_csv = root / cfg_all["paths"].get("intervention_labels_csv", "data/intervention_rollout_labels.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = {
        "num_graphs": int(len(graphs)),
        "num_action_labels": int(len(df)),
        "output_csv": str(out_csv),
        "rollouts_per_action": int(sim_cfg.get("rollouts_per_action", 96)),
        "label_source": "monte_carlo_counterfactual_rollout_independent_of_phase3_proxy_labels",
    }
    write_json(summary, root / cfg_all["paths"].get("intervention_labels_summary_json", "outputs/intervention_rollout_labels_summary.json"))
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/amc_config.yaml")
    args = parser.parse_args()
    simulate(args.config)


if __name__ == "__main__":
    main()
