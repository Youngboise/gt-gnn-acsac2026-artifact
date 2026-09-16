from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .phase2_io import get_feature_index, get_scores_for_sample
from .utils import clamp01, normalized_entropy, tensor_to_numpy


@dataclass
class ExtractionContext:
    cfg: Dict[str, Any]
    feature_names: List[str]
    feature_idx: Dict[str, int]
    pred_lookup: Dict[Tuple[int, int, str], Dict[str, float]]
    allow_proxy_fallback: bool = False


def _top_indices(values: np.ndarray, k: int, mask: Optional[np.ndarray] = None, exclude: Optional[Set[int]] = None) -> List[int]:
    values = np.asarray(values, dtype=np.float64)
    if mask is None:
        mask = np.ones_like(values, dtype=bool)
    else:
        mask = np.asarray(mask) > 0.5
    if exclude is None:
        exclude = set()
    candidates = [(float(values[i]), int(i)) for i in range(len(values)) if mask[i] and i not in exclude]
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in candidates[: max(0, int(k))]]




def _merge_unique(*groups: Iterable[int], limit: Optional[int] = None) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for group in groups:
        for item in group:
            i = int(item)
            if i not in seen:
                out.append(i)
                seen.add(i)
                if limit is not None and len(out) >= int(limit):
                    return out
    return out


def _stable_random_indices(n: int, k: int, mask: Optional[np.ndarray], seed_value: int, exclude: Optional[Set[int]] = None) -> List[int]:
    if exclude is None:
        exclude = set()
    if mask is None:
        valid = [i for i in range(n) if i not in exclude]
    else:
        m = np.asarray(mask) > 0.5
        valid = [i for i in range(n) if bool(m[i]) and i not in exclude]
    if not valid or k <= 0:
        return []
    rng = np.random.default_rng(int(seed_value) % (2**32 - 1))
    chosen = rng.choice(np.asarray(valid), size=min(int(k), len(valid)), replace=False)
    return [int(x) for x in chosen.tolist()]


def _node_feature(x: np.ndarray, idx: Dict[str, int], name: str, default: float = 0.0) -> np.ndarray:
    if name not in idx:
        return np.full(x.shape[0], default, dtype=np.float32)
    return x[:, idx[name]].astype(np.float32)


def _node_scalar(x: np.ndarray, idx: Dict[str, int], i: int, name: str, default: float = 0.0) -> float:
    if name not in idx:
        return float(default)
    return float(x[i, idx[name]])


def _role_from_features(x: np.ndarray, idx: Dict[str, int], i: int) -> str:
    role_names = ["role_workstation", "role_server", "role_dc", "role_service", "role_honeypot"]
    scored = []
    for name in role_names:
        if name in idx:
            scored.append((float(x[i, idx[name]]), name.replace("role_", "")))
    if not scored:
        return "unknown"
    scored.sort(reverse=True)
    return scored[0][1]


def _build_nx_graph(edge_index: np.ndarray, n: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_index.size > 0:
        for u, v in edge_index.T.tolist():
            u = int(u)
            v = int(v)
            if u != v:
                g.add_edge(u, v)
    return g


def _k_shortest_path_nodes(g: nx.Graph, sources: Iterable[int], targets: Iterable[int], k: int) -> Set[int]:
    out: Set[int] = set()
    for s in sources:
        for t in targets:
            if s == t:
                out.add(int(s))
                continue
            try:
                gen = nx.shortest_simple_paths(g, int(s), int(t))
                for _, path in zip(range(max(1, k)), gen):
                    out.update(int(p) for p in path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
    return out


def _khop_nodes(g: nx.Graph, centers: Iterable[int], radius: int) -> Set[int]:
    out: Set[int] = set()
    for c in centers:
        if c not in g:
            continue
        lengths = nx.single_source_shortest_path_length(g, int(c), cutoff=max(0, int(radius)))
        out.update(int(n) for n in lengths.keys())
    return out


def _belief_factor_entropies(belief: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, float]:
    labels = cfg.get("labels", {})
    offset = 0
    out: Dict[str, float] = {}
    for key in ["goal", "skill", "stealth", "decoy_awareness"]:
        n = len(labels.get(key, []))
        if n <= 0 or offset + n > len(belief):
            continue
        out[key] = normalized_entropy(belief[offset : offset + n])
        offset += n
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    else:
        out["mean"] = normalized_entropy(belief)
    return out


def _priority_scores(x: np.ndarray, sample: Dict[str, Any], scores: Dict[str, np.ndarray], ctx: ExtractionContext) -> np.ndarray:
    weights = ctx.cfg.get("scoring", {}).get("node_priority_weights", {})
    idx = ctx.feature_idx
    n = x.shape[0]
    priority = np.zeros(n, dtype=np.float32)
    for name, weight in weights.items():
        if name in scores:
            vals = scores[name]
        elif name == "candidate_mask":
            vals = tensor_to_numpy(sample.get("candidate_mask", np.zeros(n)))
        elif name == "target_mask":
            vals = tensor_to_numpy(sample.get("target_mask", np.zeros(n)))
        elif name in idx:
            vals = x[:, idx[name]]
        else:
            continue
        priority += float(weight) * vals.astype(np.float32)
    return priority


def _select_seed_nodes(sample: Dict[str, Any], x: np.ndarray, g: nx.Graph, scores: Dict[str, np.ndarray], ctx: ExtractionContext) -> Dict[str, List[int]]:
    cfg = ctx.cfg["selection"]
    idx = ctx.feature_idx
    n = x.shape[0]
    candidate_mask = tensor_to_numpy(sample.get("candidate_mask", np.zeros(n)))
    target_mask = tensor_to_numpy(sample.get("target_mask", np.zeros(n)))
    non_target = set(np.where(target_mask <= 0.5)[0].astype(int).tolist())

    target_nodes = np.where(target_mask > 0.5)[0].astype(int).tolist()
    if not target_nodes:
        target_nodes = _top_indices(_node_feature(x, idx, "criticality", 0.0), 1)

    alert = _node_feature(x, idx, "alert_intensity", 0.0)
    exposure = _node_feature(x, idx, "exposure", 0.0)
    distance = _node_feature(x, idx, "distance_to_target", 0.0)
    entry_score = 0.45 * exposure + 0.45 * alert + 0.10 * (1.0 - distance)
    entry_nodes = _top_indices(entry_score, int(cfg["top_k_entry_nodes"]), mask=(target_mask <= 0.5))

    exclude_targets = set(target_nodes)
    feasible_mask = (candidate_mask > 0.5) & (target_mask <= 0.5)
    threat_nodes = _top_indices(scores["threat_score"], int(cfg["top_k_threat_nodes"]), mask=feasible_mask, exclude=exclude_targets)
    revelation_nodes = _top_indices(scores["revelation_score"], int(cfg["top_k_revelation_nodes"]), mask=feasible_mask, exclude=exclude_targets)
    bridge_score = _node_feature(x, idx, "path_bottleneck", 0.0) + 0.6 * _node_feature(x, idx, "admin_bridge", 0.0) + 0.6 * _node_feature(x, idx, "lateral_bridge", 0.0)
    bridge_nodes = _top_indices(bridge_score, int(cfg["top_k_bridge_nodes"]), mask=feasible_mask, exclude=exclude_targets)
    centrality_score = bridge_score + 0.3 * _node_feature(x, idx, "service_surface", 0.0) + 0.2 * _node_feature(x, idx, "exposure", 0.0)
    centrality_nodes = _top_indices(centrality_score, int(cfg.get("top_k_centrality_nodes", 0)), mask=feasible_mask, exclude=exclude_targets)
    fused_nodes = _top_indices(scores["fused_score"], int(cfg["top_k_candidates"]), mask=feasible_mask, exclude=exclude_targets)
    seed_value = int(sample.get("episode_id", 0)) * 1009 + int(sample.get("t", 0)) * 9176 + n
    random_nodes = _stable_random_indices(n, int(cfg.get("top_k_random_nodes", 0)), feasible_mask, seed_value, exclude=exclude_targets)
    selected_candidates = _merge_unique(
        fused_nodes, threat_nodes, revelation_nodes, bridge_nodes, centrality_nodes, random_nodes,
        limit=int(cfg["top_k_candidates"]),
    )
    active_decoys = np.where(_node_feature(x, idx, "active_decoy", 0.0) > 0.4)[0].astype(int).tolist()
    recent_triggers = np.where(_node_feature(x, idx, "recent_trigger", 0.0) > 0.4)[0].astype(int).tolist()

    return {
        "targets": target_nodes,
        "entries": entry_nodes,
        "selected_candidates": selected_candidates,
        "high_threat": threat_nodes,
        "high_revelation": revelation_nodes,
        "bridges": bridge_nodes,
        "centrality": centrality_nodes,
        "random_exploration": random_nodes,
        "active_decoys": active_decoys,
        "recent_triggers": recent_triggers,
    }


def _compose_subgraph_nodes(sample: Dict[str, Any], x: np.ndarray, g: nx.Graph, scores: Dict[str, np.ndarray], seeds: Dict[str, List[int]], ctx: ExtractionContext) -> Set[int]:
    cfg = ctx.cfg["selection"]
    radius = int(cfg.get("neighborhood_radius", 1))
    k_paths = int(cfg.get("k_shortest_paths", 3))

    core: Set[int] = set()
    for group in seeds.values():
        core.update(int(i) for i in group)
    core.update(_k_shortest_path_nodes(g, seeds.get("entries", []), seeds.get("targets", []), k_paths))
    core.update(_k_shortest_path_nodes(g, seeds.get("selected_candidates", []), seeds.get("targets", []), k_paths))
    core.update(_khop_nodes(g, seeds.get("selected_candidates", []), radius))
    core.update(_khop_nodes(g, seeds.get("active_decoys", []) + seeds.get("recent_triggers", []), radius))

    if ctx.cfg["selection"].get("force_include_targets", True):
        core.update(seeds.get("targets", []))
    if ctx.cfg["selection"].get("force_include_top_candidate", True) and seeds.get("selected_candidates"):
        core.add(int(seeds["selected_candidates"][0]))

    max_nodes = int(cfg.get("max_strategic_nodes", 18))
    if len(core) <= max_nodes:
        return core

    priority = _priority_scores(x, sample, scores, ctx)
    must_keep = set(seeds.get("targets", []))
    if seeds.get("selected_candidates"):
        must_keep.add(int(seeds["selected_candidates"][0]))
    remaining = [i for i in core if i not in must_keep]
    remaining.sort(key=lambda i: (-float(priority[i]), i))
    selected = set(list(must_keep) + remaining[: max(0, max_nodes - len(must_keep))])
    return selected


def _transition_edge_features(src: int, dst: int, x: np.ndarray, scores: Dict[str, np.ndarray], ctx: ExtractionContext) -> Dict[str, float]:
    idx = ctx.feature_idx
    dst_exposure = _node_scalar(x, idx, dst, "exposure")
    dst_vuln = _node_scalar(x, idx, dst, "vulnerability")
    dst_alert = _node_scalar(x, idx, dst, "alert_intensity")
    dst_distance = _node_scalar(x, idx, dst, "distance_to_target")
    dst_surface = _node_scalar(x, idx, dst, "service_surface")
    dst_active_decoy = _node_scalar(x, idx, dst, "active_decoy")
    dst_recent_trigger = _node_scalar(x, idx, dst, "recent_trigger")
    dst_candidate = _node_scalar(x, idx, dst, "candidate")
    transition_base = clamp01(0.25 * dst_exposure + 0.20 * dst_vuln + 0.20 * dst_surface + 0.20 * dst_distance + 0.15 * dst_alert)
    deception_penalty = clamp01(0.45 * dst_active_decoy + 0.25 * dst_candidate + 0.25 * dst_recent_trigger + 0.25 * float(scores["revelation_score"][dst]))
    information_potential = clamp01(0.55 * float(scores["revelation_score"][dst]) + 0.25 * dst_recent_trigger + 0.20 * dst_alert)
    movement_cost = clamp01(1.0 - transition_base + 0.25 * deception_penalty)
    reliability = clamp01(0.65 + 0.25 * dst_surface - 0.15 * dst_active_decoy)
    return {
        "transition_base": transition_base,
        "deception_penalty": deception_penalty,
        "information_potential": information_potential,
        "movement_cost": movement_cost,
        "reliability": reliability,
    }


def _node_tags(i: int, seeds: Dict[str, List[int]], candidate_mask: np.ndarray, target_mask: np.ndarray, scores: Dict[str, np.ndarray]) -> List[str]:
    tags: List[str] = []
    for name, nodes in seeds.items():
        if i in set(nodes):
            tags.append(name)
    if candidate_mask[i] > 0.5:
        tags.append("candidate")
    if target_mask[i] > 0.5:
        tags.append("target")
    if scores["threat_score"][i] >= float(np.quantile(scores["threat_score"], 0.80)):
        tags.append("high_threat_score")
    if scores["revelation_score"][i] >= float(np.quantile(scores["revelation_score"], 0.80)):
        tags.append("high_revelation_score")
    return sorted(set(tags))


def _action_variant_specs(ctx: ExtractionContext) -> List[Dict[str, Any]]:
    cfg = ctx.cfg.get("action_generation", {})
    if not bool(cfg.get("enable_action_variants", True)):
        return [{"action_type": "place_or_retain_decoy", "decoy_type": "service", "exposure_level": "medium", "risk_mult": 1.0, "info_mult": 1.0, "delay_mult": 1.0, "cost_add": 0.0, "churn_mult": 1.0}]
    names = cfg.get("action_types") or ["place_high_interaction_decoy", "place_low_exposure_decoy", "credential_breadcrumb", "telemetry_trap", "path_perturbation", "rotate_or_retain_decoy"]
    library = {
        "place_high_interaction_decoy": {"decoy_type": "service", "exposure_level": "high", "risk_mult": 1.10, "info_mult": 1.10, "delay_mult": 0.90, "cost_add": 0.12, "churn_mult": 1.00},
        "place_low_exposure_decoy": {"decoy_type": "host", "exposure_level": "low", "risk_mult": 0.82, "info_mult": 1.25, "delay_mult": 1.05, "cost_add": 0.02, "churn_mult": 0.75},
        "credential_breadcrumb": {"decoy_type": "credential", "exposure_level": "medium", "risk_mult": 0.95, "info_mult": 1.35, "delay_mult": 0.95, "cost_add": 0.06, "churn_mult": 0.85},
        "telemetry_trap": {"decoy_type": "telemetry", "exposure_level": "low", "risk_mult": 0.70, "info_mult": 1.55, "delay_mult": 0.85, "cost_add": 0.01, "churn_mult": 0.55},
        "path_perturbation": {"decoy_type": "path", "exposure_level": "medium", "risk_mult": 0.90, "info_mult": 0.85, "delay_mult": 1.45, "cost_add": 0.08, "churn_mult": 0.90},
        "rotate_or_retain_decoy": {"decoy_type": "rotation", "exposure_level": "medium", "risk_mult": 0.88, "info_mult": 1.05, "delay_mult": 1.10, "cost_add": -0.04, "churn_mult": 0.25},
    }
    out: List[Dict[str, Any]] = []
    for name in names:
        spec = dict(library.get(str(name), library["place_high_interaction_decoy"]))
        spec["action_type"] = str(name)
        out.append(spec)
    return out


def _make_action_candidates(
    sample: Dict[str, Any],
    x: np.ndarray,
    node_ids: List[str],
    sub_nodes: Set[int],
    seeds: Dict[str, List[int]],
    scores: Dict[str, np.ndarray],
    belief_entropy: float,
    ctx: ExtractionContext,
) -> List[Dict[str, Any]]:
    idx = ctx.feature_idx
    w = ctx.cfg.get("scoring", {}).get("action_weights", {})
    top_nodes = [i for i in seeds.get("selected_candidates", []) if i in sub_nodes]
    variants = _action_variant_specs(ctx)
    max_actions = int(ctx.cfg.get("action_generation", {}).get("max_actions_per_graph", len(top_nodes) * max(1, len(variants))))
    diversity_penalty = float(ctx.cfg.get("action_generation", {}).get("mmr_diversity_penalty", 0.0))
    raw_actions: List[Dict[str, Any]] = []
    for node_rank, i in enumerate(top_nodes, start=1):
        bridge = max(_node_scalar(x, idx, i, "path_bottleneck"), _node_scalar(x, idx, i, "admin_bridge"), _node_scalar(x, idx, i, "lateral_bridge"))
        near_target = _node_scalar(x, idx, i, "distance_to_target")
        trigger = _node_scalar(x, idx, i, "recent_trigger")
        lateral = _node_scalar(x, idx, i, "lateral_bridge")
        criticality = _node_scalar(x, idx, i, "criticality")
        exposure = _node_scalar(x, idx, i, "exposure")
        active_decoy = _node_scalar(x, idx, i, "active_decoy")
        base_risk = clamp01(
            float(w.get("risk_proxy_threat", 0.55)) * float(scores["threat_score"][i])
            + float(w.get("risk_proxy_near_target", 0.25)) * near_target
            + float(w.get("risk_proxy_bridge", 0.20)) * bridge
        )
        base_info = clamp01(
            float(w.get("info_proxy_revelation", 0.55)) * float(scores["revelation_score"][i])
            + float(w.get("info_proxy_belief_entropy", 0.25)) * belief_entropy
            + float(w.get("info_proxy_trigger", 0.20)) * trigger
        )
        base_delay = clamp01(
            float(w.get("delay_proxy_bottleneck", 0.45)) * bridge
            + float(w.get("delay_proxy_path", 0.35)) * near_target
            + float(w.get("delay_proxy_lateral", 0.20)) * lateral
        )
        base_cost = clamp01(
            0.10
            + float(w.get("cost_criticality", 0.40)) * criticality
            + float(w.get("cost_exposure", 0.25)) * exposure
            + float(w.get("cost_active_decoy", -0.20)) * active_decoy
        )
        source_tags = []
        for name in ["high_threat", "high_revelation", "bridges", "centrality", "random_exploration", "active_decoys", "recent_triggers"]:
            if i in set(seeds.get(name, [])):
                source_tags.append(name)
        for spec_rank, spec in enumerate(variants, start=1):
            action_type = str(spec["action_type"])
            risk_proxy = clamp01(base_risk * float(spec.get("risk_mult", 1.0)))
            info_proxy = clamp01(base_info * float(spec.get("info_mult", 1.0)))
            delay_proxy = clamp01(base_delay * float(spec.get("delay_mult", 1.0)))
            operational_cost = clamp01(base_cost + float(spec.get("cost_add", 0.0)))
            churn_proxy = clamp01((1.0 - active_decoy) * float(spec.get("churn_mult", 1.0)))
            score_for_diversity = 0.45 * risk_proxy + 0.35 * info_proxy + 0.20 * delay_proxy - 0.10 * operational_cost
            raw_actions.append(
                {
                    "action_id": f"ep{int(sample['episode_id']):04d}_t{int(sample['t']):03d}_{action_type}_{node_ids[i].replace(':', '_')}",
                    "episode_id": int(sample["episode_id"]),
                    "t": int(sample["t"]),
                    "split": str(sample.get("split", "unknown")),
                    "rank": int(len(raw_actions) + 1),
                    "node_rank": int(node_rank),
                    "variant_rank": int(spec_rank),
                    "node_index": int(i),
                    "node_id": node_ids[i],
                    "action_type": action_type,
                    "decoy_type_hint": str(spec.get("decoy_type", "service")),
                    "exposure_level_hint": str(spec.get("exposure_level", "medium")),
                    "candidate_source_tags": ";".join(sorted(source_tags)) if source_tags else "fused",
                    "risk_proxy": risk_proxy,
                    "info_proxy": info_proxy,
                    "delay_proxy": delay_proxy,
                    "operational_cost": operational_cost,
                    "churn_proxy": churn_proxy,
                    "threat_score": float(scores["threat_score"][i]),
                    "revelation_score": float(scores["revelation_score"][i]),
                    "fused_score": float(scores["fused_score"][i]),
                    "candidate_score": float(scores["candidate_score"][i]),
                    "belief_entropy": float(belief_entropy),
                    "action_diversity_score": float(score_for_diversity),
                    "phase4_input_ready": True,
                }
            )
    # MMR-like pruning across node/action type to keep the table tractable while retaining diversity.
    selected: List[Dict[str, Any]] = []
    used_nodes: Dict[str, int] = {}
    used_types: Dict[str, int] = {}
    raw_actions.sort(key=lambda a: (-float(a["action_diversity_score"]), int(a["node_rank"]), int(a["variant_rank"])))
    while raw_actions and len(selected) < max_actions:
        best_idx = 0
        best_val = -1e18
        for idx_a, action in enumerate(raw_actions):
            node_pen = used_nodes.get(str(action["node_id"]), 0)
            type_pen = used_types.get(str(action["action_type"]), 0)
            val = float(action["action_diversity_score"]) - diversity_penalty * (node_pen + 0.5 * type_pen)
            if val > best_val:
                best_val = val
                best_idx = idx_a
        chosen = raw_actions.pop(best_idx)
        chosen["rank"] = len(selected) + 1
        selected.append(chosen)
        used_nodes[str(chosen["node_id"])] = used_nodes.get(str(chosen["node_id"]), 0) + 1
        used_types[str(chosen["action_type"])] = used_types.get(str(chosen["action_type"]), 0) + 1
    return selected


def extract_one(sample: Dict[str, Any], ctx: ExtractionContext) -> Dict[str, Any]:
    node_ids = [str(n) for n in sample["node_ids"]]
    x = tensor_to_numpy(sample["x"])
    edge_index = tensor_to_numpy(sample["edge_index"], dtype=np.int64)
    n = len(node_ids)
    g = _build_nx_graph(edge_index, n)
    scores = get_scores_for_sample(sample, ctx.pred_lookup, allow_proxy_fallback=ctx.allow_proxy_fallback)
    seeds = _select_seed_nodes(sample, x, g, scores, ctx)
    sub_nodes = _compose_subgraph_nodes(sample, x, g, scores, seeds, ctx)
    strategic_order = sorted(sub_nodes)
    strategic_id = {old: new for new, old in enumerate(strategic_order)}
    candidate_mask = tensor_to_numpy(sample.get("candidate_mask", np.zeros(n)))
    target_mask = tensor_to_numpy(sample.get("target_mask", np.zeros(n)))
    belief = tensor_to_numpy(sample.get("belief", np.array([])))
    factor_entropy = _belief_factor_entropies(belief, ctx.cfg)
    belief_entropy = float(factor_entropy.get("mean", 0.0))

    nodes = []
    for old in strategic_order:
        node_features = {
            name: _node_scalar(x, ctx.feature_idx, old, name)
            for name in [
                "criticality",
                "exposure",
                "vulnerability",
                "alert_intensity",
                "path_bottleneck",
                "distance_to_target",
                "active_decoy",
                "recent_trigger",
                "admin_bridge",
                "lateral_bridge",
                "service_surface",
            ]
        }
        nodes.append(
            {
                "strategic_index": int(strategic_id[old]),
                "original_index": int(old),
                "node_id": node_ids[old],
                "role": _role_from_features(x, ctx.feature_idx, old),
                "tags": _node_tags(old, seeds, candidate_mask, target_mask, scores),
                "scores": {k: float(v[old]) for k, v in scores.items()},
                "candidate_mask": float(candidate_mask[old]),
                "target_mask": float(target_mask[old]),
                "features": node_features,
            }
        )

    edges = []
    for u, v in g.edges():
        if u in sub_nodes and v in sub_nodes:
            for src, dst in [(u, v), (v, u)]:
                edges.append(
                    {
                        "source": int(strategic_id[src]),
                        "target": int(strategic_id[dst]),
                        "source_node_id": node_ids[src],
                        "target_node_id": node_ids[dst],
                        "features": _transition_edge_features(src, dst, x, scores, ctx),
                    }
                )

    actions = _make_action_candidates(sample, x, node_ids, sub_nodes, seeds, scores, belief_entropy, ctx)
    graph_obj = {
        "graph_id": f"ep{int(sample['episode_id']):04d}_t{int(sample['t']):03d}",
        "episode_id": int(sample["episode_id"]),
        "t": int(sample["t"]),
        "split": str(sample.get("split", "unknown")),
        "original_num_nodes": int(n),
        "original_num_edges_undirected": int(g.number_of_edges()),
        "strategic_num_nodes": int(len(nodes)),
        "strategic_num_edges_directed": int(len(edges)),
        "belief_vector": belief.tolist(),
        "belief_entropy": factor_entropy,
        "seed_nodes": {k: [node_ids[i] for i in vals] for k, vals in seeds.items()},
        "nodes": nodes,
        "edges": edges,
        "candidate_actions": actions,
        "notes": "Strategic subgraph extracted from Phase 2 GNN scores and action-context features. Intended as Phase 4 Game/AMC evaluator input.",
    }
    return graph_obj


def make_context(cfg: Dict[str, Any], metadata: Dict[str, Any], pred_lookup: Dict[Tuple[int, int, str], Dict[str, float]]) -> ExtractionContext:
    feature_names = list(metadata.get("node_features") or cfg.get("features", {}).get("node_features", []))
    return ExtractionContext(
        cfg=cfg,
        feature_names=feature_names,
        feature_idx=get_feature_index(feature_names),
        pred_lookup=pred_lookup,
        allow_proxy_fallback=bool(cfg.get("prediction_fallback", {}).get("allow_proxy_fallback", False)),
    )
