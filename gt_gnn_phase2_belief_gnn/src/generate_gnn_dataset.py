from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .utils import entropy_np, ensure_parent, load_yaml, normalize01, set_seed, write_json

FACTOR_KEYS = ["goal", "skill", "stealth", "decoy_awareness"]


def package_root_from_config(config_path: str) -> Path:
    p = Path(config_path).resolve()
    if p.parent.name == "config":
        return p.parent.parent
    return Path.cwd()


def one_hot_smooth(label: str, labels: List[str], smoothing: float) -> List[float]:
    n = len(labels)
    probs = np.full(n, smoothing / max(1, n - 1), dtype=np.float32)
    idx = labels.index(label)
    probs[idx] = 1.0 - smoothing
    return probs.tolist()


def make_fallback_phase1_rows(cfg: Dict, rng: np.random.Generator) -> pd.DataFrame:
    labels = cfg["labels"]
    data_cfg = cfg["data"]
    rows = []
    episode_count = int(data_cfg["fallback_num_episodes"])
    steps = int(data_cfg["fallback_steps_per_episode"])
    split_names = ["train", "val", "test"]
    split_probs = [0.70, 0.15, 0.15]
    for ep in range(episode_count):
        split = str(rng.choice(split_names, p=split_probs))
        sampled = {k: str(rng.choice(labels[k])) for k in FACTOR_KEYS}
        for t in range(steps):
            progress = t / max(1, steps - 1)
            rows.append(
                {
                    "episode_id": ep,
                    "t": t,
                    "split": split,
                    "goal_label": sampled["goal"],
                    "skill_label": sampled["skill"],
                    "stealth_label": sampled["stealth"],
                    "decoy_awareness_label": sampled["decoy_awareness"],
                    "failed_login_count": float(rng.uniform(0.0, 1.0)),
                    "lateral_auth_count": float(rng.uniform(0.0, 1.0) * (0.3 + progress)),
                    "service_probe_count": float(rng.uniform(0.0, 1.0) * (1.1 - 0.35 * progress)),
                    "privileged_edge_touch": float(rng.uniform(0.0, 1.0) * progress),
                    "admin_path_touch": float(rng.uniform(0.0, 1.0) * progress),
                    "decoy_touch_count": float(rng.uniform(0.0, 1.0)),
                    "decoy_avoidance_signal": float(rng.uniform(0.0, 1.0)),
                    "dns_entropy": float(rng.uniform(0.0, 1.0)),
                    "burstiness": float(rng.uniform(0.0, 1.0)),
                    "smb_rdp_count": float(rng.uniform(0.0, 1.0)),
                    "process_anomaly_count": float(rng.uniform(0.0, 1.0)),
                    "target_proximity_alert": float(rng.uniform(0.0, 1.0) * progress),
                    "prev_decoy_count": float(rng.uniform(0.1, 0.8)),
                    "decoy_exposure_level": float(rng.uniform(0.1, 1.0)),
                    "recent_trigger_count": float(rng.uniform(0.0, 1.0)),
                    "time_since_last_trigger": float(rng.uniform(0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def attach_belief_columns(df: pd.DataFrame, cfg: Dict, root: Path) -> pd.DataFrame:
    pred_path = root / cfg["paths"]["phase1_predictions_csv"]
    labels = cfg["labels"]
    out = df.copy()

    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        prob_cols = [c for c in pred.columns if c.startswith("p_")]
        keep = ["episode_id", "t"] + prob_cols
        out = out.merge(pred[keep], on=["episode_id", "t"], how="left")
        print(f"Loaded Phase 1 posterior probabilities from {pred_path}")
    else:
        print(f"Phase 1 prediction file not found at {pred_path}; using smoothed one-hot labels.")

    smoothing = float(cfg["belief"].get("smoothing_if_no_predictions", 0.06))
    for key in FACTOR_KEYS:
        for label in labels[key]:
            col = f"p_{key}_{label}"
            if col not in out.columns:
                out[col] = np.nan

        missing = out[[f"p_{key}_{label}" for label in labels[key]]].isna().any(axis=1)
        if missing.any():
            label_col = f"{key}_label"
            true_col = f"true_{key}"
            if label_col not in out.columns and true_col in out.columns:
                out[label_col] = out[true_col]
            if label_col not in out.columns:
                # fallback if neither labels nor predictions are available
                out[label_col] = labels[key][0]
            for idx in out.index[missing]:
                probs = one_hot_smooth(str(out.loc[idx, label_col]), labels[key], smoothing)
                for j, label in enumerate(labels[key]):
                    out.loc[idx, f"p_{key}_{label}"] = probs[j]
    return out


def load_phase1_rows(cfg: Dict, root: Path, rng: np.random.Generator) -> pd.DataFrame:
    dataset_path = root / cfg["paths"]["phase1_dataset_csv"]
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        print(f"Loaded Phase 1 dataset rows from {dataset_path}")
    else:
        print(f"Phase 1 dataset not found at {dataset_path}; generating fallback rows.")
        df = make_fallback_phase1_rows(cfg, rng)
    if "split" not in df.columns:
        df = df.copy()
        rng_values = rng.uniform(size=len(df))
        df["split"] = np.where(rng_values < 0.70, "train", np.where(rng_values < 0.85, "val", "test"))
    return attach_belief_columns(df, cfg, root)


def belief_vector_from_row(row: pd.Series, cfg: Dict) -> Tuple[np.ndarray, Dict[str, float]]:
    parts = []
    entropies = {}
    for key in FACTOR_KEYS:
        vals = np.array([float(row[f"p_{key}_{label}"]) for label in cfg["labels"][key]], dtype=np.float32)
        vals = vals / max(float(vals.sum()), 1e-8)
        parts.append(vals)
        entropies[key] = entropy_np(vals) / np.log(len(vals))
    return np.concatenate(parts).astype(np.float32), entropies


def build_episode_graph(ep: int, cfg: Dict) -> Dict:
    seed = int(cfg["seed"]) + ep * 104729
    rng = np.random.default_rng(seed)
    n_min = int(cfg["data"]["min_nodes"])
    n_max = int(cfg["data"]["max_nodes"])
    n = int(rng.integers(n_min, n_max + 1))

    node_ids = [f"host:ws{i:02d}" for i in range(n)]
    roles = ["workstation"] * n
    roles[0] = "dc"
    roles[1] = "server"
    roles[2] = "server"
    for i in range(3, min(n, 7)):
        roles[i] = "service"
    for i in range(max(7, n - 2), n):
        roles[i] = "honeypot"

    target_mask = np.zeros(n, dtype=np.float32)
    target_mask[0] = 1.0
    if n > 2:
        target_mask[1] = 1.0

    candidate_mask = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if target_mask[i] == 0 and roles[i] in {"workstation", "server", "service", "honeypot"}:
            if rng.uniform() < float(cfg["data"]["candidate_ratio"]) or roles[i] == "honeypot":
                candidate_mask[i] = 1.0
    if candidate_mask.sum() < 3:
        eligible = [i for i in range(n) if target_mask[i] == 0]
        for i in eligible[:3]:
            candidate_mask[i] = 1.0

    g = nx.Graph()
    g.add_nodes_from(range(n))
    # Core AD-like paths: workstations/services -> servers -> DC.
    for i in range(3, n):
        parent = int(rng.choice([1, 2, 3, 4, 5] if n > 6 else [1, 2]))
        parent = min(parent, n - 1)
        g.add_edge(i, parent)
    g.add_edge(1, 0)
    g.add_edge(2, 0)
    g.add_edge(3, 1)
    if n > 4:
        g.add_edge(4, 2)
    # Extra lateral/admin edges.
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform() < float(cfg["data"]["edge_density_extra"]):
                g.add_edge(i, j)

    lengths = []
    for i in range(n):
        try:
            d = min(nx.shortest_path_length(g, source=i, target=0), nx.shortest_path_length(g, source=i, target=1))
        except nx.NetworkXNoPath:
            d = n
        lengths.append(d)
    dist = normalize01(np.array(lengths, dtype=np.float32))
    # distance_to_target is closeness: high means near target.
    distance_to_target = 1.0 - dist
    bottleneck = np.array(list(nx.betweenness_centrality(g, normalized=True).values()), dtype=np.float32)
    bottleneck = normalize01(bottleneck)

    criticality = np.zeros(n, dtype=np.float32)
    exposure = np.zeros(n, dtype=np.float32)
    vulnerability = np.zeros(n, dtype=np.float32)
    admin_bridge = np.zeros(n, dtype=np.float32)
    lateral_bridge = np.zeros(n, dtype=np.float32)
    service_surface = np.zeros(n, dtype=np.float32)

    for i, role in enumerate(roles):
        if role == "dc":
            criticality[i], exposure[i], vulnerability[i] = 1.0, 0.25, 0.45
            admin_bridge[i] = 1.0
        elif role == "server":
            criticality[i], exposure[i], vulnerability[i] = rng.uniform(0.65, 0.95), rng.uniform(0.35, 0.75), rng.uniform(0.35, 0.85)
            admin_bridge[i] = rng.uniform(0.45, 0.95)
            service_surface[i] = rng.uniform(0.4, 0.8)
        elif role == "service":
            criticality[i], exposure[i], vulnerability[i] = rng.uniform(0.35, 0.75), rng.uniform(0.55, 1.0), rng.uniform(0.35, 0.9)
            service_surface[i] = rng.uniform(0.65, 1.0)
            lateral_bridge[i] = rng.uniform(0.25, 0.75)
        elif role == "honeypot":
            criticality[i], exposure[i], vulnerability[i] = rng.uniform(0.1, 0.35), rng.uniform(0.75, 1.0), rng.uniform(0.1, 0.35)
            service_surface[i] = rng.uniform(0.65, 1.0)
        else:
            criticality[i], exposure[i], vulnerability[i] = rng.uniform(0.1, 0.55), rng.uniform(0.2, 0.75), rng.uniform(0.1, 0.75)
            lateral_bridge[i] = rng.uniform(0.1, 0.75) * bottleneck[i]

    role_map = {
        "workstation": [1, 0, 0, 0, 0],
        "server": [0, 1, 0, 0, 0],
        "dc": [0, 0, 1, 0, 0],
        "service": [0, 0, 0, 1, 0],
        "honeypot": [0, 0, 0, 0, 1],
    }
    static = np.zeros((n, len(cfg["node_features"])), dtype=np.float32)
    for i, role in enumerate(roles):
        values = {
            "role_workstation": role_map[role][0],
            "role_server": role_map[role][1],
            "role_dc": role_map[role][2],
            "role_service": role_map[role][3],
            "role_honeypot": role_map[role][4],
            "criticality": criticality[i],
            "exposure": exposure[i],
            "vulnerability": vulnerability[i],
            "alert_intensity": 0.0,
            "path_bottleneck": bottleneck[i],
            "distance_to_target": distance_to_target[i],
            "active_decoy": 0.0,
            "recent_trigger": 0.0,
            "admin_bridge": admin_bridge[i],
            "lateral_bridge": lateral_bridge[i],
            "service_surface": service_surface[i],
            "candidate": candidate_mask[i],
            "target": target_mask[i],
        }
        static[i] = np.array([values[name] for name in cfg["node_features"]], dtype=np.float32)

    edges = []
    for u, v in g.edges():
        edges.append((u, v))
        edges.append((v, u))
    edge_index = np.array(edges, dtype=np.int64).T
    return {
        "node_ids": node_ids,
        "roles": roles,
        "edge_index": edge_index,
        "static_features": static,
        "candidate_mask": candidate_mask,
        "target_mask": target_mask,
    }


def row_value(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        v = float(row.get(name, default))
    except Exception:
        v = default
    if np.isnan(v):
        v = default
    return float(np.clip(v, 0.0, 1.0))


def build_snapshot_sample(row: pd.Series, graph: Dict, cfg: Dict, rng: np.random.Generator) -> Dict:
    x = graph["static_features"].copy()
    n = x.shape[0]
    belief, ent = belief_vector_from_row(row, cfg)
    labels = cfg["labels"]
    offset_goal = 0
    offset_skill = len(labels["goal"])
    offset_stealth = offset_skill + len(labels["skill"])
    offset_awareness = offset_stealth + len(labels["stealth"])

    p_goal = {name: belief[offset_goal + i] for i, name in enumerate(labels["goal"])}
    p_skill = {name: belief[offset_skill + i] for i, name in enumerate(labels["skill"])}
    p_stealth = {name: belief[offset_stealth + i] for i, name in enumerate(labels["stealth"])}
    p_aw = {name: belief[offset_awareness + i] for i, name in enumerate(labels["decoy_awareness"])}
    uncertainty = float(np.mean(list(ent.values())))

    feature_names = cfg["node_features"]
    idx = {name: i for i, name in enumerate(feature_names)}

    global_alert = (
        0.22 * row_value(row, "failed_login_count")
        + 0.33 * row_value(row, "lateral_auth_count")
        + 0.22 * row_value(row, "service_probe_count")
        + 0.28 * row_value(row, "privileged_edge_touch")
        + 0.25 * row_value(row, "target_proximity_alert")
        + 0.18 * row_value(row, "process_anomaly_count")
    )
    prev_decoy_count = row_value(row, "prev_decoy_count", 0.25)
    decoy_exposure = row_value(row, "decoy_exposure_level", 0.50)
    recent_trigger_global = row_value(row, "recent_trigger_count", 0.0)

    candidate_indices = np.where(graph["candidate_mask"] > 0.5)[0]
    rng.shuffle(candidate_indices)
    active_count = int(np.clip(round(prev_decoy_count * 5), 1, max(1, len(candidate_indices))))
    active = candidate_indices[:active_count] if len(candidate_indices) else []

    # Dynamic node features.
    for i in range(n):
        x[i, idx["alert_intensity"]] = np.clip(
            global_alert
            + 0.18 * x[i, idx["path_bottleneck"]]
            + 0.15 * x[i, idx["distance_to_target"]]
            + 0.10 * rng.normal(),
            0.0,
            1.0,
        )
        if i in active:
            x[i, idx["active_decoy"]] = 1.0
        x[i, idx["recent_trigger"]] = np.clip(
            recent_trigger_global * (0.5 + 0.5 * x[i, idx["active_decoy"]]) * decoy_exposure
            + 0.05 * rng.normal(),
            0.0,
            1.0,
        )

    criticality = x[:, idx["criticality"]]
    exposure = x[:, idx["exposure"]]
    vuln = x[:, idx["vulnerability"]]
    bottleneck = x[:, idx["path_bottleneck"]]
    near_target = x[:, idx["distance_to_target"]]
    admin_bridge = x[:, idx["admin_bridge"]]
    lateral_bridge = x[:, idx["lateral_bridge"]]
    surface = x[:, idx["service_surface"]]
    candidate = graph["candidate_mask"]
    target = graph["target_mask"]
    honeypot = x[:, idx["role_honeypot"]]
    active_decoy = x[:, idx["active_decoy"]]
    alert = x[:, idx["alert_intensity"]]

    threat_raw = (
        0.80 * criticality
        + 0.75 * bottleneck
        + 0.60 * near_target
        + 0.55 * vuln
        + 0.45 * alert
        + 0.55 * p_goal["dc_access"] * admin_bridge
        + 0.45 * p_goal["cred_collection"] * lateral_bridge
        + 0.35 * p_goal["service_disruption"] * surface
        + 0.30 * p_skill["high"] * (admin_bridge + near_target)
        + 0.20 * p_stealth["stealthy"] * bottleneck
        - 0.45 * target
        - 0.18 * active_decoy
    )
    reveal_raw = (
        0.60 * uncertainty * exposure
        + 0.50 * uncertainty * surface
        + 0.35 * candidate
        + 0.40 * p_stealth["noisy"] * exposure
        + 0.38 * p_goal["exploration"] * surface
        + 0.34 * p_aw["unaware"] * honeypot
        + 0.25 * p_aw["suspecting"] * (surface + bottleneck)
        - 0.35 * p_aw["aware"] * honeypot
        + 0.22 * active_decoy
        + 0.18 * x[:, idx["recent_trigger"]]
    )
    noise_std = float(cfg["data"].get("noise_std", 0.045))
    threat_score = np.clip(threat_raw + rng.normal(0, noise_std, size=n), 0.0, None)
    reveal_score = np.clip(reveal_raw + rng.normal(0, noise_std, size=n), 0.0, None)

    def quantile_label(score: np.ndarray, q: float) -> np.ndarray:
        cand_scores = score[candidate > 0.5]
        if len(cand_scores) == 0:
            return np.zeros_like(score, dtype=np.float32)
        threshold = float(np.quantile(cand_scores, q))
        y = ((score >= threshold) & (candidate > 0.5)).astype(np.float32)
        if y.sum() == 0:
            y[int(np.argmax(score * candidate))] = 1.0
        return y

    threat_y = quantile_label(threat_score, float(cfg["data"]["threat_positive_quantile"]))
    reveal_y = quantile_label(reveal_score, float(cfg["data"]["revelation_positive_quantile"]))
    candidate_y = np.clip((0.65 * threat_y + 0.35 * reveal_y) > 0, 0, 1).astype(np.float32)
    gate_y = np.full(n, np.clip(1.0 - uncertainty, 0.0, 1.0), dtype=np.float32)
    # More threat weight for high-criticality candidates, more revelation weight for uncertain/edge decoys.
    gate_y = np.clip(0.65 * gate_y + 0.25 * criticality + 0.10 * near_target - 0.15 * honeypot, 0.0, 1.0).astype(np.float32)

    return {
        "episode_id": int(row["episode_id"]),
        "t": int(row["t"]),
        "split": str(row["split"]),
        "node_ids": graph["node_ids"],
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
        "belief": torch.tensor(belief, dtype=torch.float32),
        "threat_y": torch.tensor(threat_y, dtype=torch.float32),
        "revelation_y": torch.tensor(reveal_y, dtype=torch.float32),
        "candidate_y": torch.tensor(candidate_y, dtype=torch.float32),
        "gate_y": torch.tensor(gate_y, dtype=torch.float32),
        "candidate_mask": torch.tensor(candidate, dtype=torch.float32),
        "target_mask": torch.tensor(target, dtype=torch.float32),
    }


def generate(config_path: str) -> Dict:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    set_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    rows = load_phase1_rows(cfg, root, rng)

    graphs: Dict[int, Dict] = {}
    samples = []
    for _, row in rows.sort_values(["episode_id", "t"]).iterrows():
        ep = int(row["episode_id"])
        if ep not in graphs:
            graphs[ep] = build_episode_graph(ep, cfg)
        sample_rng = np.random.default_rng(int(cfg["seed"]) + ep * 99991 + int(row["t"]) * 197)
        samples.append(build_snapshot_sample(row, graphs[ep], cfg, sample_rng))

    out_path = root / cfg["paths"]["gnn_dataset_pt"]
    ensure_parent(str(out_path))
    metadata = {
        "num_samples": len(samples),
        "split_counts": dict(pd.Series([s["split"] for s in samples]).value_counts()),
        "node_features": cfg["node_features"],
        "belief_dim": int(sum(len(cfg["labels"][k]) for k in FACTOR_KEYS)),
        "factor_keys": FACTOR_KEYS,
        "labels": cfg["labels"],
        "notes": "Phase 2 dataset for belief-conditioned dual-head GNN. Labels are synthetic/proxy targets for ACSAC case-study pretraining.",
    }
    torch.save({"samples": samples, "metadata": metadata}, out_path)
    write_json(metadata, str(root / cfg["paths"]["metadata_json"]))
    print(f"Wrote {len(samples):,} graph snapshots to {out_path}")
    print(f"Split counts: {metadata['split_counts']}")
    print(f"Belief dim: {metadata['belief_dim']}; node feature dim: {len(cfg['node_features'])}")
    return {"samples": samples, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gnn_config.yaml")
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
