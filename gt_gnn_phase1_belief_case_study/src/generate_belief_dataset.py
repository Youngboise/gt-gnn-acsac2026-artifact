from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_parent, load_yaml, set_seed, write_json


GOAL_EFFECTS = {
    "dc_access": {
        "lateral_auth_count": 1.15,
        "privileged_edge_touch": 0.95,
        "admin_path_touch": 1.10,
        "smb_rdp_count": 0.95,
        "target_proximity_alert": 1.25,
        "unique_dst_count": 0.65,
    },
    "cred_collection": {
        "failed_login_count": 0.75,
        "lateral_auth_count": 0.95,
        "successful_login_ratio": 0.75,
        "unique_user_count": 1.20,
        "process_anomaly_count": 0.55,
    },
    "service_disruption": {
        "service_probe_count": 1.20,
        "dns_entropy": 0.95,
        "burstiness": 1.05,
        "process_anomaly_count": 0.85,
        "unique_dst_count": 1.10,
    },
    "exploration": {
        "service_probe_count": 0.95,
        "unique_dst_count": 1.25,
        "dns_entropy": 0.75,
        "failed_login_count": 0.55,
        "target_proximity_alert": 0.35,
    },
}

SKILL_EFFECTS = {
    "low": {
        "failed_login_count": 0.95,
        "service_probe_count": 0.80,
        "burstiness": 0.65,
        "successful_login_ratio": -0.25,
    },
    "medium": {
        "lateral_auth_count": 0.55,
        "smb_rdp_count": 0.50,
        "successful_login_ratio": 0.20,
    },
    "high": {
        "privileged_edge_touch": 0.65,
        "admin_path_touch": 0.70,
        "target_proximity_alert": 0.60,
        "successful_login_ratio": 0.45,
        "failed_login_count": -0.25,
    },
}

STEALTH_EFFECTS = {
    "noisy": {
        "failed_login_count": 0.80,
        "service_probe_count": 0.90,
        "burstiness": 0.90,
        "unique_dst_count": 0.60,
        "dns_entropy": 0.55,
    },
    "balanced": {
        "lateral_auth_count": 0.30,
        "smb_rdp_count": 0.25,
    },
    "stealthy": {
        "failed_login_count": -0.30,
        "service_probe_count": -0.25,
        "burstiness": -0.35,
        "post_decoy_silence": 0.50,
        "successful_login_ratio": 0.30,
    },
}

DECOY_AWARENESS_EFFECTS = {
    "unaware": {
        "decoy_touch_count": 0.80,
        "recent_trigger_count": 0.65,
        "decoy_avoidance_signal": -0.15,
        "post_decoy_silence": -0.10,
    },
    "suspecting": {
        "decoy_touch_count": 0.25,
        "decoy_avoidance_signal": 0.45,
        "post_decoy_silence": 0.35,
    },
    "aware": {
        "decoy_touch_count": -0.20,
        "decoy_avoidance_signal": 0.95,
        "post_decoy_silence": 0.75,
        "recent_trigger_count": -0.15,
    },
}


def add_effects(base: Dict[str, float], effects: Dict[str, float]) -> None:
    for k, v in effects.items():
        base[k] = base.get(k, 0.0) + v


def sample_factor(labels: List[str], rng: np.random.Generator) -> str:
    # Mild imbalance makes the dataset more realistic but still learnable.
    raw = rng.uniform(0.75, 1.25, size=len(labels))
    probs = raw / raw.sum()
    return str(rng.choice(labels, p=probs))


def build_episode_features(
    *,
    features: List[str],
    goal: str,
    skill: str,
    stealth: str,
    decoy_awareness: str,
    steps: int,
    noise_std: float,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    proto = {name: 0.15 for name in features}
    add_effects(proto, GOAL_EFFECTS[goal])
    add_effects(proto, SKILL_EFFECTS[skill])
    add_effects(proto, STEALTH_EFFECTS[stealth])
    add_effects(proto, DECOY_AWARENESS_EFFECTS[decoy_awareness])

    rows: List[Dict[str, float]] = []
    prev_trigger = 0.0
    time_since_trigger = 1.0
    prev_decoy_count = float(rng.integers(1, 4))
    exposure = float(rng.uniform(0.25, 1.0))

    for t in range(steps):
        progress = t / max(1, steps - 1)
        row: Dict[str, float] = {}
        for f in features:
            trend = 0.0
            if f in {"target_proximity_alert", "admin_path_touch", "privileged_edge_touch"}:
                trend += 0.25 * progress
            if f in {"service_probe_count", "unique_dst_count"}:
                trend += 0.15 * (1.0 - progress)
            value = proto.get(f, 0.0) + trend + rng.normal(0.0, noise_std)
            row[f] = float(max(0.0, value))

        # Explicit action-conditioned variables.
        # These will later be replaced by real decoy policy history.
        row["prev_decoy_count"] = prev_decoy_count / 5.0
        row["decoy_exposure_level"] = exposure
        row["recent_trigger_count"] = max(0.0, prev_trigger + rng.normal(0, noise_std))
        row["time_since_last_trigger"] = time_since_trigger

        # Coupled trigger logic: unaware attackers touch exposed decoys more often.
        decoy_touch = row.get("decoy_touch_count", 0.0) * exposure
        triggered = float(decoy_touch > rng.uniform(0.25, 1.25))
        if triggered:
            prev_trigger = min(1.0, prev_trigger + 0.35)
            time_since_trigger = 0.0
        else:
            prev_trigger = max(0.0, prev_trigger * 0.85)
            time_since_trigger = min(1.0, time_since_trigger + 1.0 / steps)

        # Attacker awareness can increase after repeated trigger events.
        if decoy_awareness in {"suspecting", "aware"}:
            row["decoy_avoidance_signal"] = row.get("decoy_avoidance_signal", 0.0) + 0.25 * prev_trigger
            row["post_decoy_silence"] = row.get("post_decoy_silence", 0.0) + 0.20 * prev_trigger

        # Normalize into a practical 0..1-ish range.
        for f in features:
            row[f] = float(np.clip(row[f] / 2.5, 0.0, 1.0))

        rows.append(row)

    return rows


def assign_splits(num_episodes: int, train_ratio: float, val_ratio: float, rng: np.random.Generator) -> Dict[int, str]:
    ids = np.arange(num_episodes)
    rng.shuffle(ids)
    n_train = int(num_episodes * train_ratio)
    n_val = int(num_episodes * val_ratio)
    split_map = {}
    for i, episode_id in enumerate(ids):
        if i < n_train:
            split_map[int(episode_id)] = "train"
        elif i < n_train + n_val:
            split_map[int(episode_id)] = "val"
        else:
            split_map[int(episode_id)] = "test"
    return split_map


def generate(config_path: str) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    labels = cfg["labels"]
    features = cfg["features"]
    data_cfg = cfg["data"]
    split_map = assign_splits(
        int(data_cfg["num_episodes"]),
        float(data_cfg["train_ratio"]),
        float(data_cfg["val_ratio"]),
        rng,
    )

    rows = []
    for episode_id in range(int(data_cfg["num_episodes"])):
        goal = sample_factor(labels["goal"], rng)
        skill = sample_factor(labels["skill"], rng)
        stealth = sample_factor(labels["stealth"], rng)
        decoy_awareness = sample_factor(labels["decoy_awareness"], rng)
        steps = int(rng.integers(int(data_cfg["min_steps"]), int(data_cfg["max_steps"]) + 1))

        episode_features = build_episode_features(
            features=features,
            goal=goal,
            skill=skill,
            stealth=stealth,
            decoy_awareness=decoy_awareness,
            steps=steps,
            noise_std=float(data_cfg["noise_std"]),
            rng=rng,
        )

        for t, feat in enumerate(episode_features):
            rows.append(
                {
                    "episode_id": episode_id,
                    "t": t,
                    "split": split_map[episode_id],
                    "goal_label": goal,
                    "skill_label": skill,
                    "stealth_label": stealth,
                    "decoy_awareness_label": decoy_awareness,
                    **feat,
                }
            )

    df = pd.DataFrame(rows)
    out_csv = cfg["paths"]["dataset_csv"]
    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False)

    metadata = {
        "features": features,
        "labels": labels,
        "num_rows": int(len(df)),
        "num_episodes": int(data_cfg["num_episodes"]),
        "window_size": int(data_cfg["window_size"]),
        "split_counts": df["split"].value_counts().to_dict(),
    }
    write_json(metadata, cfg["paths"]["metadata_json"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/belief_config.yaml")
    args = parser.parse_args()
    df = generate(args.config)
    print(f"Wrote dataset with {len(df):,} snapshots and {df['episode_id'].nunique():,} episodes.")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
