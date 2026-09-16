from __future__ import annotations

import argparse

import pandas as pd

from .utils import load_yaml, resolve_path, write_json


def summarize(config_path: str) -> None:
    cfg = load_yaml(config_path)
    policy_path = resolve_path(cfg["paths"]["policy_summary_csv"])
    episode_path = resolve_path(cfg["paths"]["episode_summary_csv"])
    rollout_path = resolve_path(cfg["paths"]["rollout_csv"])
    recal_path = resolve_path(cfg["paths"]["recalibration_log_csv"])

    policy = pd.read_csv(policy_path)
    episode = pd.read_csv(episode_path)
    rollout = pd.read_csv(rollout_path)
    recal = pd.read_csv(recal_path) if recal_path.exists() else pd.DataFrame()

    proposed = policy[policy["policy"] == "proposed_value_policy"]
    oracle = policy[policy["policy"] == "amc_oracle"]
    no_dec = policy[policy["policy"] == "no_deception"]

    report = {
        "num_policies": int(policy["policy"].nunique()),
        "num_episodes": int(rollout["episode_id"].nunique()),
        "num_snapshots": int(rollout["graph_id"].nunique()),
        "policy_ranking_by_utility": policy[["policy", "mean_utility", "mean_target_reach_reduction", "mean_info_gain", "mean_delay_gain", "mean_utility_gap_to_oracle"]].to_dict(orient="records"),
        "proposed_vs_oracle": None,
        "proposed_vs_no_deception": None,
        "recalibration_error_means": {},
    }

    if len(proposed) and len(oracle):
        p = proposed.iloc[0]
        o = oracle.iloc[0]
        report["proposed_vs_oracle"] = {
            "utility_ratio": float(p["mean_utility"] / o["mean_utility"]) if abs(float(o["mean_utility"])) > 1e-12 else None,
            "target_reach_reduction_gap": float(o["mean_target_reach_reduction"] - p["mean_target_reach_reduction"]),
            "utility_gap": float(o["mean_utility"] - p["mean_utility"]),
            "cost_saving": float(o["mean_operational_cost"] - p["mean_operational_cost"]),
            "churn_saving": float(o["mean_churn"] - p["mean_churn"]),
        }
    if len(proposed) and len(no_dec):
        p = proposed.iloc[0]
        n = no_dec.iloc[0]
        report["proposed_vs_no_deception"] = {
            "target_reach_reduction_gain": float(p["mean_target_reach_reduction"] - n["mean_target_reach_reduction"]),
            "target_probability_drop": float(n["mean_action_target_prob"] - p["mean_action_target_prob"]),
            "info_gain_gain": float(p["mean_info_gain"] - n["mean_info_gain"]),
            "delay_gain_gain": float(p["mean_delay_gain"] - n["mean_delay_gain"]),
        }
    if len(recal):
        for col in [c for c in recal.columns if c.endswith("error_true_minus_pred")]:
            report["recalibration_error_means"][col] = float(pd.to_numeric(recal[col], errors="coerce").mean())

    write_json(report, cfg["paths"]["evaluation_summary_json"])

    print("\n=== Phase 6 Integrated Evaluation Summary ===")
    print(policy.to_string(index=False))
    print(f"\nDetailed JSON summary written to: {resolve_path(cfg['paths']['evaluation_summary_json'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    args = parser.parse_args()
    summarize(args.config)


if __name__ == "__main__":
    main()
