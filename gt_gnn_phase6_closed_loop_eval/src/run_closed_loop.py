from __future__ import annotations

import argparse
from pathlib import Path

from .closed_loop import merge_phase4_phase5, run_closed_loop, summarize_by_policy, make_baseline_comparison
from .utils import load_yaml, read_csv_required, resolve_path, ensure_parent, write_json


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    phase4 = read_csv_required(cfg["paths"]["phase4_value_training_csv"], "Phase 4/5 AMC outcome training data")
    phase5 = read_csv_required(cfg["paths"]["phase5_predictions_csv"], "Phase 5 value/policy predictions")

    df = merge_phase4_phase5(phase4, phase5, cfg)
    print(f"Integrated dataframe: {len(df):,} rows, {len(df.columns)} columns")

    rollout, episode_summary, recalib = run_closed_loop(df, cfg)
    policy_summary = summarize_by_policy(rollout)
    baseline_comparison = make_baseline_comparison(policy_summary)

    outputs = {
        "rollout_csv": rollout,
        "episode_summary_csv": episode_summary,
        "policy_summary_csv": policy_summary,
        "baseline_comparison_csv": baseline_comparison,
        "recalibration_log_csv": recalib,
    }
    for path_key, frame in outputs.items():
        path = resolve_path(cfg["paths"][path_key])
        ensure_parent(path)
        frame.to_csv(path, index=False)
        print(f"Wrote {path_key}: {path} ({len(frame):,} rows)")

    best_policy = policy_summary.iloc[0].to_dict() if len(policy_summary) else {}
    proposed = policy_summary[policy_summary["policy"] == "proposed_value_policy"]
    summary = {
        "num_action_rows": int(len(df)),
        "num_rollout_rows": int(len(rollout)),
        "num_episodes": int(rollout["episode_id"].nunique()) if len(rollout) else 0,
        "num_snapshots": int(rollout["graph_id"].nunique()) if len(rollout) else 0,
        "best_policy_by_mean_utility": best_policy,
        "proposed_value_policy": proposed.iloc[0].to_dict() if len(proposed) else None,
        "available_policies": list(policy_summary["policy"]),
    }
    write_json(summary, cfg["paths"]["evaluation_summary_json"])
    print(f"Wrote evaluation summary: {resolve_path(cfg['paths']['evaluation_summary_json'])}")

    print("\nPolicy summary:")
    print(policy_summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
