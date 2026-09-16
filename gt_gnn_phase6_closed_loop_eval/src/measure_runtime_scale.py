from __future__ import annotations

import argparse
import time
from typing import Any

import pandas as pd

from .closed_loop import merge_phase4_phase5, run_closed_loop
from .utils import ensure_parent, load_yaml, read_csv_required, resolve_path, write_json


def measure(config_path: str, repeats: int = 3) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    phase4 = read_csv_required(cfg["paths"]["phase4_value_training_csv"], "Phase 4/5 AMC outcome training data")
    phase5 = read_csv_required(cfg["paths"]["phase5_predictions_csv"], "Phase 5 value/policy predictions")
    df = merge_phase4_phase5(phase4, phase5, cfg)
    stats_path = cfg["paths"].get("phase3_graph_stats_csv")
    stats = pd.read_csv(resolve_path(stats_path)) if stats_path and resolve_path(stats_path).exists() else pd.DataFrame()

    timings = []
    for r in range(int(repeats)):
        t0 = time.perf_counter()
        rollout, _, _ = run_closed_loop(df, cfg)
        elapsed = time.perf_counter() - t0
        timings.append({"repeat": r, "elapsed_seconds": elapsed, "rollout_rows": len(rollout), "snapshots": rollout["graph_id"].nunique() if len(rollout) else 0})
    timing_df = pd.DataFrame(timings)
    if len(stats) and "graph_id" in stats.columns:
        # Use actual Phase 3 graph stats only; no proxy rows in this measured output.
        actual = stats.drop_duplicates("graph_id").copy()
        thresholds = cfg.get("paper_experiments", {}).get("realistic_ad_graph", {}).get("scale_thresholds", {})
        large_min = float(thresholds.get("large_min_nodes", 1000))
        med_min = float(thresholds.get("medium_min_nodes", 100))
        nodes = pd.to_numeric(actual.get("original_num_nodes", 0), errors="coerce").fillna(0.0)
        actual["scale_bucket"] = nodes.map(lambda n: "large" if n >= large_min else ("medium" if n >= med_min else "small"))
        scale_summary = actual.groupby("scale_bucket").agg(
            graphs=("graph_id", "nunique"),
            mean_original_nodes=("original_num_nodes", "mean"),
            mean_original_edges=("original_num_edges_undirected", "mean"),
            mean_strategic_nodes=("strategic_num_nodes", "mean"),
            mean_candidate_actions=("num_candidate_actions", "mean"),
            compression_ratio=("compression_ratio_nodes", "mean"),
        ).reset_index()
    else:
        scale_summary = pd.DataFrame([{"scale_bucket": "unknown", "graphs": df["graph_id"].nunique() if "graph_id" in df.columns else 0}])
    scale_summary["measured_total_runtime_seconds_mean"] = float(timing_df["elapsed_seconds"].mean())
    scale_summary["measured_total_runtime_seconds_std"] = float(timing_df["elapsed_seconds"].std(ddof=0))
    scale_summary["measured_rows_per_second_mean"] = float(timing_df["rollout_rows"].mean() / max(timing_df["elapsed_seconds"].mean(), 1e-9))
    out_csv = resolve_path(cfg["paths"].get("measured_scale_summary_csv", "outputs/phase6_measured_scale_summary.csv"))
    ensure_parent(out_csv)
    scale_summary.to_csv(out_csv, index=False)
    summary = {"repeats": int(repeats), "timings": timings, "measured_scale_summary_csv": str(out_csv), "notes": "Measured on actual Phase 3 graph stats only; proxy scale rows disabled for paper main table."}
    write_json(summary, "outputs/phase6_measured_scale_runtime.json")
    print(scale_summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    measure(args.config, args.repeats)


if __name__ == "__main__":
    main()
