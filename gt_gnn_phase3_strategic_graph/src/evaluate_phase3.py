from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .utils import load_yaml, package_root_from_config, read_jsonl, write_json


def evaluate(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    graph_path = root / cfg["paths"]["strategic_graphs_jsonl"]
    actions_path = root / cfg["paths"]["action_candidates_csv"]
    stats_path = root / cfg["paths"]["graph_stats_csv"]
    out_path = root / cfg["paths"]["evaluation_json"]

    graphs = read_jsonl(graph_path)
    actions = pd.read_csv(actions_path) if Path(actions_path).exists() and Path(actions_path).stat().st_size > 0 else pd.DataFrame()
    stats = pd.read_csv(stats_path) if Path(stats_path).exists() and Path(stats_path).stat().st_size > 0 else pd.DataFrame()

    summary: Dict[str, Any] = {
        "num_strategic_graphs": int(len(graphs)),
        "num_action_candidates": int(len(actions)),
    }
    if not stats.empty:
        summary.update(
            {
                "avg_original_nodes": float(stats["original_num_nodes"].mean()),
                "avg_strategic_nodes": float(stats["strategic_num_nodes"].mean()),
                "avg_node_compression_ratio": float(stats["compression_ratio_nodes"].mean()),
                "avg_candidate_actions_per_graph": float(stats["num_candidate_actions"].mean()),
                "min_candidate_actions_per_graph": int(stats["num_candidate_actions"].min()),
                "max_candidate_actions_per_graph": int(stats["num_candidate_actions"].max()),
                "avg_belief_entropy": float(stats["mean_belief_entropy"].mean()),
            }
        )
    if not actions.empty:
        summary.update(
            {
                "avg_risk_proxy": float(actions["risk_proxy"].mean()),
                "avg_info_proxy": float(actions["info_proxy"].mean()),
                "avg_delay_proxy": float(actions["delay_proxy"].mean()),
                "avg_operational_cost": float(actions["operational_cost"].mean()),
                "top_actions_preview": actions.sort_values(["graph_id", "rank"]).head(10).to_dict(orient="records"),
            }
        )

    write_json(summary, out_path)
    print("Phase 3 Evaluation Summary")
    print("--------------------------")
    for k, v in summary.items():
        if k == "top_actions_preview":
            continue
        print(f"{k}: {v}")
    print(f"Wrote evaluation summary to: {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/strategic_config.yaml")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
