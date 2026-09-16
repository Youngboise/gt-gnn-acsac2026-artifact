from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .phase2_io import build_prediction_lookup, filter_samples, load_phase2_bundle, load_predictions, prediction_coverage
from .strategic_graph import extract_one, make_context
from .utils import load_yaml, package_root_from_config, set_seed, write_json, write_jsonl


def extract(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    set_seed(int(cfg.get("seed", 17)))
    start = time.time()

    phase2_dataset_path = root / cfg["paths"]["phase2_dataset_pt"]
    pred_path = root / cfg["paths"]["phase2_predictions_csv"]

    samples, metadata = load_phase2_bundle(phase2_dataset_path)
    pred_df = load_predictions(pred_path)
    pred_lookup = build_prediction_lookup(pred_df)

    split = str(cfg.get("selection", {}).get("split", "all"))
    max_snapshots = cfg.get("selection", {}).get("max_snapshots", None)
    selected_samples = filter_samples(samples, split=split, max_snapshots=max_snapshots)

    coverage = prediction_coverage(selected_samples, pred_lookup)
    fallback_cfg = cfg.get("prediction_fallback", {})
    allow_fallback = bool(fallback_cfg.get("allow_proxy_fallback", False))
    min_cov = float(fallback_cfg.get("minimum_prediction_coverage", 0.99))
    if not allow_fallback and coverage["coverage"] < min_cov:
        raise RuntimeError(
            f"Phase 2 prediction coverage {coverage['coverage']:.4f} is below required {min_cov:.4f}. "
            "Run Phase 2 full inference before Phase 3. Missing examples: "
            f"{coverage['missing_examples'][:5]}"
        )
    ctx = make_context(cfg, metadata, pred_lookup)

    graphs: List[Dict[str, Any]] = []
    action_rows: List[Dict[str, Any]] = []
    stat_rows: List[Dict[str, Any]] = []

    for i, sample in enumerate(selected_samples):
        graph_obj = extract_one(sample, ctx)
        graphs.append(graph_obj)
        for action in graph_obj["candidate_actions"]:
            action_rows.append({"graph_id": graph_obj["graph_id"], **action})
        stat_rows.append(
            {
                "graph_id": graph_obj["graph_id"],
                "episode_id": graph_obj["episode_id"],
                "t": graph_obj["t"],
                "split": graph_obj["split"],
                "original_num_nodes": graph_obj["original_num_nodes"],
                "original_num_edges_undirected": graph_obj["original_num_edges_undirected"],
                "strategic_num_nodes": graph_obj["strategic_num_nodes"],
                "strategic_num_edges_directed": graph_obj["strategic_num_edges_directed"],
                "compression_ratio_nodes": graph_obj["strategic_num_nodes"] / max(1, graph_obj["original_num_nodes"]),
                "num_candidate_actions": len(graph_obj["candidate_actions"]),
                "mean_belief_entropy": graph_obj["belief_entropy"].get("mean", 0.0),
                "targets_included": len(graph_obj["seed_nodes"].get("targets", [])),
                "top_candidate": graph_obj["candidate_actions"][0]["node_id"] if graph_obj["candidate_actions"] else "",
            }
        )
        if (i + 1) % 100 == 0:
            print(f"Extracted {i + 1:,}/{len(selected_samples):,} strategic graphs...")

    strategic_jsonl = root / cfg["paths"]["strategic_graphs_jsonl"]
    action_jsonl = root / cfg["paths"]["action_inputs_jsonl"]
    actions_csv = root / cfg["paths"]["action_candidates_csv"]
    stats_csv = root / cfg["paths"]["graph_stats_csv"]
    metadata_json = root / cfg["paths"]["metadata_json"]

    write_jsonl(graphs, strategic_jsonl)
    # Phase 4 can read a compact action-focused JSONL where each line is one graph and candidate action list.
    write_jsonl(
        [
            {
                "graph_id": g["graph_id"],
                "episode_id": g["episode_id"],
                "t": g["t"],
                "split": g["split"],
                "belief_vector": g["belief_vector"],
                "belief_entropy": g["belief_entropy"],
                "candidate_actions": g["candidate_actions"],
            }
            for g in graphs
        ],
        action_jsonl,
    )
    actions_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(action_rows).to_csv(actions_csv, index=False)
    pd.DataFrame(stat_rows).to_csv(stats_csv, index=False)

    out_meta = {
        "phase": "Phase 3 - Strategic graph extraction",
        "source_phase2_dataset": str(phase2_dataset_path),
        "source_phase2_predictions": str(pred_path),
        "phase2_prediction_rows_available": int(0 if pred_df is None else len(pred_df)),
        "phase2_prediction_coverage": coverage,
        "allow_proxy_fallback": allow_fallback,
        "num_phase2_samples_total": int(len(samples)),
        "num_samples_extracted": int(len(selected_samples)),
        "split": split,
        "feature_names": ctx.feature_names,
        "output_files": {
            "strategic_graphs_jsonl": str(strategic_jsonl),
            "action_inputs_jsonl": str(action_jsonl),
            "action_candidates_csv": str(actions_csv),
            "graph_stats_csv": str(stats_csv),
        },
        "elapsed_seconds": round(time.time() - start, 4),
        "notes": "Strategic graph extraction is algorithmic: seed selection, k-shortest paths, k-hop expansion, candidate filtering, and transition/action feature construction.",
    }
    write_json(out_meta, metadata_json)

    print(f"Loaded Phase 2 samples: {len(samples):,} from {phase2_dataset_path}")
    if pred_df is None:
        print(f"Phase 2 predictions not found at {pred_path}.")
    else:
        print(f"Loaded Phase 2 prediction rows: {len(pred_df):,} from {pred_path}")
    print(f"Phase 2 prediction coverage: {coverage['coverage']:.4f} ({coverage['found_node_rows']:,}/{coverage['expected_node_rows']:,} node rows); proxy_fallback={allow_fallback}")
    print(f"Extracted strategic graphs: {len(graphs):,}")
    print(f"Wrote strategic graphs to: {strategic_jsonl}")
    print(f"Wrote Phase 4 action inputs to: {action_jsonl}")
    print(f"Wrote action candidate table to: {actions_csv}")
    print(f"Wrote graph stats to: {stats_csv}")
    return out_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/strategic_config.yaml")
    args = parser.parse_args()
    extract(args.config)


if __name__ == "__main__":
    main()
