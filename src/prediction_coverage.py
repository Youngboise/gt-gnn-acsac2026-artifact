from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .generate_gnn_dataset import package_root_from_config
from .phase2_coverage_utils import expected_node_rows
from .utils import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether Phase 2 inference covers every snapshot/node needed by Phase 3.")
    parser.add_argument("--config", default="config/gnn_config.yaml")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default="outputs/phase2_prediction_coverage.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = package_root_from_config(args.config)
    pred_path = Path(args.predictions) if args.predictions else root / cfg["paths"]["predictions_csv"]
    pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    expected = expected_node_rows(root / cfg["paths"]["gnn_dataset_pt"])

    if len(pred):
        keys_pred = set(zip(pred["episode_id"].astype(int), pred["t"].astype(int), pred["node_id"].astype(str)))
    else:
        keys_pred = set()
    missing = [k for k in expected["keys"] if k not in keys_pred]
    report = {
        "prediction_csv": str(pred_path),
        "expected_node_rows": int(len(expected["keys"])),
        "predicted_node_rows": int(len(keys_pred)),
        "missing_node_rows": int(len(missing)),
        "coverage": float(1.0 - len(missing) / max(len(expected["keys"]), 1)),
        "expected_snapshots_by_split": expected["snapshots_by_split"],
        "missing_examples": [{"episode_id": e, "t": t, "node_id": n} for e, t, n in missing[:25]],
    }
    write_json(report, str(root / args.output))
    print(report)


if __name__ == "__main__":
    main()
