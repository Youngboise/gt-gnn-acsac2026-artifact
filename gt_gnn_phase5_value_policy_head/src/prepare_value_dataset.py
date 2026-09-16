from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .utils import load_config, repo_root, resolve_path, set_seed, write_json
from .value_dataset import prepared_to_dict, prepare_from_dataframe


def prepare(config_path: str) -> None:
    root = repo_root()
    cfg = load_config(config_path)
    set_seed(int(cfg["project"].get("seed", 42)))

    csv_path = resolve_path(root, cfg["paths"]["phase5_training_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Phase 5 training CSV not found: {csv_path}\n"
            "Check config/value_config.yaml paths.phase5_training_csv."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    prepared = prepare_from_dataframe(df, cfg, seed=int(cfg["project"].get("seed", 42)))
    obj = prepared_to_dict(prepared)

    out_pt = resolve_path(root, cfg["paths"]["processed_dataset_pt"])
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out_pt)

    write_json(prepared.feature_scaler, resolve_path(root, cfg["paths"]["feature_scaler_json"]))
    write_json(prepared.target_scaler, resolve_path(root, cfg["paths"]["target_scaler_json"]))

    split_counts = pd.Series(prepared.split).value_counts().to_dict()
    metadata = {
        "source_csv": str(csv_path),
        "num_rows": len(df),
        "num_features": len(prepared.feature_columns),
        "feature_columns": prepared.feature_columns,
        "target_columns": prepared.target_columns,
        "regression_targets": obj["regression_targets"],
        "group_column": prepared.group_column,
        "split_counts": split_counts,
    }
    write_json(metadata, resolve_path(root, cfg["paths"]["metadata_json"]))

    print(f"Loaded Phase 4 value-training rows from {csv_path}")
    print(f"Wrote processed dataset to {out_pt}")
    print(f"Feature columns: {len(prepared.feature_columns)}")
    print(f"Target columns: {prepared.target_columns}")
    print(f"Split counts: {split_counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
