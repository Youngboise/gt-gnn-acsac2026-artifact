from __future__ import annotations

import argparse

from .closed_loop import merge_phase4_phase5
from .paper_experiments import run_paper_experiment_suite, write_paper_experiment_outputs
from .utils import load_yaml, read_csv_required


def run(config_path: str) -> None:
    cfg = load_yaml(config_path)
    phase4 = read_csv_required(cfg["paths"]["phase4_value_training_csv"], "Phase 4/5 AMC outcome training data")
    phase5 = read_csv_required(cfg["paths"]["phase5_predictions_csv"], "Phase 5 value/policy predictions")
    df = merge_phase4_phase5(phase4, phase5, cfg)
    print(f"Paper experiment input dataframe: {len(df):,} rows, {len(df.columns)} columns")

    outputs = run_paper_experiment_suite(df, cfg)
    write_paper_experiment_outputs(outputs, cfg)

    print("\nPaper experiment suite complete:")
    print(f"- scale rows: {len(outputs.scale_summary):,}")
    print(f"- robustness scenarios: {len(outputs.robustness_summary):,}")
    print(f"- ablations: {len(outputs.ablation_summary):,}")
    print(f"- held-out attacker scenarios: {len(outputs.heldout_attacker_summary):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
