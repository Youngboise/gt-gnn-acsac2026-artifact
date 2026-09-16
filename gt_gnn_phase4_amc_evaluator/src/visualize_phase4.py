from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .utils import load_yaml, package_root_from_config


def _save_hist(df: pd.DataFrame, col: str, out_path: Path, title: str) -> None:
    if col not in df.columns or df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(df[col].astype(float), bins=30)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    if x not in df.columns or y not in df.columns or df.empty:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(df[x].astype(float), df[y].astype(float), s=12, alpha=0.65)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def visualize(config_path: str) -> Dict[str, str]:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    csv_path = root / cfg["paths"]["action_outcomes_csv"]
    out_dir = root / cfg["paths"].get("visualization_dir", "outputs/visualizations")
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase 4 action outcomes not found: {csv_path}. Run evaluate_amc_evaluator first.")
    df = pd.read_csv(csv_path)
    outputs: Dict[str, str] = {}
    plots = [
        ("utility_amc_raw", "AMC utility distribution"),
        ("relative_risk_reduction", "Relative risk reduction distribution"),
        ("expected_info_gain", "Expected information gain distribution"),
        ("delay_gain", "Delay/diversion gain distribution"),
        ("action_decoy_absorption_prob", "Decoy absorption probability distribution"),
    ]
    for col, title in plots:
        out = out_dir / f"hist_{col}.png"
        _save_hist(df, col, out, title)
        outputs[col] = str(out)
    scatter_specs = [
        ("risk_proxy", "relative_risk_reduction", "risk_proxy_vs_amc_risk.png", "Phase 3 risk proxy vs AMC risk reduction"),
        ("info_proxy", "expected_info_gain", "info_proxy_vs_amc_info.png", "Phase 3 info proxy vs AMC info gain"),
        ("delay_proxy", "delay_gain", "delay_proxy_vs_amc_delay.png", "Phase 3 delay proxy vs AMC delay gain"),
        ("fused_score", "utility_amc_raw", "fused_score_vs_utility.png", "GNN fused score vs AMC utility"),
    ]
    for x, y, filename, title in scatter_specs:
        out = out_dir / filename
        _save_scatter(df, x, y, out, title)
        outputs[filename] = str(out)
    print(f"Wrote Phase 4 visualizations to: {out_dir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/amc_config.yaml")
    args = parser.parse_args()
    visualize(args.config)


if __name__ == "__main__":
    main()
