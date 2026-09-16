from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import load_yaml, resolve_path, ensure_dir


def _save_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path, dpi: int) -> None:
    data = df.sort_values(y, ascending=False)
    fig = plt.figure(figsize=(10, 5))
    plt.bar(data[x].astype(str), data[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {path}")


def _save_line_entropy(rollout: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    focus = rollout[rollout["policy"].isin(["proposed_value_policy", "amc_oracle", "pred_risk_only", "no_deception"])]
    if len(focus) == 0:
        return
    # Average entropy trajectory by t index. Use rank within episode to avoid non-numeric t issues.
    focus = focus.copy()
    focus["step_idx"] = focus.groupby(["policy", "episode_id"]).cumcount()
    traj = focus.groupby(["policy", "step_idx"])["belief_entropy_after"].mean().reset_index()

    fig = plt.figure(figsize=(10, 5))
    for policy, sub in traj.groupby("policy"):
        plt.plot(sub["step_idx"], sub["belief_entropy_after"], marker="o", label=policy)
    plt.title("Closed-loop belief entropy trajectory")
    plt.xlabel("Episode step")
    plt.ylabel("Mean belief entropy after action")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "belief_entropy_trajectory.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {path}")


def _save_tradeoff(policy: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(policy["mean_target_reach_reduction"], policy["mean_info_gain"])
    for _, r in policy.iterrows():
        plt.annotate(str(r["policy"]), (r["mean_target_reach_reduction"], r["mean_info_gain"]), fontsize=8)
    plt.title("Risk reduction vs information gain")
    plt.xlabel("Mean target reach reduction")
    plt.ylabel("Mean information gain")
    plt.tight_layout()
    path = out_dir / "risk_info_tradeoff.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {path}")


def _save_optional_paper_experiment_plots(cfg: dict, out_dir: Path, dpi: int) -> None:
    paths = cfg.get("paths", {})

    robustness_path = resolve_path(paths.get("robustness_summary_csv", "")) if paths.get("robustness_summary_csv") else None
    if robustness_path and robustness_path.exists():
        robust = pd.read_csv(robustness_path)
        if len(robust) and {"scenario", "proposed_minus_static_utility"}.issubset(robust.columns):
            fig = plt.figure(figsize=(11, 5))
            ordered = robust.sort_values("proposed_minus_static_utility", ascending=False)
            plt.bar(ordered["scenario"].astype(str), ordered["proposed_minus_static_utility"])
            plt.title("Robustness: proposed utility advantage over static baseline")
            plt.ylabel("Δ mean utility")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            path = out_dir / "robustness_proposed_vs_static.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            print(f"Wrote {path}")

    ablation_path = resolve_path(paths.get("ablation_summary_csv", "")) if paths.get("ablation_summary_csv") else None
    if ablation_path and ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        if len(ablation) and {"ablation", "utility_drop_vs_full"}.issubset(ablation.columns):
            fig = plt.figure(figsize=(10, 5))
            ordered = ablation.sort_values("utility_drop_vs_full", ascending=False)
            plt.bar(ordered["ablation"].astype(str), ordered["utility_drop_vs_full"])
            plt.title("Ablation: utility drop vs full GT-GNN")
            plt.ylabel("Utility drop")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            path = out_dir / "ablation_utility_drop.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            print(f"Wrote {path}")

    heldout_path = resolve_path(paths.get("heldout_attacker_summary_csv", "")) if paths.get("heldout_attacker_summary_csv") else None
    if heldout_path and heldout_path.exists():
        heldout = pd.read_csv(heldout_path)
        if len(heldout) and {"heldout_scenario", "proposed_minus_static_utility"}.issubset(heldout.columns):
            fig = plt.figure(figsize=(9, 5))
            plt.bar(heldout["heldout_scenario"].astype(str), heldout["proposed_minus_static_utility"])
            plt.title("Held-out attacker dynamics: proposed utility advantage")
            plt.ylabel("Δ mean utility")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            path = out_dir / "heldout_attacker_proposed_vs_static.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            print(f"Wrote {path}")

    scale_path = resolve_path(paths.get("scale_experiment_csv", "")) if paths.get("scale_experiment_csv") else None
    if scale_path and scale_path.exists():
        scale = pd.read_csv(scale_path)
        focus = scale[scale["policy"] == "proposed_value_policy"] if "policy" in scale.columns else pd.DataFrame()
        if len(focus) and {"scale_bucket", "estimated_runtime_ms"}.issubset(focus.columns):
            order = {"small": 0, "medium": 1, "large": 2}
            focus = focus.copy()
            focus["_order"] = focus["scale_bucket"].map(order).fillna(99)
            focus = focus.sort_values("_order")
            fig = plt.figure(figsize=(8, 5))
            plt.plot(focus["scale_bucket"].astype(str), focus["estimated_runtime_ms"], marker="o")
            plt.title("Scale experiment: estimated runtime by graph scale")
            plt.xlabel("Scale bucket")
            plt.ylabel("Estimated runtime (ms)")
            plt.tight_layout()
            path = out_dir / "scale_estimated_runtime.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            print(f"Wrote {path}")


def visualize(config_path: str) -> None:
    cfg = load_yaml(config_path)
    out_dir = resolve_path(cfg["paths"]["visualization_dir"])
    ensure_dir(out_dir)
    dpi = int(cfg.get("visualization", {}).get("dpi", 140))

    policy = pd.read_csv(resolve_path(cfg["paths"]["policy_summary_csv"]))
    rollout = pd.read_csv(resolve_path(cfg["paths"]["rollout_csv"]))

    _save_bar(
        policy,
        "policy",
        "mean_utility",
        "Mean closed-loop utility by policy",
        "Mean utility",
        out_dir / "mean_utility_by_policy.png",
        dpi,
    )
    _save_bar(
        policy,
        "policy",
        "mean_target_reach_reduction",
        "Mean target reach reduction by policy",
        "Mean target reach reduction",
        out_dir / "target_reach_reduction_by_policy.png",
        dpi,
    )
    _save_bar(
        policy,
        "policy",
        "mean_utility_gap_to_oracle",
        "Mean utility gap to AMC oracle",
        "Utility gap to oracle",
        out_dir / "utility_gap_to_oracle_by_policy.png",
        dpi,
    )
    _save_bar(
        policy,
        "policy",
        "mean_churn",
        "Mean policy churn by policy",
        "Mean churn",
        out_dir / "policy_churn_by_policy.png",
        dpi,
    )
    _save_line_entropy(rollout, out_dir, dpi)
    _save_tradeoff(policy, out_dir, dpi)
    _save_optional_paper_experiment_plots(cfg, out_dir, dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    args = parser.parse_args()
    visualize(args.config)


if __name__ == "__main__":
    main()
