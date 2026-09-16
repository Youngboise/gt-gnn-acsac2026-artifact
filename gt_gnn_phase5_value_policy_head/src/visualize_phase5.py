from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, load_config, repo_root, resolve_path


def visualize(config_path: str) -> None:
    root = repo_root()
    cfg = load_config(config_path)
    vis_dir = resolve_path(root, cfg["paths"]["visualization_dir"])
    ensure_dir(vis_dir)

    hist_path = resolve_path(root, cfg["paths"]["train_history_csv"])
    pred_path = resolve_path(root, cfg["paths"]["predictions_csv"])
    selected_path = resolve_path(root, cfg["paths"]["selected_actions_csv"])

    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        plt.figure(figsize=(8, 5))
        plt.plot(hist["epoch"], hist["train_total"], label="train total")
        plt.plot(hist["epoch"], hist["val_total"], label="val total")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Phase 5 Value/Policy Training Loss")
        plt.legend()
        plt.tight_layout()
        out = vis_dir / "training_loss.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved {out}")

    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        for target in ["risk", "info", "delay", "utility"]:
            true_col = f"true_{target}"
            pred_col = f"pred_{target}"
            if true_col in pred.columns and pred_col in pred.columns:
                plt.figure(figsize=(6, 6))
                plt.scatter(pred[true_col], pred[pred_col], s=12, alpha=0.65)
                lo = min(pred[true_col].min(), pred[pred_col].min())
                hi = max(pred[true_col].max(), pred[pred_col].max())
                plt.plot([lo, hi], [lo, hi], linestyle="--")
                plt.xlabel(f"True {target}")
                plt.ylabel(f"Predicted {target}")
                plt.title(f"True vs Predicted {target}")
                plt.tight_layout()
                out = vis_dir / f"true_vs_pred_{target}.png"
                plt.savefig(out, dpi=160)
                plt.close()
                print(f"Saved {out}")

        if "pred_best_action_prob" in pred.columns:
            plt.figure(figsize=(8, 5))
            pred["pred_best_action_prob"].hist(bins=30)
            plt.xlabel("Predicted best-action probability")
            plt.ylabel("Count")
            plt.title("Best-Action Probability Distribution")
            plt.tight_layout()
            out = vis_dir / "best_action_probability_histogram.png"
            plt.savefig(out, dpi=160)
            plt.close()
            print(f"Saved {out}")

    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        if "utility_gap_to_oracle" in selected.columns:
            plt.figure(figsize=(8, 5))
            selected["utility_gap_to_oracle"].dropna().hist(bins=30)
            plt.xlabel("Oracle utility - selected action utility")
            plt.ylabel("Count")
            plt.title("Policy Utility Gap")
            plt.tight_layout()
            out = vis_dir / "policy_utility_gap_histogram.png"
            plt.savefig(out, dpi=160)
            plt.close()
            print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/value_config.yaml")
    args = parser.parse_args()
    visualize(args.config)


if __name__ == "__main__":
    main()
