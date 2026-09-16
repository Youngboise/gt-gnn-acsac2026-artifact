from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .utils import load_config, repo_root, resolve_path, write_json


ABLATIONS = [
    "wo_belief",
    "wo_information_gain",
    "wo_amc",
    "wo_stackelberg_response",
    "wo_action_conditioned_snapshot",
    "wo_entity_alignment",
]


def _first(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((c for c in names if c in df.columns), None)


def _norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def make_variant(df: pd.DataFrame, variant: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    cfg_update: dict[str, Any] = {"notes": []}
    rng = np.random.default_rng(seed)
    risk = _first(out, ["target_reach_reduction", "relative_risk_reduction", "risk_proxy"])
    info = _first(out, ["expected_info_gain", "information_gain", "info_proxy"])
    delay = _first(out, ["delay_gain", "expected_delay_gain", "delay_proxy"])
    utility = _first(out, ["utility_amc_norm", "utility_counterfactual_norm", "utility_amc_raw", "utility"])

    if variant == "wo_belief":
        cfg_update["exclude_extra"] = [c for c in out.columns if "belief" in c.lower() or "entropy" in c.lower() or c.startswith("p_goal") or c.startswith("p_skill") or c.startswith("p_stealth")]
        cfg_update["notes"].append("Excludes belief/entropy posterior features from Phase 5 inputs and retrains.")
    elif variant == "wo_information_gain":
        if info:
            out[info] = 0.0
        if utility and risk and delay:
            out[utility] = (0.70 * _norm(out[risk]) + 0.30 * _norm(out[delay])).clip(0, 1)
        cfg_update["loss_info"] = 0.0
        cfg_update["notes"].append("Sets I(a) target/loss to zero and retrains risk-delay policy.")
    elif variant == "wo_amc":
        # Replace counterfactual/AMC labels with pre-AMC proxies to measure label circularity and evaluator need.
        rp, ip, dp = _first(out, ["risk_proxy"]), _first(out, ["info_proxy"]), _first(out, ["delay_proxy"])
        if rp and risk:
            out[risk] = out[rp]
        if ip and info:
            out[info] = out[ip]
        if dp and delay:
            out[delay] = out[dp]
        if utility and risk and info and delay:
            out[utility] = (0.50 * _norm(out[risk]) + 0.25 * _norm(out[info]) + 0.25 * _norm(out[delay])).clip(0, 1)
        cfg_update["notes"].append("Uses Phase 3 proxy targets instead of AMC/counterfactual labels.")
    elif variant == "wo_stackelberg_response":
        if utility and risk:
            out[utility] = _norm(out[risk])
        cfg_update["notes"].append("Uses risk-only utility, removing attacker response trade-off.")
    elif variant == "wo_action_conditioned_snapshot":
        for c in [info, "revelation_score", "recent_trigger_count", "churn_proxy", "active_decoy"]:
            if c and c in out.columns:
                if "episode_id" in out.columns:
                    out[c] = out.groupby("episode_id")[c].transform("mean")
                else:
                    out[c] = out[c].mean()
        cfg_update["exclude_extra"] = ["churn_proxy", "active_decoy", "recent_trigger", "recent_trigger_count", "prev_decoy_count", "decoy_exposure_level"]
        cfg_update["notes"].append("Removes/shuffles action-conditioned observation features before retraining.")
    elif variant == "wo_entity_alignment":
        # Perturb a controlled subset of action rows to emulate entity mismatch between graph and event namespaces.
        if "node_id" in out.columns:
            mask = rng.random(len(out)) < 0.15
            shuffled = out.loc[mask, "node_id"].sample(frac=1.0, random_state=seed).to_numpy()
            out.loc[mask, "node_id"] = shuffled
        for c in ["threat_score", "revelation_score", "fused_score", "candidate_score"]:
            if c in out.columns:
                mask = rng.random(len(out)) < 0.15
                out.loc[mask, c] = out[c].sample(frac=1.0, random_state=seed + 1).to_numpy()[: mask.sum()]
        cfg_update["notes"].append("Injects 15% node/score mismatch to test entity alignment sensitivity.")
    else:
        raise ValueError(variant)
    return out, cfg_update


def write_variant_config(base_cfg: dict[str, Any], variant: str, csv_path: Path, cfg_update: dict[str, Any], out_dir: Path) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg["paths"]["phase5_training_csv"] = str(csv_path.relative_to(out_dir.parents[1]) if False else csv_path)
    cfg["paths"]["processed_dataset_pt"] = f"data/ablations/{variant}/processed_dataset.pt"
    cfg["paths"]["model_checkpoint"] = f"outputs/ablations/{variant}/value_policy_head_best.pt"
    cfg["paths"]["train_history_csv"] = f"outputs/ablations/{variant}/train_history.csv"
    cfg["paths"]["predictions_csv"] = f"outputs/ablations/{variant}/predictions.csv"
    cfg["paths"]["evaluation_summary_json"] = f"outputs/ablations/{variant}/evaluation_summary_test.json"
    if cfg_update.get("exclude_extra"):
        cfg["data"].setdefault("exclude_feature_columns", [])
        cfg["data"]["exclude_feature_columns"] = sorted(set(cfg["data"]["exclude_feature_columns"] + cfg_update["exclude_extra"]))
    if "loss_info" in cfg_update:
        cfg.setdefault("loss_weights", {})["info"] = float(cfg_update["loss_info"])
    cfg.setdefault("ablation", {})["name"] = variant
    cfg["ablation"]["notes"] = cfg_update.get("notes", [])
    cfg_path = out_dir / f"{variant}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and optionally run true retraining ablations for Phase 5.")
    parser.add_argument("--config", default="config/value_config.yaml")
    parser.add_argument("--variants", nargs="*", default=ABLATIONS)
    parser.add_argument("--run", action="store_true", help="Actually run prepare/train/evaluate for each variant. Without this, only writes variant CSV/config files.")
    args = parser.parse_args()
    root = repo_root()
    base_cfg = load_config(args.config)
    source_csv = resolve_path(root, base_cfg["paths"]["phase5_training_csv"])
    df = pd.read_csv(source_csv)
    variant_cfg_dir = root / "config" / "ablations"
    manifest: dict[str, Any] = {"source_csv": str(source_csv), "variants": []}
    for i, variant in enumerate(args.variants):
        vdf, update = make_variant(df, variant, int(base_cfg["project"].get("seed", 42)) + i)
        csv_path = root / "data" / "ablations" / variant / "phase5_training_data.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        vdf.to_csv(csv_path, index=False)
        cfg_path = write_variant_config(base_cfg, variant, csv_path, update, variant_cfg_dir)
        record = {"variant": variant, "csv": str(csv_path), "config": str(cfg_path), "notes": update.get("notes", [])}
        manifest["variants"].append(record)
        print(record)
        if args.run:
            rel_cfg = str(cfg_path.relative_to(root))
            subprocess.run([sys.executable, "-m", "src.prepare_value_dataset", "--config", rel_cfg], cwd=root, check=True)
            subprocess.run([sys.executable, "-m", "src.train_value_policy", "--config", rel_cfg], cwd=root, check=True)
            subprocess.run([sys.executable, "-m", "src.evaluate_value_policy", "--config", rel_cfg, "--split", "test"], cwd=root, check=True)
    write_json(manifest, root / "outputs" / "ablations" / "phase5_retraining_ablation_manifest.json")


if __name__ == "__main__":
    main()
