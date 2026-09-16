from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import load_yaml, resolve_path, write_json


def _read(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = resolve_path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _row(df: pd.DataFrame, policy: str) -> dict[str, Any]:
    if df.empty or "policy" not in df.columns:
        return {}
    hit = df[df["policy"].astype(str) == policy]
    return hit.iloc[0].to_dict() if len(hit) else {}


def _f(row: dict[str, Any], key: str) -> float | None:
    if key not in row or pd.isna(row[key]):
        return None
    return float(row[key])


def validate(config_path: str) -> None:
    cfg = load_yaml(config_path)
    paths = cfg.get("paths", {})
    policy = _read(paths.get("policy_summary_csv"))
    ablation = _read(paths.get("ablation_summary_csv"))
    robust = _read(paths.get("robustness_summary_csv"))
    large = _read(paths.get("large_scale_summary_csv"))

    proposed = _row(policy, "proposed_value_policy")
    checks: list[dict[str, Any]] = []

    for baseline in ["static_low_cost", "random_placement", "phase3_fused", "centrality_based"]:
        b = _row(policy, baseline)
        pv, bv = _f(proposed, "mean_utility"), _f(b, "mean_utility")
        checks.append({
            "check": f"proposed_beats_{baseline}",
            "passed": bool(pv is not None and bv is not None and pv > bv),
            "proposed": pv,
            "baseline": bv,
            "margin": None if pv is None or bv is None else pv - bv,
        })

    if not ablation.empty and "utility_drop_vs_full" in ablation.columns:
        positive = pd.to_numeric(ablation["utility_drop_vs_full"], errors="coerce") > 0
        checks.append({
            "check": "most_ablations_reduce_utility",
            "passed": bool(positive.mean() >= 0.65),
            "fraction_positive_drop": float(positive.mean()),
            "num_ablations": int(len(ablation)),
        })

    if not robust.empty and "proposed_minus_static_utility" in robust.columns:
        diff = pd.to_numeric(robust["proposed_minus_static_utility"], errors="coerce")
        checks.append({
            "check": "robustness_proposed_beats_static_in_most_scenarios",
            "passed": bool((diff > 0).mean() >= 0.80),
            "fraction_positive": float((diff > 0).mean()),
            "min_margin": float(diff.min()) if len(diff) else None,
        })

    if not large.empty and "node_count" in large.columns:
        max_nodes = int(pd.to_numeric(large["node_count"], errors="coerce").max())
        checks.append({
            "check": "large_scale_benchmark_reaches_10000_nodes",
            "passed": bool(max_nodes >= 10000),
            "max_nodes": max_nodes,
        })

    report = {
        "checks": checks,
        "passed": bool(all(c.get("passed", False) for c in checks)) if checks else False,
        "notes": [
            "This is a sanity report, not a substitute for paper judgment.",
            "Oracle baselines are intentionally excluded from pass/fail checks.",
        ],
    }
    out = resolve_path("outputs/phase6_acsac_readiness_report.json")
    write_json(report, out)
    print(pd.DataFrame(checks).to_string(index=False))
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    args = parser.parse_args()
    validate(args.config)


if __name__ == "__main__":
    main()
