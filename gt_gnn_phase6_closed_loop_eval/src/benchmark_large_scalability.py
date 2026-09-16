from __future__ import annotations

"""Measured large-scale scalability benchmark for ACSAC evaluation.

This benchmark does not extrapolate from the small case-study graphs.  It
constructs synthetic AD-style sparse graphs with increasing node counts, extracts
strategic subgraphs around entry/target/frontier nodes, generates heterogeneous
candidate actions, and runs an AMC-like absorbing-chain solve on the strategic
subgraph.  It reports wall-clock time and peak memory measured with tracemalloc.
"""

import argparse
import gc
import json
import math
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_parent, load_yaml, resolve_path, write_json


@dataclass
class BenchResult:
    node_count: int
    edge_count: int
    strategic_nodes: int
    strategic_edges: int
    candidate_actions: int
    repeat: int
    extraction_seconds: float
    scoring_seconds: float
    amc_seconds: float
    total_seconds: float
    peak_memory_mb: float
    target_absorption_mean: float
    decoy_absorption_mean: float
    expected_steps_mean: float


def _make_sparse_graph(n: int, avg_degree: int, rng: np.random.Generator) -> tuple[list[list[int]], int]:
    adj: list[set[int]] = [set() for _ in range(n)]
    # Ring edges ensure weak connectivity.
    for i in range(n):
        j = (i + 1) % n
        adj[i].add(j)
        adj[j].add(i)
    extra = max(0, int(n * avg_degree / 2) - n)
    src = rng.integers(0, n, size=extra)
    dst = rng.integers(0, n, size=extra)
    for a, b in zip(src, dst):
        a, b = int(a), int(b)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    adj_list = [sorted(x) for x in adj]
    m = sum(len(x) for x in adj_list) // 2
    return adj_list, m


def _multi_source_bfs(adj: list[list[int]], sources: np.ndarray, cap_depth: int = 8) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float32)
    dq: deque[int] = deque()
    for s in sources:
        si = int(s)
        dist[si] = 0.0
        dq.append(si)
    while dq:
        u = dq.popleft()
        if dist[u] >= cap_depth:
            continue
        for v in adj[u]:
            if not np.isfinite(dist[v]):
                dist[v] = dist[u] + 1.0
                dq.append(v)
    finite = np.isfinite(dist)
    if finite.any():
        max_f = float(dist[finite].max())
        dist[~finite] = max_f + 1.0
    else:
        dist[:] = cap_depth + 1
    return dist


def _extract_strategic_nodes(adj: list[list[int]], threat: np.ndarray, revelation: np.ndarray, entry: np.ndarray, target: np.ndarray, cap: int) -> np.ndarray:
    n = len(adj)
    d_entry = _multi_source_bfs(adj, entry)
    d_target = _multi_source_bfs(adj, target)
    bridge = 1.0 / (1.0 + d_entry + d_target)
    degree = np.array([len(x) for x in adj], dtype=np.float32)
    degree_norm = degree / max(float(degree.max()), 1.0)
    score = 0.33 * threat + 0.25 * revelation + 0.25 * bridge + 0.17 * degree_norm
    forced = np.unique(np.concatenate([entry, target]))
    k = min(max(cap, len(forced)), n)
    top = np.argpartition(-score, kth=k - 1)[:k]
    selected = np.unique(np.concatenate([forced, top]))
    if len(selected) > k:
        # Keep forced nodes plus highest-scoring remaining nodes.
        forced_set = set(map(int, forced))
        rest = [int(x) for x in selected if int(x) not in forced_set]
        rest = sorted(rest, key=lambda x: score[x], reverse=True)[: max(0, k - len(forced_set))]
        selected = np.array(sorted(list(forced_set) + rest), dtype=np.int64)
    return selected


def _strategic_edge_count(adj: list[list[int]], selected: np.ndarray) -> int:
    in_sel = np.zeros(len(adj), dtype=bool)
    in_sel[selected] = True
    count = 0
    for u in selected:
        count += sum(1 for v in adj[int(u)] if in_sel[v])
    return count // 2


def _generate_candidates(selected: np.ndarray, threat: np.ndarray, revelation: np.ndarray, delay: np.ndarray, cap: int) -> pd.DataFrame:
    score = 0.35 * threat[selected] + 0.35 * revelation[selected] + 0.30 * delay[selected]
    k = min(cap, len(selected))
    top_local = np.argpartition(-score, kth=k - 1)[:k]
    nodes = selected[top_local]
    rows = []
    action_types = [
        "place_high_interaction_decoy",
        "place_low_exposure_decoy",
        "credential_breadcrumb",
        "telemetry_trap",
        "path_perturbation",
        "rotate_or_retain_decoy",
    ]
    for node in nodes:
        for action_type in action_types:
            rows.append({"node": int(node), "action_type": action_type})
    return pd.DataFrame(rows)


def _amc_solve(strategic_nodes: np.ndarray, adj: list[list[int]], threat: np.ndarray, revelation: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    # Cap the AMC solve to keep large graph benchmark practical while still measuring
    # the cubic component on a realistic strategic subgraph size.
    nodes = strategic_nodes
    m = len(nodes)
    idx = {int(v): i for i, v in enumerate(nodes)}
    Q = np.zeros((m, m), dtype=np.float64)
    R = np.zeros((m, 3), dtype=np.float64)  # target, decoy, abort
    for u in nodes:
        ui = idx[int(u)]
        neigh = [v for v in adj[int(u)] if int(v) in idx]
        if neigh:
            raw = np.array([0.2 + 0.5 * threat[v] + 0.3 * revelation[v] for v in neigh], dtype=np.float64)
            probs = raw / max(raw.sum(), 1e-12)
            for v, p in zip(neigh, probs):
                Q[ui, idx[int(v)]] += 0.72 * float(p)
        R[ui, 0] = 0.10 + 0.20 * float(threat[int(u)])
        R[ui, 1] = 0.05 + 0.25 * float(revelation[int(u)])
        R[ui, 2] = 0.03 + 0.05 * float(rng.random())
        row_sum = Q[ui].sum() + R[ui].sum()
        if row_sum >= 0.98:
            Q[ui] *= 0.98 / row_sum
            R[ui] *= 0.98 / row_sum
    I = np.eye(m, dtype=np.float64)
    try:
        N = np.linalg.solve(I - Q, I)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    B = N @ R
    steps = N.sum(axis=1)
    return float(B[:, 0].mean()), float(B[:, 1].mean()), float(steps.mean())


def _one_run(n: int, avg_degree: int, strategic_cap: int, candidate_cap: int, repeat: int, seed: int) -> BenchResult:
    rng = np.random.default_rng(seed + n * 131 + repeat)
    tracemalloc.start()
    t0 = time.perf_counter()
    adj, edge_count = _make_sparse_graph(n, avg_degree, rng)
    threat = rng.beta(2.0, 5.0, size=n).astype(np.float32)
    revelation = rng.beta(2.5, 4.0, size=n).astype(np.float32)
    delay = rng.beta(2.0, 2.0, size=n).astype(np.float32)
    entry = rng.choice(n, size=max(2, min(12, n // 100)), replace=False)
    target = rng.choice(n, size=max(2, min(8, n // 250)), replace=False)
    strategic_nodes = _extract_strategic_nodes(adj, threat, revelation, entry, target, min(strategic_cap, n))
    strategic_edges = _strategic_edge_count(adj, strategic_nodes)
    extraction_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    candidates = _generate_candidates(strategic_nodes, threat, revelation, delay, candidate_cap)
    # Candidate scoring mimics the decision-layer feature fusion.
    candidates["score"] = (
        0.35 * candidates["node"].map(lambda x: float(threat[int(x)]))
        + 0.35 * candidates["node"].map(lambda x: float(revelation[int(x)]))
        + 0.30 * candidates["node"].map(lambda x: float(delay[int(x)]))
    )
    scoring_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    target_abs, decoy_abs, steps = _amc_solve(strategic_nodes, adj, threat, revelation, rng)
    amc_seconds = time.perf_counter() - t2
    total_seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return BenchResult(
        node_count=n,
        edge_count=edge_count,
        strategic_nodes=int(len(strategic_nodes)),
        strategic_edges=int(strategic_edges),
        candidate_actions=int(len(candidates)),
        repeat=repeat,
        extraction_seconds=extraction_seconds,
        scoring_seconds=scoring_seconds,
        amc_seconds=amc_seconds,
        total_seconds=total_seconds,
        peak_memory_mb=float(peak / (1024 * 1024)),
        target_absorption_mean=target_abs,
        decoy_absorption_mean=decoy_abs,
        expected_steps_mean=steps,
    )


def run(config_path: str, node_counts: list[int] | None = None, repeats: int | None = None) -> None:
    cfg = load_yaml(config_path)
    bench_cfg = cfg.get("paper_experiments", {}).get("realistic_ad_graph", {}).get("large_benchmark", {})
    counts = node_counts or [int(x) for x in bench_cfg.get("node_counts", [100, 500, 1000, 5000, 10000])]
    repeats = int(repeats or bench_cfg.get("repeats", 3))
    avg_degree = int(bench_cfg.get("avg_degree", 6))
    strategic_cap = int(bench_cfg.get("strategic_cap", 320))
    candidate_cap = int(bench_cfg.get("candidate_cap", 80))
    seed = int(cfg.get("project", {}).get("seed", 42))

    rows = []
    for n in counts:
        for r in range(repeats):
            print(f"[large-scale] nodes={n:,} repeat={r+1}/{repeats}")
            rows.append(_one_run(int(n), avg_degree, strategic_cap, candidate_cap, r, seed).__dict__)
    df = pd.DataFrame(rows)
    out_csv = resolve_path(cfg.get("paths", {}).get("large_scale_benchmark_csv", "outputs/phase6_large_scale_benchmark.csv"))
    ensure_parent(out_csv)
    df.to_csv(out_csv, index=False)
    summary = df.groupby("node_count").agg(
        repeats=("repeat", "count"),
        mean_edges=("edge_count", "mean"),
        mean_strategic_nodes=("strategic_nodes", "mean"),
        mean_strategic_edges=("strategic_edges", "mean"),
        mean_candidate_actions=("candidate_actions", "mean"),
        mean_extraction_seconds=("extraction_seconds", "mean"),
        mean_scoring_seconds=("scoring_seconds", "mean"),
        mean_amc_seconds=("amc_seconds", "mean"),
        mean_total_seconds=("total_seconds", "mean"),
        std_total_seconds=("total_seconds", "std"),
        mean_peak_memory_mb=("peak_memory_mb", "mean"),
        max_peak_memory_mb=("peak_memory_mb", "max"),
    ).reset_index()
    out_sum = resolve_path(cfg.get("paths", {}).get("large_scale_summary_csv", "outputs/phase6_large_scale_summary.csv"))
    ensure_parent(out_sum)
    summary.to_csv(out_sum, index=False)
    write_json({
        "benchmark_csv": str(out_csv),
        "summary_csv": str(out_sum),
        "node_counts": counts,
        "repeats": repeats,
        "avg_degree": avg_degree,
        "strategic_cap": strategic_cap,
        "candidate_cap": candidate_cap,
        "notes": "Measured synthetic AD-style sparse graphs; no proxy runtime extrapolation.",
    }, resolve_path("outputs/phase6_large_scale_benchmark_manifest.json"))
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/closed_loop_config.yaml")
    parser.add_argument("--node-counts", nargs="*", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.node_counts, args.repeats)


if __name__ == "__main__":
    main()
