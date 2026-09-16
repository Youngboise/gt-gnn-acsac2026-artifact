from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import safe_torch_load, tensor_to_numpy


PredictionLookup = Dict[Tuple[int, int, str], Dict[str, float]]


def load_phase2_bundle(dataset_path: Union[str, Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bundle = safe_torch_load(dataset_path)
    if not isinstance(bundle, dict) or "samples" not in bundle:
        raise ValueError(f"Expected Phase 2 .pt bundle with keys 'samples' and 'metadata': {dataset_path}")
    samples = list(bundle["samples"])
    metadata = dict(bundle.get("metadata", {}))
    return samples, metadata


def load_predictions(predictions_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    path = Path(predictions_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = {"episode_id", "t", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase 2 predictions file is missing columns: {sorted(missing)}")
    return df


def sample_key(sample: Dict[str, Any]) -> Tuple[int, int]:
    return int(sample["episode_id"]), int(sample["t"])


def filter_samples(samples: List[Dict[str, Any]], split: str, max_snapshots: Optional[int]) -> List[Dict[str, Any]]:
    if split != "all":
        samples = [s for s in samples if str(s.get("split", "")) == split]
    samples = sorted(samples, key=lambda s: (int(s["episode_id"]), int(s["t"])))
    if max_snapshots is not None:
        samples = samples[: int(max_snapshots)]
    return samples


def build_prediction_lookup(pred_df: Optional[pd.DataFrame]) -> PredictionLookup:
    lookup: PredictionLookup = {}
    if pred_df is None:
        return lookup
    score_cols = [
        "threat_score",
        "revelation_score",
        "candidate_score",
        "threat_gate",
        "fused_score",
        "candidate_mask",
        "true_threat",
        "true_revelation",
        "true_candidate",
    ]
    for _, row in pred_df.iterrows():
        key = (int(row["episode_id"]), int(row["t"]), str(row["node_id"]))
        lookup[key] = {c: float(row[c]) for c in score_cols if c in row and not pd.isna(row[c])}
    return lookup


def prediction_coverage(samples: List[Dict[str, Any]], pred_lookup: PredictionLookup) -> Dict[str, Any]:
    expected = 0
    found = 0
    missing_examples: List[Dict[str, Any]] = []
    by_split: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        episode_id, t = sample_key(sample)
        split = str(sample.get("split", "unknown"))
        by_split.setdefault(split, {"expected": 0, "found": 0})
        for node_id in [str(n) for n in sample["node_ids"]]:
            expected += 1
            by_split[split]["expected"] += 1
            if (episode_id, t, node_id) in pred_lookup:
                found += 1
                by_split[split]["found"] += 1
            elif len(missing_examples) < 25:
                missing_examples.append({"episode_id": episode_id, "t": t, "node_id": node_id, "split": split})
    out = {
        "expected_node_rows": int(expected),
        "found_node_rows": int(found),
        "missing_node_rows": int(expected - found),
        "coverage": float(found / max(expected, 1)),
        "by_split": {
            k: {**v, "coverage": float(v["found"] / max(v["expected"], 1))}
            for k, v in sorted(by_split.items())
        },
        "missing_examples": missing_examples,
    }
    return out


def get_feature_index(feature_names: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def _proxy_scores(sample: Dict[str, Any], n: int) -> Dict[str, np.ndarray]:
    threat_y = tensor_to_numpy(sample.get("threat_y", np.zeros(n)))
    revelation_y = tensor_to_numpy(sample.get("revelation_y", np.zeros(n)))
    candidate_y = tensor_to_numpy(sample.get("candidate_y", np.zeros(n)))
    gate_y = tensor_to_numpy(sample.get("gate_y", np.full(n, 0.5)))
    fused = np.clip(gate_y * threat_y + (1.0 - gate_y) * revelation_y + 0.20 * candidate_y, 0.0, 1.0)
    return {
        "threat_score": threat_y.astype(np.float32),
        "revelation_score": revelation_y.astype(np.float32),
        "candidate_score": candidate_y.astype(np.float32),
        "threat_gate": gate_y.astype(np.float32),
        "fused_score": fused.astype(np.float32),
    }


def get_scores_for_sample(
    sample: Dict[str, Any],
    pred_lookup: PredictionLookup,
    *,
    allow_proxy_fallback: bool = False,
) -> Dict[str, np.ndarray]:
    """Return node-level Phase 2 scores.

    ACSAC-strength results should use learned Phase 2 predictions for every
    sample/node. If `allow_proxy_fallback` is False, this function raises as soon
    as a sample is missing from the inference CSV instead of silently using proxy
    labels. Proxy fallback is kept only for debugging and upper-bound sanity runs.
    """
    episode_id, t = sample_key(sample)
    node_ids = [str(n) for n in sample["node_ids"]]
    n = len(node_ids)
    out = {
        "threat_score": np.zeros(n, dtype=np.float32),
        "revelation_score": np.zeros(n, dtype=np.float32),
        "candidate_score": np.zeros(n, dtype=np.float32),
        "threat_gate": np.zeros(n, dtype=np.float32),
        "fused_score": np.zeros(n, dtype=np.float32),
    }
    missing: list[str] = []
    for i, node_id in enumerate(node_ids):
        row = pred_lookup.get((episode_id, t, node_id))
        if row is None:
            missing.append(node_id)
            continue
        for k in out:
            if k in row:
                out[k][i] = float(row[k])
            else:
                missing.append(node_id)
                break

    if not missing:
        return out

    if allow_proxy_fallback:
        return _proxy_scores(sample, n)

    examples = ", ".join(missing[:5])
    raise KeyError(
        f"Missing Phase 2 learned predictions for episode_id={episode_id}, t={t}, "
        f"missing_nodes={len(missing)}/{n}; examples=[{examples}]. "
        "Run Phase 2 inference with --split all --max_snapshots 0, or set "
        "prediction_fallback.allow_proxy_fallback=true for debug-only runs."
    )
