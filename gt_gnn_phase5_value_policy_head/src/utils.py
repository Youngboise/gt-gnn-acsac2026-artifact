from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def write_json(obj: Any, path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True, name: str = "column") -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find {name}. Tried: {list(candidates)}. Existing columns: {list(df.columns)}")
    return None


def pick_group_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def groupwise_split(
    df: pd.DataFrame,
    group_col: Optional[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        total = train_ratio + val_ratio + test_ratio
        train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype="object")

    if group_col is None:
        idx = df.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train:n_train + n_val]] = "val"
        split.loc[idx[n_train + n_val:]] = "test"
        return split.fillna("test")

    groups = df[group_col].dropna().astype(str).unique()
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])

    group_values = df[group_col].astype(str)
    split.loc[group_values.isin(train_groups)] = "train"
    split.loc[group_values.isin(val_groups)] = "val"
    split = split.fillna("test")

    # Guarantee all splits exist for small datasets.
    counts = split.value_counts().to_dict()
    if len(counts) < 3 and len(df) >= 3:
        return groupwise_split(df, None, train_ratio, val_ratio, test_ratio, seed)
    return split


def infer_numeric_feature_columns(
    df: pd.DataFrame,
    target_cols: Iterable[str],
    exclude_cols: Iterable[str],
    id_like_substrings: Sequence[str] = ("id", "name", "node", "action_key"),
) -> List[str]:
    target_set = {c for c in target_cols if c is not None}
    exclude_set = set(exclude_cols) | target_set
    numeric_cols = list(df.select_dtypes(include=[np.number, "bool"]).columns)

    features: List[str] = []
    for c in numeric_cols:
        if c in exclude_set:
            continue
        lower = c.lower()
        # Keep meaningful numeric action/node features, but avoid obvious row identifiers.
        if lower in {"row_id", "index", "unnamed: 0"}:
            continue
        if lower.endswith("_id") or lower == "id":
            continue
        features.append(c)
    if not features:
        raise ValueError("No numeric feature columns were inferred. Set data.feature_columns explicitly in config/value_config.yaml.")
    return features


def fit_standardizer(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return {"mean": mean.astype(float), "std": std.astype(float), "median": median.astype(float)}


def apply_standardizer(values: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    median = np.asarray(scaler["median"], dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    x = np.where(np.isfinite(x), x, median)
    return ((x - mean) / std).astype(np.float32)


def inverse_standardizer(values: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    y = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    return (y * std + mean).astype(np.float32)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None
