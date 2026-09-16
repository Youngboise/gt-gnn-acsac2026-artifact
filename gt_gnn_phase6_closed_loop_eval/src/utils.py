from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = resolve_path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(obj: Any, path: str | os.PathLike[str]) -> None:
    p = resolve_path(path)
    ensure_parent(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def read_csv_required(path: str | os.PathLike[str], label: str) -> pd.DataFrame:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find {label}: {p}")
    df = pd.read_csv(p)
    print(f"Loaded {label}: {p} ({len(df):,} rows, {len(df.columns)} columns)")
    return df


def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str] | str,
    required: bool = True,
    name: str = "column",
) -> Optional[str]:
    if isinstance(candidates, str):
        candidates = [candidates]
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            f"Could not find {name}. Tried: {list(candidates)}. Existing columns: {list(df.columns)}"
        )
    return None


def minmax_by_group(df: pd.DataFrame, value_col: str, group_col: str, out_col: str) -> pd.Series:
    grouped = df.groupby(group_col)[value_col]
    min_v = grouped.transform("min")
    max_v = grouped.transform("max")
    denom = (max_v - min_v).replace(0, np.nan)
    out = (df[value_col] - min_v) / denom
    df[out_col] = out.fillna(0.0)
    return df[out_col]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    lo = s.min()
    hi = s.max()
    if abs(float(hi - lo)) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def stable_hash_int(value: Any, modulo: int | None = None) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    out = int(digest[:16], 16)
    return out % int(modulo) if modulo else out
