from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import yaml


def package_root_from_config(config_path: str) -> Path:
    p = Path(config_path).resolve()
    if p.parent.name == "config":
        return p.parent.parent
    return Path.cwd()


def ensure_parent(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    return obj


def write_json(obj: Any, path: Union[str, Path]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def read_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Union[str, Path]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def clamp01(x: float) -> float:
    return clamp(float(x), 0.0, 1.0)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and math.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def sigmoid(x: float) -> float:
    x = clamp(x, -50.0, 50.0)
    return float(1.0 / (1.0 + math.exp(-x)))


def softmax(scores: List[float], temperature: float = 1.0) -> np.ndarray:
    if not scores:
        return np.asarray([], dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    s = np.asarray(scores, dtype=np.float64) / temp
    s = s - np.max(s)
    e = np.exp(s)
    denom = float(e.sum())
    if denom <= 0:
        return np.ones_like(e) / max(1, len(e))
    return e / denom


def normalized_entropy(probs: List[float]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0.0
    p = p / max(float(p.sum()), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def pearsonr(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
