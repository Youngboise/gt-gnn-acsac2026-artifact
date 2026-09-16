from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    """Convert NumPy/PyTorch scalar containers into JSON-serializable Python types."""
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
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def write_json(obj: Any, path: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str, base_dir: str | None = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if base_dir is None:
        base_dir = os.getcwd()
    return str(Path(base_dir) / p)


def entropy_np(probs: np.ndarray, eps: float = 1e-9) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / max(float(probs.sum()), eps)
    return float(-(probs * np.log(probs + eps)).sum())


def normalize01(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def safe_auc_ap(y_true, y_score):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
