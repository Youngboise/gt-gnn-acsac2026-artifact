from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import (
    apply_standardizer,
    first_existing_column,
    fit_standardizer,
    groupwise_split,
    infer_numeric_feature_columns,
    pick_group_column,
)


REGRESSION_TARGETS = ["risk", "info", "delay", "utility"]


@dataclass
class PreparedValueData:
    x: torch.Tensor
    y_reg_scaled: torch.Tensor
    y_reg_raw: torch.Tensor
    y_best: torch.Tensor
    y_rank_utility: torch.Tensor
    group_id: torch.Tensor
    split: np.ndarray
    row_info: pd.DataFrame
    feature_columns: List[str]
    target_columns: Dict[str, str]
    feature_scaler: Dict[str, object]
    target_scaler: Dict[str, object]
    group_column: Optional[str]


class ValuePolicyDataset(Dataset):
    def __init__(self, prepared: Dict[str, object], split_name: str):
        split = np.asarray(prepared["split"])
        self.indices = np.where(split == split_name)[0]
        if len(self.indices) == 0:
            raise ValueError(f"Split '{split_name}' has no rows.")
        self.x = prepared["x"]
        self.y_reg_scaled = prepared["y_reg_scaled"]
        self.y_reg_raw = prepared["y_reg_raw"]
        self.y_best = prepared["y_best"]
        self.y_rank_utility = prepared.get("y_rank_utility", prepared["y_reg_raw"][:, 3:4])
        self.group_id = prepared.get("group_id", torch.arange(len(prepared["split"]), dtype=torch.long))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = int(self.indices[idx])
        return {
            "x": self.x[row],
            "y_reg_scaled": self.y_reg_scaled[row],
            "y_reg_raw": self.y_reg_raw[row],
            "y_best": self.y_best[row],
            "y_rank_utility": self.y_rank_utility[row],
            "group_id": self.group_id[row],
            "row_index": torch.tensor(row, dtype=torch.long),
        }


def _resolve_target_columns(df: pd.DataFrame, target_config: Dict[str, List[str]]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for name in REGRESSION_TARGETS:
        resolved[name] = first_existing_column(df, target_config[name], required=True, name=f"target '{name}'")

    best_col = first_existing_column(df, target_config["best_action"], required=False, name="best action target")
    if best_col is not None:
        resolved["best_action"] = best_col
    return resolved


def _groupwise_rank_utility(df: pd.DataFrame, utility_col: str, group_col: Optional[str]) -> pd.Series:
    if group_col and group_col in df.columns:
        lo = df.groupby(group_col)[utility_col].transform("min")
        hi = df.groupby(group_col)[utility_col].transform("max")
        denom = (hi - lo).replace(0.0, np.nan)
        return ((df[utility_col] - lo) / denom).fillna(0.5).clip(0.0, 1.0)
    v = pd.to_numeric(df[utility_col], errors="coerce").fillna(0.0)
    lo, hi = float(v.min()), float(v.max())
    return ((v - lo) / max(hi - lo, 1e-12)).clip(0.0, 1.0)


def _make_margin_good_action(df: pd.DataFrame, utility_col: str, group_col: Optional[str], margin: float) -> pd.Series:
    if group_col and group_col in df.columns:
        max_utility = df.groupby(group_col)[utility_col].transform("max")
        return (df[utility_col] >= max_utility - float(margin)).astype(float)
    threshold = df[utility_col].quantile(0.90)
    return (df[utility_col] >= threshold).astype(float)


def _make_row_info(df: pd.DataFrame, split: pd.Series, group_col: Optional[str]) -> pd.DataFrame:
    candidate_cols = [
        "graph_id", "snapshot_id", "sample_id", "episode_id", "action_id", "action_key",
        "candidate_node", "candidate_node_id", "candidate", "node_id", "node_name", "decoy_node",
        "place_node", "place_nodes", "action_type", "decoy_type_hint", "exposure_level_hint", "split",
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    if group_col and group_col not in cols:
        cols.append(group_col)
    info = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    info["row_index"] = np.arange(len(df))
    info["split"] = split.to_numpy()
    return info


def prepare_from_dataframe(df: pd.DataFrame, cfg: Dict[str, object], seed: int) -> PreparedValueData:
    data_cfg = cfg["data"]
    target_cfg = data_cfg["target_columns"]
    target_cols = _resolve_target_columns(df, target_cfg)

    group_col = pick_group_column(df, data_cfg.get("group_columns", []))

    if "split" in df.columns and data_cfg.get("split_strategy") == "use_existing_or_group":
        split = df["split"].astype(str).str.lower().replace({"valid": "val", "validation": "val"})
        if not set(split.unique()).intersection({"train", "val", "test"}):
            split = groupwise_split(df, group_col, float(data_cfg["train_ratio"]), float(data_cfg["val_ratio"]), float(data_cfg["test_ratio"]), seed)
    else:
        split = groupwise_split(df, group_col, float(data_cfg["train_ratio"]), float(data_cfg["val_ratio"]), float(data_cfg["test_ratio"]), seed)

    # Ranking target is always graph-wise normalized raw utility, even when regression uses raw utility.
    df = df.copy()
    margin = float(cfg.get("selection", {}).get("margin_best_action", 0.02))
    df["utility_rank_target"] = _groupwise_rank_utility(df, target_cols["utility"], group_col)
    df["is_good_action_margin"] = _make_margin_good_action(df, target_cols["utility"], group_col, margin)
    target_cols["best_action"] = "is_good_action_margin"

    explicit_features = data_cfg.get("feature_columns")
    if explicit_features:
        missing = [c for c in explicit_features if c not in df.columns]
        if missing:
            raise ValueError(f"Configured feature columns are missing: {missing}")
        feature_cols = list(explicit_features)
    else:
        feature_cols = infer_numeric_feature_columns(df, target_cols=list(target_cols.values()) + ["utility_rank_target"], exclude_cols=data_cfg.get("exclude_feature_columns", []))

    train_mask = split.to_numpy() == "train"
    if train_mask.sum() == 0:
        raise ValueError("No training rows were found after splitting.")

    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_reg_raw_np = df[[target_cols[t] for t in REGRESSION_TARGETS]].to_numpy(dtype=np.float32)
    y_best_np = df[target_cols["best_action"]].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_rank_np = df["utility_rank_target"].to_numpy(dtype=np.float32).reshape(-1, 1)

    feature_scaler = fit_standardizer(x_raw[train_mask])
    x_scaled = apply_standardizer(x_raw, feature_scaler)

    target_scaler = fit_standardizer(y_reg_raw_np[train_mask])
    y_reg_scaled_np = apply_standardizer(y_reg_raw_np, target_scaler)

    if group_col and group_col in df.columns:
        codes, _ = pd.factorize(df[group_col].astype(str), sort=False)
    else:
        codes = np.arange(len(df))
    row_info = _make_row_info(df, split, group_col)

    return PreparedValueData(
        x=torch.tensor(x_scaled, dtype=torch.float32),
        y_reg_scaled=torch.tensor(y_reg_scaled_np, dtype=torch.float32),
        y_reg_raw=torch.tensor(y_reg_raw_np, dtype=torch.float32),
        y_best=torch.tensor(y_best_np, dtype=torch.float32),
        y_rank_utility=torch.tensor(y_rank_np, dtype=torch.float32),
        group_id=torch.tensor(codes, dtype=torch.long),
        split=split.to_numpy(),
        row_info=row_info,
        feature_columns=feature_cols,
        target_columns=target_cols,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        group_column=group_col,
    )


def prepared_to_dict(prepared: PreparedValueData) -> Dict[str, object]:
    return {
        "x": prepared.x,
        "y_reg_scaled": prepared.y_reg_scaled,
        "y_reg_raw": prepared.y_reg_raw,
        "y_best": prepared.y_best,
        "y_rank_utility": prepared.y_rank_utility,
        "group_id": prepared.group_id,
        "split": prepared.split,
        "row_info": prepared.row_info,
        "feature_columns": prepared.feature_columns,
        "target_columns": prepared.target_columns,
        "feature_scaler": prepared.feature_scaler,
        "target_scaler": prepared.target_scaler,
        "group_column": prepared.group_column,
        "regression_targets": REGRESSION_TARGETS,
    }
