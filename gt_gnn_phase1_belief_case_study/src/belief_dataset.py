from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabelMaps:
    goal: Dict[str, int]
    skill: Dict[str, int]
    stealth: Dict[str, int]
    decoy_awareness: Dict[str, int]

    @classmethod
    def from_config(cls, labels: Dict[str, List[str]]) -> "LabelMaps":
        return cls(
            goal={name: i for i, name in enumerate(labels["goal"])},
            skill={name: i for i, name in enumerate(labels["skill"])},
            stealth={name: i for i, name in enumerate(labels["stealth"])},
            decoy_awareness={name: i for i, name in enumerate(labels["decoy_awareness"])},
        )


class BeliefSequenceDataset(Dataset):
    """Builds rolling windows from action-conditioned episode snapshots.

    Each item returns:
      x: [window_size, num_features]
      y: dict of factor labels
      meta: episode_id and t for traceability
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        features: List[str],
        label_maps: LabelMaps,
        window_size: int,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].copy()
        self.df.sort_values(["episode_id", "t"], inplace=True)
        self.features = features
        self.label_maps = label_maps
        self.window_size = window_size
        self.items: List[Tuple[np.ndarray, Dict[str, int], Dict[str, int]]] = []
        self._build_items()

    def _build_items(self) -> None:
        for episode_id, group in self.df.groupby("episode_id", sort=True):
            group = group.sort_values("t")
            values = group[self.features].to_numpy(dtype=np.float32)
            rows = group.to_dict("records")
            for idx, row in enumerate(rows):
                start = max(0, idx - self.window_size + 1)
                window = values[start : idx + 1]
                if len(window) < self.window_size:
                    pad = np.zeros((self.window_size - len(window), len(self.features)), dtype=np.float32)
                    window = np.vstack([pad, window])
                labels = {
                    "goal": self.label_maps.goal[row["goal_label"]],
                    "skill": self.label_maps.skill[row["skill_label"]],
                    "stealth": self.label_maps.stealth[row["stealth_label"]],
                    "decoy_awareness": self.label_maps.decoy_awareness[row["decoy_awareness_label"]],
                }
                meta = {"episode_id": int(episode_id), "t": int(row["t"])}
                self.items.append((window, labels, meta))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        window, labels, meta = self.items[idx]
        x = torch.tensor(window, dtype=torch.float32)
        y = {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()}
        return x, y, meta


def collate_belief_batch(batch):
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = {k: torch.stack([item[k] for item in ys], dim=0) for k in ys[0].keys()}
    meta = {
        "episode_id": torch.tensor([m["episode_id"] for m in metas], dtype=torch.long),
        "t": torch.tensor([m["t"] for m in metas], dtype=torch.long),
    }
    return x, y, meta
