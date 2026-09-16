from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class GraphSnapshotDataset(Dataset):
    def __init__(self, pt_path: str, split: str):
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.metadata = obj["metadata"]
        self.samples = [s for s in obj["samples"] if s["split"] == split]
        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {pt_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_graph_snapshots(batch: List[Dict]) -> Dict:
    # We keep each graph separate because the package intentionally avoids PyTorch Geometric.
    return {"graphs": batch}


def split_batches(graphs: List[Dict], batch_size: int):
    for i in range(0, len(graphs), batch_size):
        yield graphs[i : i + batch_size]
