from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def expected_node_rows(dataset_pt: str | Path) -> dict[str, Any]:
    bundle = torch.load(dataset_pt, map_location="cpu", weights_only=False)
    samples = bundle["samples"] if isinstance(bundle, dict) and "samples" in bundle else bundle
    keys: list[tuple[int, int, str]] = []
    split_counts: dict[str, int] = {}
    for s in samples:
        split = str(s.get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1
        for node_id in s["node_ids"]:
            keys.append((int(s["episode_id"]), int(s["t"]), str(node_id)))
    return {"keys": keys, "snapshots_by_split": split_counts}
