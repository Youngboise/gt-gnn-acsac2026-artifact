from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import read_jsonl


def load_phase3_strategic_graphs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 3 strategic graph file not found: {path}\n"
            "Run Phase 3 first, or update paths.phase3_strategic_graphs_jsonl in config/amc_config.yaml."
        )
    graphs = read_jsonl(path)
    if not graphs:
        raise ValueError(f"No strategic graphs found in {path}")
    required = {"graph_id", "nodes", "edges", "candidate_actions", "belief_vector"}
    missing_examples = []
    for g in graphs[:10]:
        missing = required.difference(g.keys())
        if missing:
            missing_examples.append((g.get("graph_id", "unknown"), sorted(missing)))
    if missing_examples:
        raise ValueError(
            "Phase 4 needs full Phase 3 strategic graphs, not only the compact action input file. "
            f"Missing fields examples: {missing_examples}"
        )
    return graphs


def filter_graphs(graphs: List[Dict[str, Any]], split: str = "all", max_graphs: Optional[int] = None) -> List[Dict[str, Any]]:
    if split and split != "all":
        out = [g for g in graphs if str(g.get("split", "unknown")) == split]
    else:
        out = list(graphs)
    if max_graphs is not None:
        out = out[: int(max_graphs)]
    return out
