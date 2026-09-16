from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx

from .utils import load_yaml, package_root_from_config, read_jsonl


def visualize(config_path: str, graph_id: Optional[str], limit: int) -> None:
    cfg = load_yaml(config_path)
    root = package_root_from_config(config_path)
    graphs = read_jsonl(root / cfg["paths"]["strategic_graphs_jsonl"])
    out_dir = root / cfg["paths"]["visualization_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if graph_id:
        graphs = [g for g in graphs if g["graph_id"] == graph_id]
    graphs = graphs[:limit]
    if not graphs:
        print("No graphs selected for visualization.")
        return

    for graph in graphs:
        G = nx.DiGraph()
        labels: Dict[int, str] = {}
        for n in graph["nodes"]:
            idx = int(n["strategic_index"])
            G.add_node(idx, tags=n.get("tags", []), role=n.get("role", "unknown"))
            labels[idx] = str(n["node_id"]).replace("host:", "")
        for e in graph["edges"]:
            G.add_edge(int(e["source"]), int(e["target"]))
        pos = nx.spring_layout(G, seed=7)
        node_sizes: List[float] = []
        for idx in G.nodes:
            tags = set(G.nodes[idx].get("tags", []))
            if "target" in tags:
                node_sizes.append(900)
            elif "selected_candidates" in tags or "candidate" in tags:
                node_sizes.append(650)
            else:
                node_sizes.append(450)
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=10, alpha=0.45)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        plt.title(f"Strategic Graph: {graph['graph_id']}")
        plt.axis("off")
        out_path = out_dir / f"{graph['graph_id']}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/strategic_config.yaml")
    parser.add_argument("--graph_id", default=None)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    visualize(args.config, args.graph_id, args.limit)


if __name__ == "__main__":
    main()
