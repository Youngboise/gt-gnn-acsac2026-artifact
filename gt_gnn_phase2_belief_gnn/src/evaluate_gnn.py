from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .generate_gnn_dataset import package_root_from_config
from .gnn_dataset import GraphSnapshotDataset, collate_graph_snapshots
from .train_gnn import build_model, evaluate
from .utils import load_yaml


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gnn_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = package_root_from_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / cfg["paths"]["checkpoint"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]

    ds = GraphSnapshotDataset(str(root / cfg["paths"]["gnn_dataset_pt"]), split=args.split)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_graph_snapshots,
    )
    model = build_model(cfg, metadata).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, loader, device, int(cfg["ranking"]["top_k"]))

    print(f"Metrics for split={args.split}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
