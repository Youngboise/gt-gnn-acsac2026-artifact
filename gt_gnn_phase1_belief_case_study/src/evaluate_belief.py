from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .belief_dataset import BeliefSequenceDataset, LabelMaps, collate_belief_batch
from .belief_model import FactorizedBeliefGRU
from .train_belief import FACTOR_KEYS, build_model, evaluate
from .utils import ensure_parent, load_yaml


@torch.no_grad()
def export_predictions(model, loader, cfg: Dict, device, out_csv: str) -> None:
    model.eval()
    label_names = cfg["labels"]
    rows = []
    for x, y, meta in loader:
        x = x.to(device)
        logits = model(x)
        probs = {k: torch.softmax(v, dim=-1).cpu().numpy() for k, v in logits.items()}
        batch_size = x.size(0)
        for i in range(batch_size):
            row = {
                "episode_id": int(meta["episode_id"][i]),
                "t": int(meta["t"][i]),
            }
            for key in FACTOR_KEYS:
                p = probs[key][i]
                pred_idx = int(np.argmax(p))
                row[f"true_{key}"] = label_names[key][int(y[key][i])]
                row[f"pred_{key}"] = label_names[key][pred_idx]
                row[f"conf_{key}"] = float(p[pred_idx])
                for j, name in enumerate(label_names[key]):
                    row[f"p_{key}_{name}"] = float(p[j])
            rows.append(row)
    ensure_parent(out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/belief_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    checkpoint_path = args.checkpoint or cfg["paths"]["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_maps = LabelMaps.from_config(cfg["labels"])
    ds = BeliefSequenceDataset(
        csv_path=cfg["paths"]["dataset_csv"],
        split=args.split,
        features=cfg["features"],
        label_maps=label_maps,
        window_size=int(cfg["data"]["window_size"]),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_belief_batch,
    )

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(model, loader, device, float(cfg["training"]["label_smoothing"]))
    print(f"Metrics for split={args.split}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    export_predictions(model, loader, cfg, device, cfg["paths"]["predictions_csv"])
    print(f"Wrote predictions to {cfg['paths']['predictions_csv']}")


if __name__ == "__main__":
    main()
