from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .train_belief import FACTOR_KEYS, build_model
from .utils import load_yaml


def make_window(df: pd.DataFrame, features: List[str], window_size: int) -> torch.Tensor:
    values = df.sort_values("t")[features].tail(window_size).to_numpy(dtype=np.float32)
    if len(values) < window_size:
        pad = np.zeros((window_size - len(values), len(features)), dtype=np.float32)
        values = np.vstack([pad, values])
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/belief_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--snapshot_csv", required=True, help="CSV containing recent rows for one episode/window.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    checkpoint_path = args.checkpoint or cfg["paths"]["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv(args.snapshot_csv)
    x = make_window(df, cfg["features"], int(cfg["data"]["window_size"])).to(device)
    logits = model(x)

    result: Dict[str, Dict[str, float]] = {}
    for key in FACTOR_KEYS:
        p = torch.softmax(logits[key], dim=-1).squeeze(0).cpu().numpy()
        result[key] = {name: float(p[i]) for i, name in enumerate(cfg["labels"][key])}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
