from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader

from .belief_dataset import BeliefSequenceDataset, LabelMaps, collate_belief_batch
from .train_belief import FACTOR_KEYS, build_model
from .utils import expected_calibration_error, load_yaml, write_json


def _corrupt(x: torch.Tensor, missing_rate: float, noise_std: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    out = x.clone()
    if missing_rate > 0:
        mask = torch.rand(out.shape, generator=gen, device=out.device) < float(missing_rate)
        out = out.masked_fill(mask, 0.0)
    if noise_std > 0:
        out = out + torch.randn(out.shape, generator=gen, device=out.device) * float(noise_std)
    return out


@torch.no_grad()
def evaluate_scenario(model, loader, cfg: dict[str, Any], device, scenario: str, missing_rate: float, noise_std: float, seed: int) -> dict[str, Any]:
    model.eval()
    probs_by_factor = {k: [] for k in FACTOR_KEYS}
    labels_by_factor = {k: [] for k in FACTOR_KEYS}
    for batch_i, (x, y, _) in enumerate(loader):
        x = _corrupt(x.to(device), missing_rate, noise_std, seed + batch_i)
        logits = model(x)
        for key in FACTOR_KEYS:
            probs_by_factor[key].append(torch.softmax(logits[key], dim=-1).cpu().numpy())
            labels_by_factor[key].append(y[key].numpy())
    row: dict[str, Any] = {"scenario": scenario, "missing_rate": missing_rate, "noise_std": noise_std}
    accs = []
    eces = []
    f1s = []
    nlls = []
    for key in FACTOR_KEYS:
        p = np.concatenate(probs_by_factor[key], axis=0)
        y = np.concatenate(labels_by_factor[key], axis=0)
        pred = p.argmax(axis=1)
        acc = float((pred == y).mean())
        f1 = float(f1_score(y, pred, average="macro", zero_division=0))
        ece = expected_calibration_error(p, y)
        try:
            nll = float(log_loss(y, p, labels=list(range(p.shape[1]))))
        except Exception:
            nll = float("nan")
        row[f"{key}_accuracy"] = acc
        row[f"{key}_macro_f1"] = f1
        row[f"{key}_ece"] = ece
        row[f"{key}_nll"] = nll
        accs.append(acc); f1s.append(f1); eces.append(ece); nlls.append(nll)
    row["macro_accuracy"] = float(np.mean(accs))
    row["macro_f1"] = float(np.mean(f1s))
    row["macro_ece"] = float(np.mean(eces))
    row["macro_nll"] = float(np.nanmean(nlls))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Belief estimator stress tests for noisy/missing observations; outputs ACSAC-ready F1/NLL/ECE table.")
    parser.add_argument("--config", default="config/belief_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_csv", default="outputs/phase1_belief_stress_summary.csv")
    parser.add_argument("--output_json", default="outputs/phase1_belief_stress_summary.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_maps = LabelMaps.from_config(cfg["labels"])
    ds = BeliefSequenceDataset(
        csv_path=cfg["paths"]["dataset_csv"],
        split=args.split,
        features=cfg["features"],
        label_maps=label_maps,
        window_size=int(cfg["data"]["window_size"]),
    )
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, collate_fn=collate_belief_batch)
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint or cfg["paths"]["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    scenarios = [
        ("clean", 0.0, 0.0),
        ("missing_10pct", 0.10, 0.0),
        ("missing_30pct", 0.30, 0.0),
        ("noise_005", 0.0, 0.05),
        ("noise_010", 0.0, 0.10),
        ("missing_30pct_noise_010", 0.30, 0.10),
    ]
    rows = [evaluate_scenario(model, loader, cfg, device, name, miss, noise, int(cfg.get("seed", 7)) + i * 997) for i, (name, miss, noise) in enumerate(scenarios)]
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    write_json({"split": args.split, "num_windows": len(ds), "rows": rows}, args.output_json)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
