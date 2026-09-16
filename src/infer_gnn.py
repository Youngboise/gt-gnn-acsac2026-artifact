from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch

from .generate_gnn_dataset import package_root_from_config
from .gnn_dataset import GraphSnapshotDataset
from .train_gnn import build_model, move_graph_to_device
from .utils import ensure_parent, load_yaml, write_json


def _split_list(split: str) -> List[str]:
    if split == "all":
        return ["train", "val", "test"]
    return [split]


def _limit(n: int, max_snapshots: int | None) -> int:
    if max_snapshots is None or int(max_snapshots) <= 0:
        return n
    return min(int(max_snapshots), n)


@torch.no_grad()
def _infer_split(model, cfg: dict, root: Path, split: str, device: torch.device, max_snapshots: int | None) -> tuple[list[dict], dict]:
    ds = GraphSnapshotDataset(str(root / cfg["paths"]["gnn_dataset_pt"]), split=split)
    total = len(ds)
    n_take = _limit(total, max_snapshots)
    rows: list[dict] = []
    for sample_idx in range(n_take):
        raw = ds[sample_idx]
        g = move_graph_to_device(raw, device)
        out = model.forward_graph(g)
        threat = torch.sigmoid(out["threat_logits"]).cpu().numpy()
        revelation = torch.sigmoid(out["revelation_logits"]).cpu().numpy()
        candidate = torch.sigmoid(out["candidate_logits"]).cpu().numpy()
        gate = torch.sigmoid(out["gate_logits"]).cpu().numpy()
        fused = torch.sigmoid(out["fused_logits"]).cpu().numpy()
        mask = g["candidate_mask"].cpu().numpy()
        for i, node_id in enumerate(raw["node_ids"]):
            rows.append(
                {
                    "episode_id": int(raw["episode_id"]),
                    "t": int(raw["t"]),
                    "split": split,
                    "node_id": node_id,
                    "candidate_mask": float(mask[i]),
                    "threat_score": float(threat[i]),
                    "revelation_score": float(revelation[i]),
                    "candidate_score": float(candidate[i]),
                    "threat_gate": float(gate[i]),
                    "fused_score": float(fused[i]),
                    "true_threat": float(raw["threat_y"][i]),
                    "true_revelation": float(raw["revelation_y"][i]),
                    "true_candidate": float(raw["candidate_y"][i]),
                    "inference_source": "learned_gnn",
                }
            )
    return rows, {"split": split, "snapshots_available": int(total), "snapshots_inferred": int(n_take), "node_rows": int(len(rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 2 node predictions for all requested snapshots. Use --split all and --max_snapshots 0 for paper results.")
    parser.add_argument("--config", default="config/gnn_config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--max_snapshots", type=int, default=0, help="0 or negative means infer every snapshot in each selected split.")
    parser.add_argument("--output", default=None, help="Optional output CSV path. Defaults to config paths.predictions_csv.")
    parser.add_argument("--coverage_json", default=None, help="Optional coverage JSON path.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = package_root_from_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else root / cfg["paths"]["checkpoint"]
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    model = build_model(cfg, metadata).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    max_snapshots = None if int(args.max_snapshots) <= 0 else int(args.max_snapshots)
    all_rows: list[dict] = []
    split_reports: list[dict] = []
    for split in _split_list(args.split):
        rows, report = _infer_split(model, cfg, root, split, device, max_snapshots)
        all_rows.extend(rows)
        split_reports.append(report)
        print(f"split={split}: inferred {report['snapshots_inferred']:,}/{report['snapshots_available']:,} snapshots; rows={report['node_rows']:,}")

    out_csv = Path(args.output) if args.output else root / cfg["paths"]["predictions_csv"]
    ensure_parent(str(out_csv))
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    coverage = {
        "prediction_csv": str(out_csv),
        "split_requested": args.split,
        "max_snapshots": "all" if max_snapshots is None else max_snapshots,
        "total_node_rows": int(len(df)),
        "splits": split_reports,
        "required_for_acsac": "Use split=all and max_snapshots=0, then disable Phase 3 proxy fallback.",
    }
    cov_path = Path(args.coverage_json) if args.coverage_json else out_csv.with_suffix(".coverage.json")
    write_json(coverage, str(cov_path))

    print(f"Wrote node-level predictions to {out_csv}")
    print(f"Wrote coverage report to {cov_path}")
    if len(df):
        ranked = df[df["candidate_mask"] > 0.5].sort_values(["episode_id", "t", "fused_score"], ascending=[True, True, False])
        print("Top candidate examples:")
        print(ranked.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
