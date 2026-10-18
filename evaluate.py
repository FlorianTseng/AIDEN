"""Evaluate an Aiden checkpoint with MAE and per-structure NMAE."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from aiden.model import Aiden
from aiden.data import CHGCARRhoDataset, collate_rho
from train import move_batch_to_device, structure_nmae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Aiden")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--elem-csv", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-queries", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-structures", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="./eval-aiden")
    parser.add_argument("--raw-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if not isinstance(model_args, dict):
        raise RuntimeError("checkpoint does not contain Aiden model_args")
    model = Aiden(**model_args)
    state_key = "model" if args.raw_weights or "ema_model" not in checkpoint else "ema_model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    saved_args = checkpoint.get("args", {})
    split_ratio = (
        float(saved_args.get("train_ratio", 0.8)),
        float(saved_args.get("val_ratio", 0.1)),
        float(saved_args.get("test_ratio", 0.1)),
    )
    max_files = args.max_files
    if max_files is None:
        max_files = saved_args.get("max_files", None)
    dataset = CHGCARRhoDataset(
        data_root=args.data_root,
        elem_csv=args.elem_csv,
        split=args.split,
        split_ratio=split_ratio,
        seed=int(saved_args.get("seed", 42)),
        num_queries=args.num_queries,
        max_files=max_files,
        fixed_queries=True,
        atom_cutoff=float(model_args["atom_cutoff"]),
        max_atom_neighbors=int(model_args["max_atom_neighbors"]),
        query_cutoff=float(model_args["orbital_cutoff"]),
        max_probe_neighbors=0,
        in_memory_cache_size=0,
        use_sample_probs=False,
    )
    if 0 < args.max_structures < len(dataset):
        indices = np.linspace(
            0,
            len(dataset) - 1,
            args.max_structures,
            dtype=np.int64,
        ).tolist()
        evaluation_set = Subset(dataset, indices)
    else:
        evaluation_set = dataset
    loader = DataLoader(
        evaluation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_rho,
    )

    rows = []
    absolute_error_sum = 0.0
    target_abs_sum = 0.0
    point_count = 0
    nmae_sum = 0.0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = move_batch_to_device(cpu_batch, device)
            outputs = model(batch)
            for output, sample in zip(outputs, batch):
                target = sample["target"].to(output.pred)
                error_sum = float((output.pred - target).abs().sum().cpu())
                sample_points = int(target.numel())
                sample_target_abs = float(target.abs().sum().cpu())
                sample_mae = error_sum / max(sample_points, 1)
                sample_nmae = float(structure_nmae(output.pred, target).cpu())
                rows.append(
                    {
                        "name": sample.get("name", "<unknown>"),
                        "points": sample_points,
                        "mae": sample_mae,
                        "nmae": sample_nmae,
                    }
                )
                absolute_error_sum += error_sum
                target_abs_sum += sample_target_abs
                point_count += sample_points
                nmae_sum += sample_nmae

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": state_key,
        "split": args.split,
        "num_structures": len(rows),
        "num_points": point_count,
        "mae": absolute_error_sum / max(point_count, 1),
        "nmae": nmae_sum / max(len(rows), 1),
        "global_nmae": absolute_error_sum / max(target_abs_sum, 1.0e-8),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "per_structure.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "points", "mae", "nmae"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
