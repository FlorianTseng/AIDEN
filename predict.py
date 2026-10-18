"""Predict charge density for one dataset record with Aiden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aiden.model import Aiden
from aiden.data import CHGCARRhoDataset
from train import move_batch_to_device, structure_nmae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiden single-structure prediction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--elem-csv", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="aiden_prediction.npz")
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
    max_files = args.max_files
    if max_files is None:
        max_files = saved_args.get("max_files", None)
    dataset = CHGCARRhoDataset(
        data_root=args.data_root,
        elem_csv=args.elem_csv,
        split="all",
        split_ratio=(0.8, 0.1, 0.1),
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
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} is outside [0, {len(dataset)})")
    cpu_sample = dataset[args.index]
    sample = move_batch_to_device([cpu_sample], device)[0]
    with torch.inference_mode():
        output = model([sample])[0]
    pred = output.pred.detach().cpu().numpy().astype(np.float32)
    initial = output.pred_initial.detach().cpu().numpy().astype(np.float32)
    learned = output.pred_learned.detach().cpu().numpy().astype(np.float32)
    target_tensor = sample["target"].to(output.pred)
    target = target_tensor.detach().cpu().numpy().astype(np.float32)
    mae = float(np.abs(pred - target).mean())
    nmae = float(structure_nmae(output.pred, target_tensor).cpu())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        name=np.asarray(str(sample.get("name", "<unknown>"))),
        query_frac=cpu_sample["query_frac"].numpy(),
        query_cart=cpu_sample["query_cart"].numpy(),
        density_pred=pred,
        density_initial=initial,
        density_learned=learned,
        density_target=target,
        mae=np.asarray(mae, dtype=np.float64),
        nmae=np.asarray(nmae, dtype=np.float64),
    )
    print(
        json.dumps(
            {
                "name": str(sample.get("name", "<unknown>")),
                "points": int(pred.size),
                "mae": mae,
                "nmae": nmae,
                "weights": state_key,
                "output": str(output_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
