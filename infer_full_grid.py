"""Full-grid charge-density inference for one structure from a CHGCAR.

The structure encoder runs once; the complete FFT-grid density is then decoded
block by block from the cached atom representation.  Density arrays are
flattened in VASP/CHGCAR order (x is the fastest index), so a reshaped array
is indexed as ``rho.reshape(grid_shape, order="F")[ix, iy, iz]``.  The complete
CHGCAR target is retained and full-grid MAE/NMAE are reported for the
structure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from aiden.data import (
    build_atom_graph_np,
    load_element_descriptors,
    parse_chgcar_file,
)
from aiden.fast_infer import FastAidenInference
from aiden.model import Aiden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, help="Short system label used in metadata")
    parser.add_argument("--input", type=Path, required=True, help="Raw CHGCAR path")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--elem-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-block-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expected-grid-shape",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        help="Fail if the CHGCAR grid does not match these dimensions",
    )
    parser.add_argument(
        "--expected-atoms",
        type=int,
        help="Fail if the CHGCAR atom count does not match this value",
    )
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Use raw model weights instead of checkpoint EMA weights",
    )
    parser.add_argument(
        "--non-strict-graph-order",
        action="store_true",
        help="Skip probe-edge ordering; strict ordering is the reference default",
    )
    parser.add_argument(
        "--omit-target-from-npz",
        action="store_true",
        help="Do not duplicate the CHGCAR target array in the prediction archive",
    )
    return parser.parse_args()


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


def _atomic_save_npz(path: Path, payload: Dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def move_batch_to_device(
    batch_list: Sequence[Dict[str, object]],
    device: torch.device,
) -> list:
    moved = []
    for batch in batch_list:
        moved.append(
            {
                key: (
                    value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in batch.items()
            }
        )
    return moved


def structure_nmae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().sum() / target.abs().sum().clamp_min(1.0e-8)


@torch.inference_mode()
def density_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    absolute_error = (prediction - target).abs()
    return {
        "mae": float(absolute_error.mean().cpu()),
        "nmae": float(structure_nmae(prediction, target).cpu()),
        "max_abs_error": float(absolute_error.max().cpu()),
    }


def main() -> None:
    args = parse_args()
    if args.query_block_size <= 0:
        raise ValueError("--query-block-size must be positive")

    input_path = _require_file(args.input, "CHGCAR input")
    checkpoint_path = _require_file(args.checkpoint, "checkpoint")
    elem_csv_path = _require_file(args.elem_csv, "element descriptor CSV")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "full_grid_prediction.npz"
    summary_path = output_dir / "summary.json"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    total_start = time.perf_counter()
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint.get("model_args")
    if not isinstance(model_args, dict):
        raise RuntimeError("checkpoint does not contain Aiden model_args")
    state_key = "model" if args.raw_weights or "ema_model" not in checkpoint else "ema_model"
    model = Aiden(**model_args)
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval()

    parse_start = time.perf_counter()
    symbol_to_z, symbol_to_feat, _, feature_names, _ = load_element_descriptors(
        elem_csv_path
    )
    print(f"Parsing complete CHGCAR target: {input_path}", flush=True)
    parsed = parse_chgcar_file(input_path, symbol_to_z, symbol_to_feat)
    parse_seconds = time.perf_counter() - parse_start

    grid_shape = tuple(int(value) for value in parsed.grid_shape)
    num_atoms = int(parsed.atomic_numbers.size)
    num_points = int(parsed.density.size)
    if num_points != int(np.prod(grid_shape, dtype=np.int64)):
        raise RuntimeError(
            f"density length {num_points} does not match grid product {grid_shape}"
        )
    if args.expected_grid_shape is not None:
        expected_grid_shape = tuple(int(value) for value in args.expected_grid_shape)
        if grid_shape != expected_grid_shape:
            raise RuntimeError(
                f"CHGCAR grid {grid_shape} does not match expected {expected_grid_shape}"
            )
    if args.expected_atoms is not None and num_atoms != args.expected_atoms:
        raise RuntimeError(
            f"CHGCAR atom count {num_atoms} does not match expected {args.expected_atoms}"
        )
    expected_feature_dim = int(model_args["atom_feat_dim"])
    if len(feature_names) != expected_feature_dim:
        raise RuntimeError(
            f"descriptor width {len(feature_names)} does not match model width "
            f"{expected_feature_dim}"
        )

    print(
        f"Building atom graph: atoms={num_atoms}, species={parsed.species}, "
        f"counts={parsed.counts}, grid={grid_shape}, points={num_points}",
        flush=True,
    )
    atom_edge_src, atom_edge_dst, atom_edge_vec, atom_edge_dist = build_atom_graph_np(
        parsed.frac_coords.astype(np.float32, copy=False),
        parsed.lattice.astype(np.float32, copy=False),
        cutoff=float(model_args["atom_cutoff"]),
        exclude_self=True,
        max_neighbors=int(model_args["max_atom_neighbors"]),
    )
    cpu_sample: Dict[str, object] = {
        "name": parsed.name,
        "lattice": torch.from_numpy(parsed.lattice.astype(np.float32, copy=False)),
        "atom_frac": torch.from_numpy(parsed.frac_coords.astype(np.float32, copy=False)),
        "atom_z": torch.from_numpy(parsed.atomic_numbers.astype(np.int64, copy=False)),
        "atom_features": torch.from_numpy(parsed.atom_features.astype(np.float32, copy=False)),
        "atom_edge_src": torch.from_numpy(atom_edge_src),
        "atom_edge_dst": torch.from_numpy(atom_edge_dst),
        "atom_edge_vec": torch.from_numpy(atom_edge_vec),
        "atom_edge_dist": torch.from_numpy(atom_edge_dist),
        "grid_shape": torch.tensor(grid_shape, dtype=torch.long),
        "volume": torch.tensor(parsed.volume, dtype=torch.float32),
    }
    sample = move_batch_to_device([cpu_sample], device)[0]

    fast_model = FastAidenInference(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encoder_start = time.perf_counter()
    with torch.inference_mode():
        prepared = fast_model.prepare(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encoder_seconds = time.perf_counter() - encoder_start

    strict_graph_order = not args.non_strict_graph_order
    print(
        f"Running encoder-cached full-grid inference in blocks of "
        f"{args.query_block_size} points",
        flush=True,
    )
    fast_start = time.perf_counter()
    with torch.inference_mode():
        prediction = fast_model.predict_full_grid(
            prepared,
            grid_shape,
            args.query_block_size,
            strict_graph_order=strict_graph_order,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - fast_start
    if int(prediction.numel()) != num_points:
        raise RuntimeError(
            f"prediction length {prediction.numel()} does not match target "
            f"length {num_points}"
        )

    target = torch.from_numpy(parsed.density).to(device=device, dtype=prediction.dtype)
    metrics = density_metrics(prediction, target)

    prediction_cpu = (
        prediction.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    target_density = parsed.density.astype(np.float32, copy=False)
    target_integrated_electrons = float(
        np.mean(target_density, dtype=np.float64) * parsed.volume
    )
    predicted_integrated_electrons = float(
        np.mean(prediction_cpu, dtype=np.float64) * parsed.volume
    )

    archive_payload = {
        "name": np.asarray(args.system),
        "source_name": np.asarray(parsed.name),
        "density_pred": prediction_cpu,
        "grid_shape": np.asarray(grid_shape, dtype=np.int64),
        "lattice": parsed.lattice.astype(np.float32, copy=False),
        "species": np.asarray(parsed.species),
        "counts": np.asarray(parsed.counts, dtype=np.int64),
        "atom_frac": parsed.frac_coords.astype(np.float32, copy=False),
        "atomic_numbers": parsed.atomic_numbers.astype(np.int64, copy=False),
        "volume": np.asarray(parsed.volume, dtype=np.float64),
        "mae": np.asarray(metrics["mae"], dtype=np.float64),
        "nmae": np.asarray(metrics["nmae"], dtype=np.float64),
    }
    if not args.omit_target_from_npz:
        archive_payload["density_target"] = target_density

    save_start = time.perf_counter()
    print(f"Saving prediction archive: {prediction_path}", flush=True)
    _atomic_save_npz(prediction_path, archive_payload)
    save_seconds = time.perf_counter() - save_start

    summary: Dict[str, object] = {
        "system": args.system,
        "source_name": parsed.name,
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "weight_state": state_key,
        "device": str(device),
        "strict_graph_order": strict_graph_order,
        "query_block_size": args.query_block_size,
        "grid_shape": list(grid_shape),
        "num_points": num_points,
        "num_atoms": num_atoms,
        "species": parsed.species,
        "counts": parsed.counts,
        "volume_angstrom3": parsed.volume,
        "density_units": "electron/angstrom^3",
        "metric_reference": "supplied CHGCAR grid",
        **metrics,
        "nmae_percent": 100.0 * metrics["nmae"],
        "integrated_electrons": predicted_integrated_electrons,
        "integrated_electron_error": (
            predicted_integrated_electrons - target_integrated_electrons
        ),
        "target_integrated_electrons": target_integrated_electrons,
        "prediction_archive": str(prediction_path),
        "archive_contains_target": not args.omit_target_from_npz,
        "timing_seconds": {
            "parse_chgcar": parse_seconds,
            "encode_structure": encoder_seconds,
            "decode_full_grid": inference_seconds,
            "inference_total": encoder_seconds + inference_seconds,
            "save_archive": save_seconds,
            "total": time.perf_counter() - total_start,
        },
    }
    _atomic_save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
