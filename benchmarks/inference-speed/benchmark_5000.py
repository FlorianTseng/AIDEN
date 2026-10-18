"""Numerically validate and benchmark cached inference on one 5,000-probe MP record."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch


FAST_INFER_DIR = Path(__file__).resolve().parent
REPO_ROOT = FAST_INFER_DIR.parent
if str(FAST_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_INFER_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiden.model import Aiden
from aiden.data import CHGCARRhoDataset
from aiden.fast_infer import FastAidenInference, slice_prebuilt_sample
from train import move_batch_to_device, structure_nmae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/aiden-pbe-l3.pt")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/aiden-mp")
    parser.add_argument("--elem-csv", type=Path, default=REPO_ROOT / "data/elem-des.csv")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--query-block-size", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--same-state-atol",
        type=float,
        default=2.0e-6,
        help="Strict tolerance for fast and original decoding from identical cached coefficients.",
    )
    parser.add_argument(
        "--same-state-rtol",
        type=float,
        default=2.0e-5,
        help="Strict relative tolerance for fast and original decoding from identical cached coefficients.",
    )
    parser.add_argument(
        "--chunk-atol",
        type=float,
        default=2.0e-4,
        help="Absolute tolerance for chunked CUDA runs with different reduction order.",
    )
    parser.add_argument(
        "--chunk-rtol",
        type=float,
        default=5.0e-4,
        help="Relative tolerance for chunked CUDA runs with different reduction order.",
    )
    parser.add_argument(
        "--nmae-tolerance",
        type=float,
        default=1.0e-6,
        help="Maximum allowed absolute NMAE difference for chunked inference.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=FAST_INFER_DIR / "benchmark_5000.json")
    return parser.parse_args()


def load_model_and_sample(args: argparse.Namespace) -> Tuple[Aiden, Dict[str, torch.Tensor], str]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["model_args"]
    model = Aiden(**model_args)
    state_key = "ema_model" if "ema_model" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    saved_args = checkpoint["args"]
    dataset = CHGCARRhoDataset(
        data_root=args.data_root,
        elem_csv=args.elem_csv,
        split=args.split,
        split_ratio=(
            float(saved_args.get("train_ratio", 0.8)),
            float(saved_args.get("val_ratio", 0.1)),
            float(saved_args.get("test_ratio", 0.1)),
        ),
        seed=int(saved_args.get("seed", 42)),
        num_queries=5000,
        max_files=saved_args.get("max_files"),
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
    sample = move_batch_to_device([dataset[args.index]], device)[0]
    if int(sample["query_frac"].shape[0]) != 5000:
        raise RuntimeError(
            f"benchmark requires 5,000 stored probes, got {sample['query_frac'].shape[0]}"
        )
    return model, sample, state_key


def timed(
    function: Callable[[], torch.Tensor],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed: List[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(time.perf_counter() - start)
    assert output is not None
    return output, sum(elapsed) / len(elapsed)


def main() -> None:
    args = parse_args()
    if args.query_block_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("query block size and repeats must be positive; warmup must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model, sample, state_key = load_model_and_sample(args)
    fast = FastAidenInference(model)
    chunks = [
        slice_prebuilt_sample(sample, start, min(start + args.query_block_size, 5000))
        for start in range(0, 5000, args.query_block_size)
    ]

    def original_one_shot() -> torch.Tensor:
        return model([sample])[0].pred

    def original_chunked() -> torch.Tensor:
        return torch.cat([model([chunk])[0].pred for chunk in chunks], dim=0)

    def fast_one_shot() -> torch.Tensor:
        prepared = fast.prepare(sample)
        return fast.predict_prebuilt(prepared, sample)

    def fast_chunked() -> torch.Tensor:
        prepared = fast.prepare(sample)
        return fast.predict_prebuilt_chunked(prepared, sample, args.query_block_size)

    with torch.inference_mode():
        reference = original_one_shot()
        original_chunk_prediction = original_chunked()
        fast_one_prediction = fast_one_shot()
        fast_chunk_prediction = fast_chunked()
        prepared_reference = fast.prepare(sample)
        cached_decoder_reference = fast.predict_prebuilt_reference(prepared_reference, sample)
        cached_fast_prediction = fast.predict_prebuilt(prepared_reference, sample)
    target = sample["target"]
    nmae_reference = float(structure_nmae(reference, target).cpu())
    nmae_original_chunked = float(structure_nmae(original_chunk_prediction, target).cpu())
    nmae_fast_one_shot = float(structure_nmae(fast_one_prediction, target).cpu())
    nmae_fast_chunked = float(structure_nmae(fast_chunk_prediction, target).cpu())

    # This controls the exact tensors consumed by the decoder, so it validates
    # the optimization itself rather than comparing separate non-deterministic
    # CUDA sparse-encoder launches.
    torch.testing.assert_close(
        cached_fast_prediction,
        cached_decoder_reference,
        atol=args.same_state_atol,
        rtol=args.same_state_rtol,
    )

    # The original chunked baseline runs the sparse CUDA atom encoder once per
    # probe block while the fast path runs it once. CUDA index_add/scatter
    # reductions are not bitwise deterministic across those separate launches,
    # although both execute the same model. The tolerance is only for that
    # observed accumulation-order noise; NMAE remains an independent accuracy
    # gate at a much tighter application-level tolerance.
    torch.testing.assert_close(
        fast_chunk_prediction,
        original_chunk_prediction,
        atol=args.chunk_atol,
        rtol=args.chunk_rtol,
    )
    nmae_delta_chunked = nmae_fast_chunked - nmae_original_chunked
    if abs(nmae_delta_chunked) > args.nmae_tolerance:
        raise AssertionError(
            "fast chunked NMAE differs from original chunked NMAE by "
            f"{nmae_delta_chunked:.8g}, exceeding {args.nmae_tolerance:.8g}"
        )

    _, original_one_seconds = timed(original_one_shot, args.warmup, args.repeats, device)
    _, original_chunk_seconds = timed(original_chunked, args.warmup, args.repeats, device)
    _, fast_one_seconds = timed(fast_one_shot, args.warmup, args.repeats, device)
    _, fast_chunk_seconds = timed(fast_chunked, args.warmup, args.repeats, device)

    num_points = int(reference.numel())
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_state": state_key,
        "sample": str(sample.get("name", "<unknown>")),
        "split": args.split,
        "split_index": args.index,
        "num_points": num_points,
        "query_block_size": args.query_block_size,
        "same_state_atol": args.same_state_atol,
        "same_state_rtol": args.same_state_rtol,
        "chunk_atol": args.chunk_atol,
        "chunk_rtol": args.chunk_rtol,
        "nmae_tolerance": args.nmae_tolerance,
        "nmae_original_one_shot": nmae_reference,
        "nmae_original_chunked": nmae_original_chunked,
        "nmae_fast_one_shot": nmae_fast_one_shot,
        "nmae_fast_chunked": nmae_fast_chunked,
        "nmae_delta_fast_chunked_vs_original_chunked": nmae_delta_chunked,
        "nmae_delta_fast_chunked_vs_original_one_shot": nmae_fast_chunked - nmae_reference,
        "max_abs_diff_fast_vs_cached_original_decoder": float(
            (cached_fast_prediction - cached_decoder_reference).abs().max().cpu()
        ),
        "max_abs_diff_fast_one_shot_vs_original_one_shot": float(
            (fast_one_prediction - reference).abs().max().cpu()
        ),
        "max_abs_diff_fast_chunked": float((fast_chunk_prediction - original_chunk_prediction).abs().max().cpu()),
        "original_one_shot_seconds": original_one_seconds,
        "original_chunked_seconds": original_chunk_seconds,
        "fast_one_shot_seconds": fast_one_seconds,
        "fast_chunked_seconds": fast_chunk_seconds,
        "original_one_shot_us_per_point": original_one_seconds * 1.0e6 / num_points,
        "original_chunked_us_per_point": original_chunk_seconds * 1.0e6 / num_points,
        "fast_one_shot_us_per_point": fast_one_seconds * 1.0e6 / num_points,
        "fast_chunked_us_per_point": fast_chunk_seconds * 1.0e6 / num_points,
        "speedup_fast_chunked_vs_original_chunked": original_chunk_seconds / fast_chunk_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
