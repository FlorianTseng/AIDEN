"""Fail-fast Aiden GPU/data/gradient/equivariance smoke test."""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from aiden.model import Aiden
from aiden.data import CHGCARRhoDataset, collate_rho
from train import mae_objective, move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aiden packed-data smoke test")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--elem-csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-queries", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-std", type=float, default=1.258352004595664)
    parser.add_argument("--num-channels", type=int, default=48)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--mmax", type=int, default=3)
    parser.add_argument("--correlation", type=int, default=2)
    parser.add_argument("--edge-correlation", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--edge-hidden", type=int, default=48)
    parser.add_argument("--num-orbitals", type=int, default=8)
    parser.add_argument("--num-gto-radial", type=int, default=8)
    parser.add_argument("--atom-cutoff", type=float, default=4.0)
    parser.add_argument("--orbital-cutoff", type=float, default=3.0)
    parser.add_argument("--max-atom-neighbors", type=int, default=100)
    parser.add_argument("--num-atom-rbf", type=int, default=8)
    parser.add_argument("--radial-hidden", type=int, default=128)
    parser.add_argument("--encoder-interaction-mode", default="alternating")
    return parser.parse_args()


def rotation_matrix(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    axis = torch.tensor([0.37, -0.51, 0.78], dtype=dtype, device=device)
    axis = axis / axis.norm()
    angle = torch.tensor(0.713, dtype=dtype, device=device)
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=dtype, device=device)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=dtype, device=device)
    return eye + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def rotate_batch(
    batch_list: List[Dict[str, object]],
    rotation: torch.Tensor,
) -> List[Dict[str, object]]:
    rotated_batches: List[Dict[str, object]] = []
    for batch in batch_list:
        rotated: Dict[str, object] = {}
        for key, value in batch.items():
            rotated[key] = value.clone() if isinstance(value, torch.Tensor) else value
        for key in ("atom_cart", "query_cart", "atom_edge_vec", "probe_edge_vec"):
            value = rotated.get(key)
            if isinstance(value, torch.Tensor):
                rotated[key] = value @ rotation.transpose(0, 1)
        lattice = rotated.get("lattice")
        if isinstance(lattice, torch.Tensor):
            rotated["lattice"] = lattice @ rotation.transpose(0, 1)
        rotated_batches.append(rotated)
    return rotated_batches


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise AssertionError(f"{name} contains NaN or Inf")


def gradient_summary(module: torch.nn.Module) -> tuple[float, int]:
    total = 0.0
    count = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        assert_finite_tensor("gradient", parameter.grad)
        total += float(parameter.grad.detach().abs().sum().cpu())
        count += parameter.numel()
    return total, count


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_queries <= 0:
        raise ValueError("batch size and query count must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested but CUDA is unavailable")

    dataset = CHGCARRhoDataset(
        data_root=args.data_root,
        elem_csv=args.elem_csv,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        seed=args.seed,
        num_queries=args.num_queries,
        max_files=max(64, args.batch_size * 4),
        fixed_queries=True,
        atom_cutoff=args.atom_cutoff,
        max_atom_neighbors=args.max_atom_neighbors,
        query_cutoff=args.orbital_cutoff,
        max_probe_neighbors=0,
        in_memory_cache_size=0,
        use_sample_probs=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_rho,
    )
    cpu_batch = next(iter(loader))
    observed_queries = [int(batch["target"].numel()) for batch in cpu_batch]
    if any(count != args.num_queries for count in observed_queries):
        raise AssertionError(
            f"packed smoke expected {args.num_queries} queries, got {observed_queries}"
        )
    batch = move_batch_to_device(cpu_batch, device)

    model = Aiden(
        atom_feat_dim=dataset.feature_dim,
        num_channels=args.num_channels,
        num_encoder_layers=args.num_encoder_layers,
        lmax=args.lmax,
        correlation=args.correlation,
        atom_cutoff=args.atom_cutoff,
        max_atom_neighbors=args.max_atom_neighbors,
        num_atom_rbf=args.num_atom_rbf,
        radial_hidden=args.radial_hidden,
        atom_radial_type="bessel",
        mmax=args.mmax,
        edge_correlation=args.edge_correlation,
        num_heads=args.num_heads,
        edge_hidden=args.edge_hidden,
        encoder_interaction_mode=args.encoder_interaction_mode,
        num_orbitals=args.num_orbitals,
        num_gto_radial=args.num_gto_radial,
        initial_guess_rank=1,
        orbital_cutoff=args.orbital_cutoff,
        target_std=args.target_std,
        scale_factor=0.1,
    ).to(device)

    # The randomly initialized model must start on a sane scale.  This gate
    # would have rejected the previous contribution head's ~60x initial NMAE.
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        initial_outputs = model.forward_initial_guess(batch)
        _, metrics = mae_objective(outputs, batch, model.density_scale)
    initial_nmae = metrics["nmae_sum"] / metrics["num_structures"]
    if not math.isfinite(initial_nmae) or initial_nmae > 5.0:
        raise AssertionError(
            f"untrained Aiden has pathological NMAE={initial_nmae:.6f}"
        )
    for index, (output, initial_output) in enumerate(zip(outputs, initial_outputs)):
        assert_finite_tensor(f"prediction[{index}]", output.pred)
        assert_finite_tensor(f"initial_prediction[{index}]", initial_output.pred)

    # Rotation invariance covers both the Cartesian encoder and GTO decoder.
    rotation = rotation_matrix(batch[0]["lattice"].dtype, device)
    rotated_batch = rotate_batch(batch, rotation)
    with torch.no_grad():
        rotated_outputs = model(rotated_batch)
    max_rotation_error = 0.0
    for output, rotated_output in zip(outputs, rotated_outputs):
        denominator = output.pred.abs().max().clamp_min(1.0)
        relative = float(
            ((output.pred - rotated_output.pred).abs().max() / denominator).cpu()
        )
        max_rotation_error = max(max_rotation_error, relative)
    if max_rotation_error > 2.0e-4:
        raise AssertionError(
            f"Aiden rotation invariance error is {max_rotation_error:.3e}"
        )

    # One initial-guess update must produce finite, non-zero gradients.
    model.train()
    for parameter in model.parameters():
        parameter.grad = None
    initial_outputs = model.forward_initial_guess(batch)
    initial_loss, _ = mae_objective(initial_outputs, batch, model.density_scale)
    initial_loss.backward()
    initial_grad, initial_grad_count = gradient_summary(model.decoder.initial_guess)
    if initial_grad_count == 0 or initial_grad <= 0.0:
        raise AssertionError("initial-guess module received no gradient")

    # One complete Adam update must reach both encoder and decoder.
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": 2.0e-4},
            {"params": model.decoder.parameters(), "lr": 1.0e-3},
        ]
    )
    outputs = model(batch)
    loss, _ = mae_objective(outputs, batch, model.density_scale)
    loss.backward()
    encoder_grad, encoder_grad_count = gradient_summary(model.encoder)
    decoder_grad, decoder_grad_count = gradient_summary(model.decoder)
    if encoder_grad_count == 0 or encoder_grad <= 0.0:
        raise AssertionError("Cartesian encoder received no gradient")
    if decoder_grad_count == 0 or decoder_grad <= 0.0:
        raise AssertionError("decoder received no gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    # Checkpoint round-trip catches missing buffers and stale constructor args.
    model.eval()
    with torch.no_grad():
        reference = [output.pred.clone() for output in model(batch)]
    with tempfile.TemporaryDirectory(prefix="aiden-smoke-") as temporary_dir:
        path = Path(temporary_dir) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = Aiden(
            atom_feat_dim=dataset.feature_dim,
            num_channels=args.num_channels,
            num_encoder_layers=args.num_encoder_layers,
            lmax=args.lmax,
            correlation=args.correlation,
            atom_cutoff=args.atom_cutoff,
            max_atom_neighbors=args.max_atom_neighbors,
            num_atom_rbf=args.num_atom_rbf,
            radial_hidden=args.radial_hidden,
            mmax=args.mmax,
            edge_correlation=args.edge_correlation,
            num_heads=args.num_heads,
            edge_hidden=args.edge_hidden,
            encoder_interaction_mode=args.encoder_interaction_mode,
            num_orbitals=args.num_orbitals,
            num_gto_radial=args.num_gto_radial,
            orbital_cutoff=args.orbital_cutoff,
            target_std=args.target_std,
        ).to(device)
        restored.load_state_dict(torch.load(path, map_location=device), strict=True)
        restored.eval()
        with torch.no_grad():
            restored_values = [output.pred for output in restored(batch)]
    max_restore_error = max(
        float((expected - actual).abs().max().cpu())
        for expected, actual in zip(reference, restored_values)
    )
    if max_restore_error != 0.0:
        raise AssertionError(f"checkpoint round-trip error={max_restore_error:.3e}")

    points = sum(observed_queries)
    print(
        "Aiden smoke PASS | "
        f"device={device} structures={len(batch)} points={points} "
        f"initial_nmae={initial_nmae:.6f} rotation_error={max_rotation_error:.3e} "
        f"loss={float(loss.detach().cpu()):.6f} "
        f"encoder_grad_l1={encoder_grad:.3e} decoder_grad_l1={decoder_grad:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
