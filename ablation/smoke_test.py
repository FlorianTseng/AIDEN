"""Fail-fast real-PBE forward/backward test for one ablation variant."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation.variants import ABLATION_VARIANTS, AIDENAblation  # noqa: E402
from aiden.data import CHGCARRhoDataset, collate_rho  # noqa: E402
from train import mae_objective, move_batch_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test an AIDEN ablation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--elem-csv", required=True)
    parser.add_argument("--num-queries", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def gradient_sum(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError("non-finite gradient")
            total += float(parameter.grad.detach().abs().sum().cpu())
    return total


def main() -> None:
    args = parse_args()
    variant = os.environ.get("AIDEN_ABLATION_VARIANT", "").strip()
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            "AIDEN_ABLATION_VARIANT must be one of: "
            + ", ".join(ABLATION_VARIANTS)
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(42)
    np.random.seed(42)
    dataset = CHGCARRhoDataset(
        data_root=args.data_root,
        elem_csv=args.elem_csv,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
        num_queries=args.num_queries,
        max_files=64,
        fixed_queries=True,
        atom_cutoff=4.0,
        max_atom_neighbors=100,
        query_cutoff=3.0,
        max_probe_neighbors=0,
        in_memory_cache_size=0,
        use_sample_probs=False,
    )
    batch = collate_rho([dataset[0]])
    batch = move_batch_to_device(batch, device)

    model_args = {
        "ablation_variant": variant,
        "atom_feat_dim": dataset.feature_dim,
        "num_channels": 48,
        "num_encoder_layers": 2,
        "lmax": 4,
        "mmax": 4,
        "correlation": 2,
        "edge_correlation": 2,
        "num_heads": 4,
        "edge_hidden": 48,
        "atom_cutoff": 4.0,
        "max_atom_neighbors": 100,
        "num_atom_rbf": 8,
        "radial_hidden": 128,
        "atom_radial_type": "bessel",
        "encoder_interaction_mode": "alternating",
        "num_orbitals": 8,
        "num_gto_radial": 8,
        "initial_guess_rank": 1,
        "orbital_cutoff": 3.0,
        "gto_min_exponent": 0.15,
        "gto_max_exponent": 256.0,
        "max_atomic_number": 118,
        "target_std": 1.258352004595664,
        "scale_factor": 0.1,
        "model_name": f"AIDEN-L4-PBE2k-ablation-{variant}",
    }
    model = AIDENAblation(**model_args).to(device).train()
    outputs = model(batch)
    loss, metrics = mae_objective(outputs, batch, model.density_scale)
    if not torch.isfinite(loss):
        raise AssertionError("non-finite smoke-test loss")
    loss.backward()
    encoder_grad = gradient_sum(model.encoder)
    decoder_grad = gradient_sum(model.decoder)
    if encoder_grad <= 0.0 or decoder_grad <= 0.0:
        raise AssertionError(
            f"missing gradients: encoder={encoder_grad}, decoder={decoder_grad}"
        )
    if variant == "no_rho_init":
        for output in outputs:
            if torch.count_nonzero(output.pred_initial).item() != 0:
                raise AssertionError("no_rho_init produced a nonzero initial density")

    state = model.state_dict()
    restored = AIDENAblation(**model_args)
    restored.load_state_dict(state, strict=True)
    print(
        f"smoke_test=pass variant={variant} "
        f"loss={float(loss.detach().cpu()):.8f} "
        f"nmae={metrics['nmae_sum'] / metrics['num_structures']:.8f} "
        f"encoder_grad={encoder_grad:.6e} decoder_grad={decoder_grad:.6e} "
        f"params={sum(parameter.numel() for parameter in model.parameters())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
