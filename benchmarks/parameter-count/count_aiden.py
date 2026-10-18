"""Count AIDEN parameters at L = 0..4.

L = M (the paper's convention); every other hyperparameter keeps the released
defaults of the ``aiden`` package (constructor defaults, which match the
recorded paper configs ``configs/pbe-l{0..4}.json`` exactly:
48 channels, 2 encoder layers, alternating ACE/TECE, correlations (2, 2),
4 heads, 8 Bessel functions, r_at = 4.0 A, K = 8, K0 = 1, N_G = 8,
r_orb = 3.0 A, 133-dim element input).

The source repositories are only read, never modified.

Validation built in: the L = 4 model must strict-load the EMA weights of the
released checkpoint ``checkpoints/aiden-pbe.pt``.

Usage (from parameter-count/):  python3 count_aiden.py > aiden_params.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PARAM_QUANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARAM_QUANT_DIR.parents[1]
AIDEN_REPO = REPO_ROOT                            # read-only
CHECKPOINT = REPO_ROOT / "checkpoints" / "aiden-pbe.pt"          # read-only

sys.path.insert(0, str(AIDEN_REPO))

import torch  # noqa: E402

from aiden import Aiden  # noqa: E402


# Parameter-name prefixes used for the component breakdown.
FINE_PREFIXES = {
    "gie_params": "encoder.gie.",
    "ace_params": "encoder.blocks.",
    "tece_params": "encoder.tece_blocks.",
    "encoder_norms_params": "encoder.norms.",
    "encoder_final_norm_params": "encoder.final_norm.",
    "coeff_head_params": "decoder.coefficient_head.",
    "radial_correction_params": "decoder.gto.radial_correction.",
    "initial_guess_params": "decoder.initial_guess.",
}
COARSE_PREFIXES = {
    "encoder_params": "encoder.",
    "decoder_params": "decoder.",
}


def breakdown(model: torch.nn.Module) -> dict:
    named = dict(model.named_parameters())
    total = sum(t.numel() for t in named.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    components = {
        label: sum(t.numel() for k, t in named.items() if k.startswith(prefix))
        for label, prefix in {**FINE_PREFIXES, **COARSE_PREFIXES}.items()
    }
    fine_encoder = (
        components["gie_params"]
        + components["ace_params"]
        + components["tece_params"]
        + components["encoder_norms_params"]
        + components["encoder_final_norm_params"]
    )
    fine_decoder = (
        components["coeff_head_params"]
        + components["radial_correction_params"]
        + components["initial_guess_params"]
    )
    assert components["encoder_params"] + components["decoder_params"] == total, (
        "encoder + decoder do not cover all parameters"
    )
    assert fine_encoder == components["encoder_params"], "fine encoder breakdown incomplete"
    assert fine_decoder == components["decoder_params"], "fine decoder breakdown incomplete"
    return {
        "total_params": total,
        "trainable_params": trainable,
        **components,
    }


def validate_l3_against_checkpoint() -> str:
    """Strict-load the trained L=3 EMA weights into a freshly built model."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = Aiden(atom_feat_dim=133, lmax=4, mmax=4)
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    checkpoint_params = sum(
        v.numel()
        for k, v in checkpoint["ema_model"].items()
        if k in dict(model.named_parameters())
    )
    return (
        f"checkpoints/aiden-pbe.pt EMA weights strict-loaded into "
        f"Aiden(lmax=4, mmax=4); checkpoint holds {checkpoint_params} learnable "
        f"parameters, matching the constructed model"
    )


def main() -> None:
    rows = []
    for l_value in range(5):
        model = Aiden(atom_feat_dim=133, lmax=l_value, mmax=l_value)
        rows.append(
            {
                "model": "AIDEN",
                "L": l_value,
                "mmax": l_value,
                **breakdown(model),
            }
        )
    payload = {
        "rows": rows,
        "defaults": (
            "aiden constructor defaults == configs/pbe-l{L}.json: "
            "atom_feat_dim=133, num_channels=48, num_encoder_layers=2, "
            "correlation=2, edge_correlation=2, edge_hidden=48, num_heads=4, "
            "num_atom_rbf=8 (bessel), radial_hidden=128, num_orbitals=8, "
            "num_gto_radial=8, initial_guess_rank=1, atom_cutoff=4.0, "
            "orbital_cutoff=3.0, encoder_interaction_mode=alternating, all "
            "components (GIE/attention/RMSNorm/TECE/ECE/RRA/radial phase/"
            "temperature) enabled"
        ),
        "validation": validate_l3_against_checkpoint(),
    }
    json.dump(payload, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
