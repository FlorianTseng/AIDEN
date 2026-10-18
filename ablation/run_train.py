"""Use the production trainer with an ablation model selected by environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train as production_train  # noqa: E402
from ablation.data import (  # noqa: E402
    AIDENRapidAblationDataset,
    SPLIT_SEED,
    SUBSET_SEED,
    TRAIN_STRUCTURES,
    VALIDATION_STRUCTURES,
)
from ablation.variants import ABLATION_VARIANTS, AIDENAblation  # noqa: E402


VARIANT = os.environ.get("AIDEN_ABLATION_VARIANT", "").strip()
if VARIANT not in ABLATION_VARIANTS:
    raise ValueError(
        "AIDEN_ABLATION_VARIANT must be one of: " + ", ".join(ABLATION_VARIANTS)
    )


_production_parse_args = production_train.parse_args
_production_model_arguments = production_train.model_arguments


def parse_args():
    args = _production_parse_args()
    preparing_initial_guess = (
        os.environ.get("AIDEN_PREPARE_SHARED_INITIAL_GUESS", "0") == "1"
    )
    if args.lmax != 4 or args.mmax != 4:
        raise ValueError("this rapid ablation study is fixed at lmax=mmax=4")
    if args.num_encoder_layers != 2:
        raise ValueError("this ablation study requires two interaction layers")
    if args.seed != SPLIT_SEED:
        raise ValueError(f"this ablation study requires seed={SPLIT_SEED}")
    if args.val_max_structures != VALIDATION_STRUCTURES:
        raise ValueError(
            "this ablation study requires "
            f"val_max_structures={VALIDATION_STRUCTURES}"
        )
    if preparing_initial_guess:
        if args.max_steps != 1 or args.initial_guess_steps != 1000:
            raise ValueError(
                "shared initialization requires max_steps=1 and "
                "initial_guess_steps=1000"
            )
    elif args.max_steps != 100000:
        raise ValueError("formal rapid ablations require max_steps=100000")
    args.model_name = f"AIDEN-L4-PBE2k-ablation-{VARIANT}"
    # These extra fields are persisted in config.json for an auditable record
    # of the deterministic data view used by all variants.
    args.ablation_train_structures = TRAIN_STRUCTURES
    args.ablation_subset_seed = SUBSET_SEED
    args.ablation_preparing_shared_initial_guess = preparing_initial_guess
    args.encoder_interaction_mode = {
        "ace_ace": "ace_only",
        "tece_tece": "tece_only",
        "tece_ace": "stacked",
    }.get(VARIANT, "alternating")
    args.disable_gie = VARIANT == "no_gie"
    args.disable_rra = VARIANT == "no_rra"
    args.disable_radial_correction = VARIANT == "fixed_gto"
    if VARIANT == "no_rho_init":
        # There is no one-center module to pretrain in this variant.
        args.initial_guess_steps = 0
        args.initial_guess_checkpoint = None
    return args


def model_arguments(args, atom_feature_dim: int, target_std: float):
    values = _production_model_arguments(args, atom_feature_dim, target_std)
    values["ablation_variant"] = VARIANT
    return values


production_train.parse_args = parse_args
production_train.model_arguments = model_arguments
production_train.Aiden = AIDENAblation
production_train.CHGCARRhoDataset = AIDENRapidAblationDataset


if __name__ == "__main__":
    production_train.main()
