"""Count ChargE3Net parameters at L (lmax) = 0..4.

Every hyperparameter other than lmax keeps the repository's current defaults
from ``charge3net/configs/charge3net/model/e3_density.yaml`` — the config the
published mp/qm9/nmc ``*_e3_final`` runs resolve to: num_interactions=3,
num_neighbors=20, mul=500, cutoff=4.0, basis="gaussian", num_basis=20.

The source repository is only read, never modified.  e3nn is supplied from the
local ``parameter-count/libs`` install when present (otherwise the ambient
e3nn is used).

Validation built in: the L = 4 model must strict-load the official released
weights ``charge3net/models/charge3net_mp.pt``, proving the default
configuration matches the published model.

Usage (from parameter-count/):  python3 count_charge3net.py > charge3net_params.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PARAM_QUANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARAM_QUANT_DIR.parents[1]
CHARGE3NET_REPO = REPO_ROOT / "third_party" / "charge3net"   # read-only
CHECKPOINT = CHARGE3NET_REPO / "models" / "charge3net_mp.pt"  # read-only

# Optional isolated e3nn install (parameter-count/libs) first, then the
# read-only charge3net checkout.
_libs = PARAM_QUANT_DIR / "libs"
if _libs.is_dir():
    sys.path.insert(0, str(_libs))
sys.path.insert(0, str(CHARGE3NET_REPO))

import torch  # noqa: E402

from src.charge3net.models.e3 import E3DensityModel  # noqa: E402


# Hyperparameters from configs/charge3net/model/e3_density.yaml (lmax excluded).
MODEL_KWARGS = {
    "num_interactions": 3,
    "num_neighbors": 20,
    "mul": 500,
    "cutoff": 4.0,
    "basis": "gaussian",
    "num_basis": 20,
}

FINE_PREFIXES = {
    "c3n_atom_convs_params": "atom_model.convolutions.",
    "c3n_atom_gates_params": "atom_model.gates.",
    "c3n_probe_convs_params": "probe_model.convolutions.",
    "c3n_probe_gates_params": "probe_model.gates.",
    "c3n_probe_readout_params": "probe_model.readout.",
}
COARSE_PREFIXES = {
    "atom_model_params": "atom_model.",
    "probe_model_params": "probe_model.",
}


def breakdown(model: torch.nn.Module) -> dict:
    named = dict(model.named_parameters())
    total = sum(t.numel() for t in named.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    components = {
        label: sum(t.numel() for k, t in named.items() if k.startswith(prefix))
        for label, prefix in {**FINE_PREFIXES, **COARSE_PREFIXES}.items()
    }
    assert components["atom_model_params"] + components["probe_model_params"] == total, (
        "atom_model + probe_model do not cover all parameters"
    )
    fine_atom = components["c3n_atom_convs_params"] + components["c3n_atom_gates_params"]
    fine_probe = (
        components["c3n_probe_convs_params"]
        + components["c3n_probe_gates_params"]
        + components["c3n_probe_readout_params"]
    )
    assert fine_atom == components["atom_model_params"], "fine atom_model breakdown incomplete"
    assert fine_probe == components["probe_model_params"], "fine probe_model breakdown incomplete"
    return {
        "total_params": total,
        "trainable_params": trainable,
        **components,
    }


def validate_l4_against_official_checkpoint() -> str:
    """Strict-load the official released MP weights into the L=4 model."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = E3DensityModel(lmax=4, **MODEL_KWARGS)
    model.load_state_dict(checkpoint["model"], strict=True)
    checkpoint_params = sum(
        v.numel()
        for k, v in checkpoint["model"].items()
        if k in dict(model.named_parameters())
    )
    return (
        f"charge3net/models/charge3net_mp.pt strict-loaded into "
        f"E3DensityModel(lmax=4, **e3_density.yaml defaults); checkpoint holds "
        f"{checkpoint_params} learnable parameters, matching the constructed model"
    )


def main() -> None:
    rows = []
    for lmax in range(5):
        model = E3DensityModel(lmax=lmax, **MODEL_KWARGS)
        rows.append(
            {
                "model": "ChargE3Net",
                "L": lmax,
                "lmax": lmax,
                **breakdown(model),
            }
        )
    payload = {
        "rows": rows,
        "defaults": (
            "charge3net/configs/charge3net/model/e3_density.yaml: "
            "num_interactions=3, num_neighbors=20, mul=500, cutoff=4.0, "
            'basis="gaussian", num_basis=20 (the code constructor default '
            "num_basis=10 is overridden by every published config)"
        ),
        "validation": validate_l4_against_official_checkpoint(),
    }
    json.dump(payload, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
