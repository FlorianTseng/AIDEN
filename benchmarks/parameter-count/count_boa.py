"""Count BOA parameters at maximum angular order L = 0..4.

BOA's angular structure lives in its quantum-chemistry GTO basis, not in an
``lmax`` argument: every per-element equivariant linear operates on that
element's basis irreps.  The L sweep therefore filters BOA's default basis to
shells with l <= L:

    default (configs/data/qm9_vasp_small.yaml): def2-qzvppd, uncontracted,
    even_tempered=False, elements Z = [1, 6, 7, 8, 9] (QM9).

def2-qzvppd reaches l = 4 on C/N/O/F (g shells) and l = 3 on H, so the nested
filter family has L = 4 as its top row — which is *exactly* the default
configuration.  L = 0..4 are the same basis with p/d/f/g shells progressively
removed.  All other hyperparameters keep the defaults of
``configs/model/net/boa.yaml`` (4 BoaBlocks, hidden_channels=32,
num_orbitals=8) and ``configs/model/boa.yaml`` (orb_cutoff=3.0,
use_radial_correction=true), mirroring how ``boa/model/module.py`` assembles
the model: BOA net + per-element ``GTOs`` (whose radial-correction MLPs are
learnable and are included, matching AIDEN's counting convention).

Source repositories are only read, never modified.

Validation built in: the L = 4 basis must equal the unfiltered default
constructor path (``BasisInfo.from_atomic_numbers_with_even_tempered_basis``)
in irreps and in resulting parameter count.

Usage (from parameter-count/):  python3 count_boa.py > boa_params.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PARAM_QUANT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PARAM_QUANT_DIR.parents[1]
BOA_REPO = REPO_ROOT / "third_party" / "boa"              # read-only

# Local dependency install (e3nn, pyscf, torch_geometric, zarr, loguru,
# omegaconf) first, then the read-only BOA bundle.
for path in (
    # PARAM_QUANT_DIR / "libs",  # optional isolated deps
    BOA_REPO,
    BOA_REPO / "structures25",   # provides mldft
    BOA_REPO / "scdp",           # provides scdp
):
    sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import pyscf  # noqa: E402
import torch  # noqa: E402

from boa.data.basis_info import BasisInfo  # noqa: E402
from boa.model.gtos import GTOs  # noqa: E402
from boa.model.net.boa_block import BoaBlock  # noqa: E402
from boa.model.net.boa_block_stack import BoaBlockStack  # noqa: E402
from boa.model.net.boa_net import BOA, ReducedEdgeEmbedding  # noqa: E402
from pyscf.data.elements import ELEMENTS  # noqa: E402
from scdp.model.basis_set import basis_from_pyscf  # noqa: E402


# Defaults from configs/data/qm9_vasp_small.yaml and configs/model/net/boa.yaml.
DEFAULT_BASIS = "def2-qzvppd"
ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
N_BLOCKS = 4
HIDDEN_CHANNELS = 32
NUM_ORBITALS = 8
ORB_CUTOFF = 3.0
USE_RADIAL_CORRECTION = True


def build_basis_info(max_l: int) -> BasisInfo:
    """Default BasisInfo path with shells restricted to l <= max_l.

    Mirrors ``BasisInfo.from_atomic_numbers_with_even_tempered_basis`` with
    its default arguments (even_tempered=False, uncontracted=True); with
    max_l >= 4 it is bit-identical to that constructor.
    """
    one_atom_mols = []
    for z in ATOMIC_NUMBERS:
        symbol = ELEMENTS[z]
        mol = pyscf.gto.M(atom=f"{z} 0 0 0", spin=None, basis=DEFAULT_BASIS)
        uncontracted = {key: pyscf.gto.uncontract(shells) for key, shells in mol._basis.items()}
        filtered = [shell for shell in uncontracted[symbol] if shell[0] <= max_l]
        one_atom_mols.append(
            pyscf.gto.M(atom=f"{z} 0 0 0", spin=None, basis={symbol: filtered})
        )
    basis_dict, irreps_per_atom, integrals = BasisInfo.get_irreps_and_integrals(one_atom_mols)
    return BasisInfo(
        atomic_numbers=np.array(ATOMIC_NUMBERS, dtype=np.uint8),
        basis_dict=basis_dict,
        irreps_per_atom=irreps_per_atom,
        integrals=integrals,
    )


def build_model(basis_info: BasisInfo) -> torch.nn.Module:
    """Assemble BOA net + per-element GTOs like ChgLightningModule does."""

    class BoaWithGtos(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.boa_net = BOA(
                basis_info=basis_info,
                boa_stack=BoaBlockStack(
                    block=BoaBlock,
                    n_blocks=N_BLOCKS,
                    basis_info=basis_info,
                    hidden_channels=HIDDEN_CHANNELS,
                ),
                initial_guess_module=ReducedEdgeEmbedding(basis_info, channels=1),
                direct_gs_prediction=True,
                num_orbitals=NUM_ORBITALS,
            )
            self.gto_dict = torch.nn.ModuleDict(
                {
                    str(z): GTOs(
                        **basis_from_pyscf(basis_info.basis_dict[ELEMENTS[z]]),
                        cutoff=ORB_CUTOFF,
                        use_radial_correction=USE_RADIAL_CORRECTION,
                    )
                    for z in ATOMIC_NUMBERS
                }
            )

    return BoaWithGtos()


def breakdown(model: torch.nn.Module) -> dict:
    named = dict(model.named_parameters())

    def group(*, prefix: str = "", contains: str | None = None) -> int:
        return sum(
            t.numel()
            for k, t in named.items()
            if k.startswith(prefix) and (contains is None or contains in k)
        )

    stack_linears = sum(
        group(prefix=f"boa_net.boa_stack.blocks.{i}.linear{n}.") for i in range(N_BLOCKS) for n in range(3)
    )
    components = {
        "boa_stack_params": group(prefix="boa_net.boa_stack."),
        "boa_stable_linears_params": stack_linears,
        "boa_l2_nonlinearity_params": group(prefix="boa_net.boa_stack.", contains=".l2_nonlinearity."),
        "boa_message_pass_params": group(prefix="boa_net.boa_stack.", contains=".message_pass."),
        "boa_edge_update_params": group(prefix="boa_net.boa_stack.", contains=".edge_update."),
        "boa_node_embedding_params": group(prefix="boa_net.node_embedding."),
        "boa_edge_embedding_params": group(prefix="boa_net.edge_embedding."),
        "boa_initial_guess_params": group(prefix="boa_net.initial_guess_module."),
        "boa_gto_radial_correction_params": group(prefix="gto_dict."),
    }
    # Fixed (requires_grad=False) constants registered as Parameters: the
    # per-element L2 function-norm scales live under '.norm1.'.
    fixed_norms = group(prefix="boa_net.boa_stack.", contains=".norm1.")
    total = sum(t.numel() for t in named.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (
        components["boa_stable_linears_params"]
        + components["boa_l2_nonlinearity_params"]
        + components["boa_message_pass_params"]
        + components["boa_edge_update_params"]
        + fixed_norms
        == components["boa_stack_params"]
    ), "fine boa_stack breakdown incomplete"
    coarse = (
        components["boa_stack_params"]
        + components["boa_node_embedding_params"]
        + components["boa_edge_embedding_params"]
        + components["boa_initial_guess_params"]
        + components["boa_gto_radial_correction_params"]
    )
    assert coarse == total, "breakdown does not cover all parameters"
    return {
        "total_params": total,
        "trainable_params": trainable,
        **components,
    }


def validate_l4_is_default() -> str:
    """The L=4 filtered basis must equal the unfiltered default constructor."""
    filtered = build_basis_info(4)
    default = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        np.array(ATOMIC_NUMBERS, dtype=np.uint8),
        DEFAULT_BASIS,
        beta=None,
        even_tempered=False,
        uncontracted=True,
    )
    assert [str(i) for i in filtered.irreps_per_atom] == [str(i) for i in default.irreps_per_atom]
    assert int(np.max(default.l_per_basis_func)) == 4
    return (
        "L=4 filtered basis == BasisInfo.from_atomic_numbers_with_even_tempered_basis"
        "(def2-qzvppd, uncontracted) in per-element irreps; default basis max l = 4 "
        "(C/N/O/F carry g shells, H stops at f), so the L=4 row is the default model"
    )


def main() -> None:
    rows = []
    import contextlib
    import io

    for l_value in range(5):
        # GTOs prints "Using radial correction..." on construction; keep stdout JSON-clean.
        with contextlib.redirect_stdout(io.StringIO()):
            basis_info = build_basis_info(l_value)
            model = build_model(basis_info)
        row = {
            "model": "BOA",
            "L": l_value,
            "basis_max_l": int(np.max(basis_info.l_per_basis_func)),
        }
        row.update(breakdown(model))
        rows.append(row)
    payload = {
        "rows": rows,
        "defaults": (
            "boa/configs defaults: basis def2-qzvppd uncontracted (qm9_vasp_small), "
            "elements Z=[1,6,7,8,9], 4 BoaBlocks, hidden_channels=32, "
            "num_orbitals=8, orb_cutoff=3.0, radial correction on; L = max shell "
            "angular momentum of the basis (L=4 is the unfiltered default)"
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        payload["validation"] = validate_l4_is_default()
    json.dump(payload, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
