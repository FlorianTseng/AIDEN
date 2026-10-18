"""Build model-param-compare.csv: AIDEN vs ChargE3Net vs BOA parameter counts, L = 0..4.

Runs the per-model counters as isolated subprocesses (the models must not
share one interpreter: AIDEN vendors a minimal ``aiden.e3nn`` subpackage while
ChargE3Net and BOA import the full ``e3nn`` from parameter-count/libs, and each
counter sets up its own sys.path over the read-only source repositories), then
merges the JSON payloads into one CSV.

Usage (from parameter-count/):  python3 run_compare.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

PARAM_QUANT_DIR = Path(__file__).resolve().parent

# Column order of the output CSV; models leave each other's columns empty.
COLUMNS = [
    "model",
    "L",
    "total_params",
    "trainable_params",
    "total_params_M",
    # AIDEN breakdown (aiden package)
    "encoder_params",
    "decoder_params",
    "gie_params",
    "ace_params",
    "tece_params",
    "encoder_norms_params",
    "encoder_final_norm_params",
    "coeff_head_params",
    "radial_correction_params",
    "initial_guess_params",
    # ChargE3Net breakdown (charge3net/src/charge3net/models/e3.py)
    "atom_model_params",
    "probe_model_params",
    "c3n_atom_convs_params",
    "c3n_atom_gates_params",
    "c3n_probe_convs_params",
    "c3n_probe_gates_params",
    "c3n_probe_readout_params",
    # BOA breakdown (boa/boa/model: BOA net + per-element GTOs)
    "boa_stack_params",
    "boa_stable_linears_params",
    "boa_l2_nonlinearity_params",
    "boa_message_pass_params",
    "boa_edge_update_params",
    "boa_node_embedding_params",
    "boa_edge_embedding_params",
    "boa_initial_guess_params",
    "boa_gto_radial_correction_params",
    # Provenance
    "config",
]


def run_counter(script_name: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(PARAM_QUANT_DIR / script_name)],
        capture_output=True,
        text=True,
        cwd=PARAM_QUANT_DIR,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)


def main() -> None:
    aiden = run_counter("count_aiden.py")
    charge3net = run_counter("count_charge3net.py")
    boa = run_counter("count_boa.py")

    rows = []
    for payload, config_note in (
        (aiden, "L=M; aiden constructor defaults (= configs/pbe-l{L}.json)"),
        (charge3net, "lmax=L; charge3net e3_density.yaml defaults (mul=500, 3 interactions, gaussian, num_basis=20)"),
        (boa, "L=max basis angular momentum; boa defaults (def2-qzvppd uncontracted filtered to l<=L, 4 blocks, hidden 32); L=4 is the unfiltered default"),
    ):
        for row in payload["rows"]:
            merged = {column: "" for column in COLUMNS}
            merged.update({k: v for k, v in row.items() if k in COLUMNS})
            merged["total_params_M"] = f"{row['total_params'] / 1e6:.3f}"
            merged["config"] = config_note
            rows.append(merged)

    output = PARAM_QUANT_DIR / "model-param-compare.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    details = {
        "aiden": aiden,
        "charge3net": charge3net,
        "boa": boa,
    }
    (PARAM_QUANT_DIR / "model-param-compare-details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    print(f"wrote {output}")
    for model_name, payload in details.items():
        print(f"[{model_name}] {payload['validation']}")
    for row in rows:
        print(
            f"{row['model']:>11}  L={row['L']}  "
            f"{int(row['total_params']):>10,d} params  ({row['total_params_M']} M)"
        )


if __name__ == "__main__":
    main()
