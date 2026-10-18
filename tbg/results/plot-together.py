from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.linewidth'] = 1.1
mpl.rcParams['xtick.major.width'] = 1.1
mpl.rcParams['ytick.major.width'] = 1.1

HERE = Path(__file__).resolve().parent
BANDS = HERE / "bands"
STAGES = ("reference", "ml_charge_norm_paw_ref", "ml_raw_paw_ref")

COLORS = {
    "reference": "#D19C1D",
    "ml_raw_paw_ref": "#6594B8",
    "ml_charge_norm_paw_ref": "#592E83"
}

LINESTYLES = {
    "reference": "-",
    "ml_raw_paw_ref": (0, (3, 1.5)),
    "ml_charge_norm_paw_ref": (0, (1, 1.5))
}

ZORDERS = {
    "reference": 2,
    "ml_charge_norm_paw_ref": 3,
    "ml_raw_paw_ref": 4
}

LEGEND_LABELS = {
    "reference": "DFT",
    "ml_raw_paw_ref": r"AIDEN",
    "ml_charge_norm_paw_ref": r"AIDEN (norm.)"
}

PRINT_LABELS = {
    "ml_raw_paw_ref": "Aiden",
    "ml_charge_norm_paw_ref": "Aiden (norm.)"
}

KPATH_LABELS = ["Γ", "X", "M", "Γ", "Z", "R", "A", "Z"]

def fermi_energy(outcar: Path) -> float:
    values = re.findall(r"E-fermi\s*:\s*([-+0-9.Ee]+)", outcar.read_text(encoding="utf-8", errors="replace"))
    if not values:
        raise ValueError(f"no E-fermi value in {outcar}")
    return float(values[-1])

def read_eigenval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"short EIGENVAL: {path}")
    header = lines[5].split()
    if len(header) < 3:
        raise ValueError(f"invalid EIGENVAL dimensions in {path}")
    nkpoints, nbands = int(header[1]), int(header[2])
    kpoints = []
    energies = []
    cursor = 6
    while len(kpoints) < nkpoints and cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        fields = lines[cursor].split()
        if len(fields) < 4:
            raise ValueError(f"bad k-point line in {path}: {lines[cursor]!r}")
        kpoints.append([float(value) for value in fields[:3]])
        cursor += 1
        band_energies = []
        for _ in range(nbands):
            fields = lines[cursor].split()
            if len(fields) < 2:
                raise ValueError(f"bad band line in {path}: {lines[cursor]!r}")
            band_energies.append(float(fields[1]))
            cursor += 1
        energies.append(band_energies)
    if len(kpoints) != nkpoints:
        raise ValueError(f"expected {nkpoints} k-points, parsed {len(kpoints)} from {path}")
    return np.asarray(kpoints), np.asarray(energies)

def compare_metrics(reference: np.ndarray, candidate: np.ndarray, fermi: float) -> dict[str, float | int]:
    nk = min(reference.shape[0], candidate.shape[0])
    nb = min(reference.shape[1], candidate.shape[1])
    ref = reference[:nk, :nb]
    trial = candidate[:nk, :nb]
    mask = np.abs(ref - fermi) <= 5.0
    if not np.any(mask):
        raise ValueError("reference has no states in Fermi +/- 5 eV")
    raw_delta = trial[mask] - ref[mask]
    shift = -float(raw_delta.mean())
    aligned_delta = raw_delta + shift
    return {
        "states": int(mask.sum()),
        "bands_compared": int(nb),
        "kpoints_compared": int(nk),
        "rigid_shift_eV": shift,
        "raw_mae_eV": float(np.abs(raw_delta).mean()),
        "raw_rmse_eV": float(np.sqrt(np.mean(raw_delta ** 2))),
        "raw_max_abs_eV": float(np.abs(raw_delta).max()),
        "aligned_mae_eV": float(np.abs(aligned_delta).mean()),
        "aligned_rmse_eV": float(np.sqrt(np.mean(aligned_delta ** 2))),
        "aligned_max_abs_eV": float(np.abs(aligned_delta).max())
    }

def formula_to_mathtext(formula: str) -> str:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    parts = []
    for element, number in tokens:
        if number and number != "1":
            parts.append(f"{element}$_{number}$")
        else:
            parts.append(element)
    return "".join(parts)

def print_metrics(sample_name: str, formula: str, fermi: float, metrics_all: dict[str, dict[str, float | int]]) -> None:
    print("=" * 92)
    print(f"{sample_name}    Formula: {formula}    Reference E_F: {fermi:.8f} eV")
    print("-" * 92)
    for stage in ("ml_raw_paw_ref", "ml_charge_norm_paw_ref"):
        m = metrics_all[stage]
        print(f"{PRINT_LABELS[stage]}")
        print(f"  states              : {m['states']}")
        print(f"  bands compared      : {m['bands_compared']}")
        print(f"  k-points compared   : {m['kpoints_compared']}")
        print(f"  rigid shift         : {m['rigid_shift_eV']:+.8f} eV")
        print(f"  raw MAE             : {m['raw_mae_eV']:.8f} eV")
        print(f"  raw RMSE            : {m['raw_rmse_eV']:.8f} eV")
        print(f"  raw max |error|     : {m['raw_max_abs_eV']:.8f} eV")
        print(f"  aligned MAE         : {m['aligned_mae_eV']:.8f} eV")
        print(f"  aligned RMSE        : {m['aligned_rmse_eV']:.8f} eV")
        print(f"  aligned max |error| : {m['aligned_max_abs_eV']:.8f} eV")
        print("-" * 92)

def main() -> None:
    sample_dirs = sorted(path for path in BANDS.iterdir() if path.is_dir())
    if len(sample_dirs) != 10:
        raise ValueError(f"expected 10 samples, found {len(sample_dirs)}")

    fig, axes = plt.subplots(2, 5, figsize=(7.5, 2.5), sharey=True)
    axes = np.asarray(axes)

    all_metrics = {}

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
        ax.tick_params(axis='both', which='major', direction='out', width=1.1, top=False, right=False)
        ax.minorticks_off()
        ax.set_ylim(-6, 6)

    for idx, sample_dir in enumerate(sample_dirs):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]

        sample_name = sample_dir.name
        formula = sample_name.split("_", 1)[1] if "_" in sample_name else sample_name

        reference_dir = sample_dir / "reference"
        reference_kpoints, reference_bands = read_eigenval(reference_dir / "EIGENVAL")
        fermi = fermi_energy(reference_dir / "OUTCAR")

        nk = reference_bands.shape[0]
        axis = np.arange(nk, dtype=float)
        xticks = np.linspace(0, nk - 1, len(KPATH_LABELS))

        for x in xticks:
            ax.axvline(x, color="0.85", linewidth=0.6, zorder=0)

        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)

        sample_metrics = {}

        for stage in STAGES:
            stage_dir = sample_dir / stage
            kpoints, bands = read_eigenval(stage_dir / "EIGENVAL")

            if kpoints.shape != reference_kpoints.shape or not np.allclose(kpoints, reference_kpoints, atol=1.0e-10):
                raise ValueError(f"k-point mismatch for {sample_name}/{stage}")

            if stage == "reference":
                shift = 0.0
            else:
                metrics = compare_metrics(reference_bands, bands, fermi)
                sample_metrics[stage] = metrics
                shift = float(metrics["rigid_shift_eV"])

            nb = min(reference_bands.shape[1], bands.shape[1])

            for band in range(nb):
                ax.plot(axis, bands[:nk, band] + shift - fermi, color=COLORS[stage], linestyle=LINESTYLES[stage], linewidth=1.1, zorder=ZORDERS[stage])

        all_metrics[sample_name] = sample_metrics
        print_metrics(sample_name, formula, fermi, sample_metrics)

        ax.set_xlim(axis[0], axis[-1])
        ax.set_xticks(xticks)

        if row == 1:
            ax.set_xticklabels(KPATH_LABELS)
        else:
            ax.set_xticklabels([])

        if col != 0:
            ax.tick_params(labelleft=False)

        if row != 1:
            ax.tick_params(labelbottom=False)

        panel_label = chr(ord('a') + idx)
        ax.text(0.03, 0.96, f"{panel_label}) {formula_to_mathtext(formula)}", transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=1000)

    print("=" * 92)
    print("Aggregate comparison over all samples")
    print("=" * 92)

    for stage in ("ml_raw_paw_ref", "ml_charge_norm_paw_ref"):
        raw_mae = np.asarray([all_metrics[name][stage]["raw_mae_eV"] for name in all_metrics], dtype=float)
        raw_rmse = np.asarray([all_metrics[name][stage]["raw_rmse_eV"] for name in all_metrics], dtype=float)
        raw_max = np.asarray([all_metrics[name][stage]["raw_max_abs_eV"] for name in all_metrics], dtype=float)
        aligned_mae = np.asarray([all_metrics[name][stage]["aligned_mae_eV"] for name in all_metrics], dtype=float)
        aligned_rmse = np.asarray([all_metrics[name][stage]["aligned_rmse_eV"] for name in all_metrics], dtype=float)
        aligned_max = np.asarray([all_metrics[name][stage]["aligned_max_abs_eV"] for name in all_metrics], dtype=float)
        shifts = np.asarray([all_metrics[name][stage]["rigid_shift_eV"] for name in all_metrics], dtype=float)

        print(f"{PRINT_LABELS[stage]}")
        print(f"  systems                  : {len(sample_dirs)}")
        print(f"  mean rigid shift         : {np.mean(shifts):+.8f} eV")
        print(f"  mean |rigid shift|       : {np.mean(np.abs(shifts)):.8f} eV")
        print(f"  median |rigid shift|     : {np.median(np.abs(shifts)):.8f} eV")
        print(f"  mean raw MAE             : {np.mean(raw_mae):.8f} eV")
        print(f"  median raw MAE           : {np.median(raw_mae):.8f} eV")
        print(f"  mean raw RMSE            : {np.mean(raw_rmse):.8f} eV")
        print(f"  mean raw max |error|     : {np.mean(raw_max):.8f} eV")
        print(f"  mean aligned MAE         : {np.mean(aligned_mae):.8f} eV")
        print(f"  median aligned MAE       : {np.median(aligned_mae):.8f} eV")
        print(f"  mean aligned RMSE        : {np.mean(aligned_rmse):.8f} eV")
        print(f"  mean aligned max |error| : {np.mean(aligned_max):.8f} eV")
        print("-" * 92)

    fig.supylabel(r"$E-E_{\rm F}$ [eV]", x=0.01, fontsize=10)

    legend_handles = [
        Line2D([0], [0], color=COLORS["reference"], linestyle=LINESTYLES["reference"], linewidth=1.1, label=LEGEND_LABELS["reference"]),
        Line2D([0], [0], color=COLORS["ml_raw_paw_ref"], linestyle=LINESTYLES["ml_raw_paw_ref"], linewidth=1.1, label=LEGEND_LABELS["ml_raw_paw_ref"]),
        Line2D([0], [0], color=COLORS["ml_charge_norm_paw_ref"], linestyle=LINESTYLES["ml_charge_norm_paw_ref"], linewidth=1.1, label=LEGEND_LABELS["ml_charge_norm_paw_ref"])
    ]

    legend = fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, handlelength=2.4, handletextpad=0.5, columnspacing=1.0)
    legend.set_zorder(2000)

    plt.subplots_adjust(left=0.07, right=0.995, bottom=0.13, top=0.850, wspace=0.090, hspace=0.150)

    fig.savefig(HERE / "bands_grid.pdf", bbox_inches='tight')
    fig.savefig(HERE / "bands_grid.jpg", dpi=1500, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()