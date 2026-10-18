from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

FS=10
DOS_SIGMA=0.10
DOS_NGRID=1600

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = FS
mpl.rcParams['axes.labelsize'] = FS
mpl.rcParams['xtick.labelsize'] = FS
mpl.rcParams['ytick.labelsize'] = FS
mpl.rcParams['legend.fontsize'] = FS
mpl.rcParams['axes.linewidth'] = 1.1
mpl.rcParams['xtick.major.width'] = 1.1
mpl.rcParams['ytick.major.width'] = 1.1
mpl.rcParams['xtick.minor.width'] = 1.1
mpl.rcParams['ytick.minor.width'] = 1.1

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "single_band_figures"

STAGES = ("reference", "ml_charge_norm_paw_ref", "ml_raw_paw_ref")

COLORS = {
    "reference": "#D19C1D",
    "ml_raw_paw_ref": "#6594B8",
    "ml_charge_norm_paw_ref": "#592E83"
}

LINESTYLES = {
    "reference": "-",
    "ml_raw_paw_ref": (0, (3.0, 1.4)),
    "ml_charge_norm_paw_ref": (0, (1.0, 1.2))
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

FALLBACK_KPATH_LABELS = ["Γ", "X", "M", "Γ", "Z", "R", "A", "Z"]

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
        kpoints.append([float(fields[0]), float(fields[1]), float(fields[2])])
        cursor += 1
        band_energies = []
        for _ in range(nbands):
            if cursor >= len(lines):
                raise ValueError(f"unexpected end of EIGENVAL: {path}")
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

def pseudo_dos_from_bands(bands: np.ndarray, shift: float, fermi: float, emin: float=-6.0, emax: float=6.0, sigma: float=DOS_SIGMA, ngrid: int=DOS_NGRID) -> tuple[np.ndarray, np.ndarray]:
    energy_grid = np.linspace(emin, emax, ngrid)
    eigenvalues = (bands + shift - fermi).ravel()
    lower = emin - 5.0 * sigma
    upper = emax + 5.0 * sigma
    eigenvalues = eigenvalues[(eigenvalues >= lower) & (eigenvalues <= upper)]
    if eigenvalues.size == 0:
        return energy_grid, np.zeros_like(energy_grid)
    delta = energy_grid[:, None] - eigenvalues[None, :]
    dos = np.exp(-0.5 * (delta / sigma) ** 2).sum(axis=1)
    dos /= sigma * np.sqrt(2.0 * np.pi)
    maximum = float(np.max(dos))
    if maximum > 0:
        dos /= maximum
    return energy_grid, dos

def formula_to_mathtext(formula: str) -> str:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    parts = []
    for element, number in tokens:
        if number and number != "1":
            parts.append(f"{element}$_{{{number}}}$")
        else:
            parts.append(element)
    return "".join(parts)

def clean_formula_from_directory(name: str) -> str:
    match = re.match(r"^\d+[_-](.+)$", name)
    if match:
        return match.group(1)
    return name

def find_sample_dirs(root: Path) -> list[Path]:
    samples = []
    seen = set()
    for reference_dir in root.rglob("reference"):
        if not reference_dir.is_dir():
            continue
        sample_dir = reference_dir.parent
        if OUTPUT in sample_dir.parents or sample_dir == OUTPUT:
            continue
        resolved = sample_dir.resolve()
        if resolved in seen:
            continue
        required = [
            sample_dir / "reference" / "EIGENVAL",
            sample_dir / "reference" / "OUTCAR",
            sample_dir / "ml_raw_paw_ref" / "EIGENVAL",
            sample_dir / "ml_charge_norm_paw_ref" / "EIGENVAL"
        ]
        if all(path.is_file() for path in required):
            seen.add(resolved)
            samples.append(sample_dir)
    return sorted(samples, key=lambda path: str(path).lower())

def normalize_klabel(label: str) -> str:
    value = label.strip()
    value = value.replace("Γ", "Gamma")
    value = value.replace("\\Gamma", "Gamma")
    value = value.replace("$", "")
    low = value.lower()
    if low == "gamma":
        return "Γ"
    return value

def extract_k_label(line: str) -> tuple[np.ndarray, str] | None:
    raw = line.strip()
    if not raw:
        return None
    label = ""
    if "!" in raw:
        coord_text, label = raw.split("!", 1)
    elif "#" in raw:
        coord_text, label = raw.split("#", 1)
    else:
        coord_text = raw
    fields = coord_text.split()
    if len(fields) < 3:
        return None
    try:
        coords = np.asarray([float(fields[0]), float(fields[1]), float(fields[2])], dtype=float)
    except ValueError:
        return None
    if not label and len(fields) >= 4:
        label = fields[3]
    return coords, normalize_klabel(label)

def read_line_mode_kpath(kpoints_file: Path, nk: int) -> tuple[np.ndarray, list[str]] | None:
    if not kpoints_file.is_file():
        return None
    lines = kpoints_file.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        return None
    try:
        points_per_segment = int(lines[1].split()[0])
    except Exception:
        return None
    if "line" not in lines[2].lower():
        return None
    entries = []
    for line in lines[4:]:
        item = extract_k_label(line)
        if item is not None:
            entries.append(item)
    if len(entries) < 2 or len(entries) % 2 != 0:
        return None
    nsegments = len(entries) // 2
    expected_nk = nsegments * points_per_segment
    if expected_nk != nk:
        return None
    tick_positions = [0.0]
    tick_labels = [entries[0][1] if entries[0][1] else ""]
    for seg in range(nsegments):
        end_index = (seg + 1) * points_per_segment - 1
        end_label = entries[2 * seg + 1][1]
        tick_positions.append(float(end_index))
        tick_labels.append(end_label if end_label else "")
    cleaned_positions = [tick_positions[0]]
    cleaned_labels = [tick_labels[0]]
    for pos, lab in zip(tick_positions[1:], tick_labels[1:]):
        if abs(pos - cleaned_positions[-1]) < 1.0e-12:
            if lab and lab != cleaned_labels[-1]:
                cleaned_labels[-1] = lab
        else:
            cleaned_positions.append(pos)
            cleaned_labels.append(lab)
    return np.asarray(cleaned_positions, dtype=float), cleaned_labels

def determine_kpath(sample_dir: Path, nk: int) -> tuple[np.ndarray, list[str]]:
    candidates = [
        sample_dir / "reference" / "KPOINTS",
        sample_dir / "KPOINTS"
    ]
    for candidate in candidates:
        parsed = read_line_mode_kpath(candidate, nk)
        if parsed is not None:
            positions, labels = parsed
            if len(positions) >= 2:
                return positions, labels
    if nk >= len(FALLBACK_KPATH_LABELS):
        return np.linspace(0, nk - 1, len(FALLBACK_KPATH_LABELS)), FALLBACK_KPATH_LABELS
    return np.linspace(0, nk - 1, min(nk, 6)), [""] * min(nk, 6)

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

def plot_sample(sample_dir: Path) -> None:
    sample_name = sample_dir.name
    formula = clean_formula_from_directory(sample_name)
    reference_dir = sample_dir / "reference"
    reference_kpoints, reference_bands = read_eigenval(reference_dir / "EIGENVAL")
    fermi = fermi_energy(reference_dir / "OUTCAR")
    nk = reference_bands.shape[0]
    axis = np.arange(nk, dtype=float)
    xticks, xticklabels = determine_kpath(sample_dir, nk)

    stage_bands = {"reference": reference_bands}
    stage_shifts = {"reference": 0.0}
    sample_metrics = {}

    for stage in ("ml_charge_norm_paw_ref", "ml_raw_paw_ref"):
        stage_dir = sample_dir / stage
        kpoints, bands = read_eigenval(stage_dir / "EIGENVAL")
        if kpoints.shape != reference_kpoints.shape or not np.allclose(kpoints, reference_kpoints, atol=1.0e-10):
            raise ValueError(f"k-point mismatch for {sample_name}/{stage}")
        metrics = compare_metrics(reference_bands, bands, fermi)
        sample_metrics[stage] = metrics
        stage_bands[stage] = bands
        stage_shifts[stage] = float(metrics["rigid_shift_eV"])

    figure, axes = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    panel, dos_panel = axes

    figure.patch.set_facecolor("white")
    panel.set_facecolor("white")
    dos_panel.set_facecolor("white")

    panel.set_position([0.1567, 0.1140, 0.5447, 0.8205])
    dos_panel.set_position([0.7251, 0.1140, 0.2253, 0.8205])

    panel.set_ylim(-6, 6)
    panel.set_xlim(axis[0], axis[-1])

    panel.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, width=1.1)
    panel.tick_params(axis='x', which='minor', top=False, bottom=False)
    panel.tick_params(axis='y', which='major', direction='in', left=True, right=True, width=1.1)
    panel.tick_params(axis='y', which='minor', direction='in', left=True, right=True, width=1.1)
    panel.yaxis.set_minor_locator(AutoMinorLocator(2))

    dos_panel.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, width=1.1)
    dos_panel.tick_params(axis='x', which='minor', direction='in', top=True, bottom=True, width=1.1)
    dos_panel.tick_params(axis='y', which='major', direction='in', left=True, right=True, labelleft=False, labelright=False, width=1.1)
    dos_panel.tick_params(axis='y', which='minor', direction='in', left=True, right=True, width=1.1)
    dos_panel.xaxis.set_minor_locator(AutoMinorLocator(2))
    dos_panel.yaxis.set_minor_locator(AutoMinorLocator(2))

    for value in xticks:
        panel.axvline(value, color="#999999", linewidth=1.0, zorder=1)

    panel.axhline(0.0, color="#8b102b", linewidth=1.0, linestyle="--", zorder=1)

    for stage in STAGES:
        bands = stage_bands[stage]
        shift = stage_shifts[stage]
        nb = min(reference_bands.shape[1], bands.shape[1])
        for band in range(nb):
            panel.plot(axis, bands[:nk, band] + shift - fermi, color=COLORS[stage], linestyle=LINESTYLES[stage], linewidth=1.1, zorder=ZORDERS[stage], alpha=1.0, solid_capstyle='butt', dash_capstyle='butt', solid_joinstyle='round', dash_joinstyle='round', antialiased=True)

    for stage in STAGES:
        energy_grid, pseudo_dos = pseudo_dos_from_bands(stage_bands[stage], stage_shifts[stage], fermi)
        dos_panel.plot(pseudo_dos, energy_grid, color=COLORS[stage], linestyle=LINESTYLES[stage], linewidth=1.1, zorder=ZORDERS[stage], alpha=1.0, solid_capstyle='butt', dash_capstyle='butt', solid_joinstyle='round', dash_joinstyle='round', antialiased=True)

    dos_panel.set_xlim(0, 1.05)
    dos_panel.set_xticks([0.0, 0.5, 1.0])

    panel.set_xticks(xticks)
    panel.set_xticklabels(xticklabels)
    panel.set_ylabel(r"$E-E_{\rm Fermi}$ [eV]")
    dos_panel.set_xlabel("Pseudo-DOS [a.u.]")

    panel.text(0.035, 0.965, formula_to_mathtext(formula), transform=panel.transAxes, ha='left', va='top', fontsize=FS, zorder=3000)

    legend_handles = [
        Line2D([0], [0], color=COLORS["reference"], linestyle=LINESTYLES["reference"], linewidth=1.1, label=LEGEND_LABELS["reference"]),
        Line2D([0], [0], color=COLORS["ml_raw_paw_ref"], linestyle=LINESTYLES["ml_raw_paw_ref"], linewidth=1.1, label=LEGEND_LABELS["ml_raw_paw_ref"]),
        Line2D([0], [0], color=COLORS["ml_charge_norm_paw_ref"], linestyle=LINESTYLES["ml_charge_norm_paw_ref"], linewidth=1.1, label=LEGEND_LABELS["ml_charge_norm_paw_ref"])
    ]

    legend = panel.legend(handles=legend_handles, loc="upper right", frameon=True, fancybox=False, framealpha=0.72, facecolor="white", edgecolor="0.65", borderpad=0.35, labelspacing=0.25, handlelength=2.1, handletextpad=0.45, borderaxespad=0.45)
    legend.get_frame().set_linewidth(0.8)
    legend.set_zorder(2500)

    for a in axes:
        a.set_axisbelow(False)
        a.patch.set_zorder(0)
        for spine in a.spines.values():
            spine.set_linewidth(1.1)
            spine.set_zorder(4000)

    panel.tick_params(axis='x', which='minor', bottom=False, top=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_name)

    figure.savefig(OUTPUT / f"{output_name}_bands_pdos.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / f"{output_name}_bands_pdos.jpg", dpi=1500, bbox_inches="tight")

    plt.close(figure)

    print_metrics(sample_name, formula, fermi, sample_metrics)

def main() -> None:
    sample_dirs = find_sample_dirs(HERE)

    if not sample_dirs:
        raise SystemExit(f"No valid band-result systems were found below {HERE}")

    print("=" * 92)
    print(f"Found {len(sample_dirs)} valid systems")
    print(f"Output directory: {OUTPUT}")
    print(f"Pseudo-DOS Gaussian sigma: {DOS_SIGMA:.3f} eV")
    print("=" * 92)

    succeeded = 0
    failed = []

    for sample_dir in sample_dirs:
        try:
            plot_sample(sample_dir)
            succeeded += 1
        except Exception as exc:
            failed.append((sample_dir, exc))
            print("=" * 92)
            print(f"FAILED: {sample_dir}")
            print(f"{type(exc).__name__}: {exc}")

    print("=" * 92)
    print(f"Completed: {succeeded}/{len(sample_dirs)}")
    print(f"Failed   : {len(failed)}")
    print(f"Figures  : {OUTPUT}")

    if failed:
        print("-" * 92)
        for path, exc in failed:
            print(f"{path}: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()