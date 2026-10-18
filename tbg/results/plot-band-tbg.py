from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


FS = 9
DOS_SIGMA = 0.05
DOS_NGRID = 1600
DOS_CHUNK_SIZE = 512
ENERGY_MIN = -2.5
ENERGY_MAX = 2.5

OUTER_WSPACE = 0.05
PAIR_WSPACE = 0.06
LEGEND_Y = 0.915
PANEL_LABEL_X = 0.035
PANEL_LABEL_Y = 0.965
FIGURE_SIZE = (7, 2)

INPUT_CSV = Path(__file__).resolve().parent / "bands_long.csv"
OUTPUT_BASENAME = "tbg-band"
ENERGY_MODE = "aligned"
EMIN = ENERGY_MIN
EMAX = ENERGY_MAX
SIGMA = DOS_SIGMA
DPI = 1200

CASE_ORDER = (
    "tbg_m01_21p79deg",
    "tbg_m02_13p17deg",
    "tbg_m03_9p43deg",
)

METHOD_ORDER = ("dft", "aiden", "charge3net")

COLORS = {
    "dft": "#D19C1D",
    "aiden": "#6594B8",
    "charge3net": "#592E83",
}

LINESTYLES = {
    "dft": "-",
    "aiden": (0, (3.0, 1.4)),
    "charge3net": (0, (1.0, 1.2)),
}

ZORDERS = {
    "dft": 2,
    "charge3net": 3,
    "aiden": 4,
}

LEGEND_LABELS = {
    "dft": "DFT",
    "aiden": "AIDEN",
    "charge3net": "ChargE3Net",
}

ENERGY_COLUMNS = {
    "aligned": "energy_aligned_minus_dft_fermi_ev",
    "raw": "energy_minus_dft_fermi_ev",
    "own": "energy_minus_own_fermi_ev",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": FS,
            "axes.labelsize": FS,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": FS,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.minor.width": 1.1,
            "xtick.minor.width": 1.1,
            "ytick.minor.width": 1.1,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_data(csv_path: Path, energy_column: str) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Band data not found: {csv_path}")

    required = {
        "method",
        "case",
        "angle_deg",
        "atoms",
        "kpoint_index",
        "kpoint_path_coordinate",
        "special_point",
        "band_index",
        energy_column,
    }
    data = pd.read_csv(csv_path, low_memory=False)
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    missing_cases = [case for case in CASE_ORDER if case not in set(data["case"])]
    missing_methods = [
        method for method in METHOD_ORDER if method not in set(data["method"])
    ]
    if missing_cases:
        raise ValueError(f"Missing TBG cases: {', '.join(missing_cases)}")
    if missing_methods:
        raise ValueError(f"Missing methods: {', '.join(missing_methods)}")

    data = data[data["case"].isin(CASE_ORDER) & data["method"].isin(METHOD_ORDER)]
    if data[energy_column].isna().any():
        raise ValueError(f"NaN values found in {energy_column}")
    return data.copy()


def dos_from_eigenvalues(
    eigenvalues: np.ndarray,
    emin: float,
    emax: float,
    sigma: float = DOS_SIGMA,
    ngrid: int = DOS_NGRID,
) -> tuple[np.ndarray, np.ndarray]:
    if sigma <= 0:
        raise ValueError("DOS sigma must be positive")
    energy_grid = np.linspace(emin, emax, ngrid)
    eigenvalues = np.asarray(eigenvalues, dtype=float).ravel()
    lower = emin - 5.0 * sigma
    upper = emax + 5.0 * sigma
    eigenvalues = eigenvalues[
        np.isfinite(eigenvalues)
        & (eigenvalues >= lower)
        & (eigenvalues <= upper)
    ]
    if eigenvalues.size == 0:
        return energy_grid, np.zeros_like(energy_grid)

    # Exact Gaussian sum in bounded-memory chunks.  Building the full
    # (energy-grid x eigenvalue) matrix can require several hundred MB.
    dos = np.zeros_like(energy_grid)
    for start in range(0, eigenvalues.size, DOS_CHUNK_SIZE):
        chunk = eigenvalues[start : start + DOS_CHUNK_SIZE]
        delta = energy_grid[:, None] - chunk[None, :]
        dos += np.exp(-0.5 * (delta / sigma) ** 2).sum(axis=1)
    dos /= sigma * np.sqrt(2.0 * np.pi)
    maximum = float(np.max(dos))
    if maximum > 0:
        dos /= maximum
    return energy_grid, dos


def method_band_segments(
    method_data: pd.DataFrame, energy_column: str
) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for _, band_data in method_data.groupby("band_index", sort=True):
        band_data = band_data.sort_values("kpoint_index")
        x = band_data["kpoint_path_coordinate"].to_numpy(dtype=float)
        y = band_data[energy_column].to_numpy(dtype=float)
        segments.append(np.column_stack((x, y)))
    return segments


def high_symmetry_ticks(case_data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    points = (
        case_data.loc[
            case_data["special_point"].notna(),
            ["kpoint_path_coordinate", "special_point"],
        ]
        .drop_duplicates(subset="kpoint_path_coordinate")
        .sort_values("kpoint_path_coordinate")
    )
    positions = points["kpoint_path_coordinate"].to_numpy(dtype=float)
    labels = [
        r"$\Gamma$" if str(label).upper() == "GAMMA" else str(label)
        for label in points["special_point"]
    ]
    return positions, labels


def style_band_axis(
    ax: mpl.axes.Axes,
    xticks: np.ndarray,
    xticklabels: list[str],
    emin: float,
    emax: float,
) -> None:
    ax.set_xlim(float(xticks[0]), float(xticks[-1]))
    ax.set_ylim(emin, emax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        axis="x", which="major", direction="in", top=True, bottom=True, width=1.1
    )
    ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    ax.tick_params(
        axis="y", which="major", direction="in", left=True, right=True, width=1.1
    )
    ax.tick_params(
        axis="y", which="minor", direction="in", left=True, right=True, width=1.1
    )
    for value in xticks:
        ax.axvline(value, color="#999999", linewidth=0.9, zorder=1)
    ax.axhline(0.0, color="#8b102b", linewidth=1.0, linestyle="--", zorder=1)


def style_dos_axis(ax: mpl.axes.Axes, emin: float, emax: float) -> None:
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(emin, emax)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["", "0.5", "1.0"])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        axis="x", which="major", direction="in", top=True, bottom=True, width=1.1
    )
    ax.tick_params(
        axis="x", which="minor", direction="in", top=True, bottom=True, width=1.1
    )
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=True,
        labelleft=False,
        labelright=False,
        width=1.1,
    )
    ax.tick_params(
        axis="y", which="minor", direction="in", left=True, right=True, width=1.1
    )
    ax.axhline(0.0, color="#8b102b", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xlabel("DOS [a.u.]", labelpad=4)


def align_band_dos_pair(band_ax: mpl.axes.Axes, dos_ax: mpl.axes.Axes) -> None:
    """Make a band/DOS pair use one identical vertical data region.

    The nested GridSpec normally gives the two axes the same height.  Setting
    the DOS position explicitly keeps that invariant if the layout is later
    adjusted (for example, to accommodate different tick labels) and makes
    the horizontal Fermi lines directly comparable.
    """
    band_position = band_ax.get_position()
    dos_position = dos_ax.get_position()
    dos_ax.set_position(
        (
            dos_position.x0,
            band_position.y0,
            dos_position.width,
            band_position.height,
        )
    )


def plot_case(
    band_ax: mpl.axes.Axes,
    dos_ax: mpl.axes.Axes,
    case_data: pd.DataFrame,
    energy_column: str,
    emin: float,
    emax: float,
    sigma: float,
    panel_label: str,
) -> None:
    xticks, xticklabels = high_symmetry_ticks(case_data)
    if len(xticks) < 2:
        raise ValueError("At least two high-symmetry k-point ticks are required")

    style_band_axis(band_ax, xticks, xticklabels, emin, emax)
    style_dos_axis(dos_ax, emin, emax)

    angle = float(case_data["angle_deg"].iloc[0])
    atoms = int(case_data["atoms"].iloc[0])
    band_ax.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y,
        rf"{panel_label}) {angle:.2f}$^\circ$ ($N={atoms}$)",
        transform=band_ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS,
        fontweight="normal",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=3000,
    )

    for method in METHOD_ORDER:
        method_data = case_data[case_data["method"] == method]
        segments = method_band_segments(method_data, energy_column)
        collection = LineCollection(
            segments,
            colors=COLORS[method],
            linestyles=LINESTYLES[method],
            linewidths=1.0,
            zorder=ZORDERS[method],
            alpha=1.0,
            antialiaseds=True,
        )
        band_ax.add_collection(collection)

        eigenvalues = method_data[energy_column].to_numpy(dtype=float)
        energy_grid, dos = dos_from_eigenvalues(
            eigenvalues, emin=emin, emax=emax, sigma=sigma
        )
        dos_ax.plot(
            dos,
            energy_grid,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.0,
            zorder=ZORDERS[method],
            solid_capstyle="butt",
            dash_capstyle="butt",
            solid_joinstyle="round",
            dash_joinstyle="round",
            antialiased=True,
        )

    for ax in (band_ax, dos_ax):
        ax.set_facecolor("white")
        ax.set_axisbelow(False)
        ax.patch.set_zorder(0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_zorder(4000)


def make_figure(
    data: pd.DataFrame,
    energy_mode: str,
    emin: float,
    emax: float,
    sigma: float,
) -> mpl.figure.Figure:
    if emin >= emax:
        raise ValueError("emin must be smaller than emax")

    energy_column = ENERGY_COLUMNS[energy_mode]
    figure = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    outer = figure.add_gridspec(
        1,
        3,
        left=0.060,
        right=0.992,
        bottom=0.170,
        top=0.835,
        wspace=OUTER_WSPACE,
    )

    band_axes: list[mpl.axes.Axes] = []
    panel_labels = ("a", "b", "c")

    for index, (case, panel_label) in enumerate(zip(CASE_ORDER, panel_labels)):
        inner = outer[index].subgridspec(
            1,
            2,
            width_ratios=(2.65, 1.0),
            wspace=PAIR_WSPACE,
        )
        if not band_axes:
            band_ax = figure.add_subplot(inner[0, 0])
        else:
            band_ax = figure.add_subplot(inner[0, 0], sharey=band_axes[0])
        # Share with the matching band axis directly, so their energy scale
        # remains tied together even if the panel layout changes later.
        dos_ax = figure.add_subplot(inner[0, 1], sharey=band_ax)
        band_axes.append(band_ax)

        case_data = data[data["case"] == case]
        plot_case(
            band_ax,
            dos_ax,
            case_data,
            energy_column,
            emin,
            emax,
            sigma,
            panel_label,
        )
        align_band_dos_pair(band_ax, dos_ax)

        if index > 0:
            band_ax.tick_params(axis="y", labelleft=False)

    if energy_mode == "own":
        ylabel = r"$E-E_{\rm Fermi}$ [eV]"
    elif energy_mode == "raw":
        ylabel = r"$E-E_{\rm Fermi}$ [eV]"
    else:
        ylabel = r"$E-E_{\rm Fermi}$ [eV]"
    band_axes[0].set_ylabel(ylabel)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.3,
            label=LEGEND_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, LEGEND_Y + 0.05),
        ncol=3,
        frameon=False,
        labelspacing=0.25,
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    return figure


def save_figure(figure: mpl.figure.Figure, output_basename: str, dpi: int) -> list[Path]:
    out_path = Path(output_basename).expanduser()
    if out_path.suffix:
        out_path = out_path.with_suffix("")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for obsolete_suffix in (".png", ".svg"):
        obsolete_path = out_path.with_suffix(obsolete_suffix)
        if obsolete_path.is_file():
            obsolete_path.unlink()

    pdf_path = out_path.with_suffix(".pdf")
    jpg_path = out_path.with_suffix(".jpg")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(
        jpg_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 95},
    )
    return [pdf_path, jpg_path]


def main() -> None:
    apply_style()
    energy_column = ENERGY_COLUMNS[ENERGY_MODE]
    data = load_data(INPUT_CSV, energy_column)
    figure = make_figure(
        data,
        energy_mode=ENERGY_MODE,
        emin=EMIN,
        emax=EMAX,
        sigma=SIGMA,
    )
    output_paths = save_figure(figure, OUTPUT_BASENAME, DPI)

    print(f"Input       : {INPUT_CSV.resolve()}")
    print(f"Energy mode : {ENERGY_MODE} ({energy_column})")
    print(f"Energy range: [{EMIN:.2f}, {EMAX:.2f}] eV")
    print(f"DOS sigma   : {SIGMA:.3f} eV")
    for path in output_paths:
        print(f"Saved       : {path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
