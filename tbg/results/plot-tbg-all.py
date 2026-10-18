from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

FS = 8.5
DPI = 1200
FIGURE_SIZE = (7.5, 1.55)

BASE_DIR = Path(__file__).resolve().parent
PROPERTY_CSV = BASE_DIR / "property_metrics.csv"
TIMING_CSV = BASE_DIR / "inference_timing.csv"
FORCES_CSV = BASE_DIR / "forces_per_atom.csv"
OUTPUT_BASENAME = BASE_DIR / "tbg-summary"
EXTRA_FIG_DIR = None

CASE_ORDER = ("tbg_m01_21p79deg", "tbg_m02_13p17deg", "tbg_m03_9p43deg")
MODEL_ORDER = ("aiden", "charge3net")

BAR_COLORS = ("#245297", "#537CB0", "#85C17E", "#A4D09D", "#C8E6C9")
LEFT_AXIS_COLOR = "#3C67A4"
RIGHT_AXIS_COLOR = "#95C98E"


def apply_style():
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Times New Roman"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = FS
    mpl.rcParams["axes.labelsize"] = FS
    mpl.rcParams["xtick.labelsize"] = FS
    mpl.rcParams["ytick.labelsize"] = FS
    mpl.rcParams["legend.fontsize"] = FS
    mpl.rcParams["axes.linewidth"] = 1.1
    mpl.rcParams["lines.linewidth"] = 1.1
    mpl.rcParams["xtick.major.width"] = 1.1
    mpl.rcParams["ytick.major.width"] = 1.1
    mpl.rcParams["xtick.minor.width"] = 1.1
    mpl.rcParams["ytick.minor.width"] = 1.1
    mpl.rcParams["pdf.fonttype"] = 42


def require_columns(df, columns, name):
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def load_property_data():
    df = pd.read_csv(PROPERTY_CSV)
    require_columns(df, {"case", "method", "smooth_charge_error_e", "energy_fixed_density_ev", "energy_error_mev_per_atom"}, PROPERTY_CSV.name)
    return df[df["case"].isin(CASE_ORDER) & df["method"].isin(MODEL_ORDER)].copy()


def load_timing_data():
    df = pd.read_csv(TIMING_CSV)
    require_columns(df, {"sample", "aiden_full_grid_inference_seconds", "charge3net_official_full_grid_per_sample_seconds", "dft_scf_wall_seconds"}, TIMING_CSV.name)
    return df[df["sample"].isin(CASE_ORDER)].copy()


def load_force_data():
    df = pd.read_csv(FORCES_CSV)
    require_columns(df, {"case", "method", "fx_ev_per_angstrom", "fy_ev_per_angstrom", "fz_ev_per_angstrom", "delta_fx_ev_per_angstrom", "delta_fy_ev_per_angstrom", "delta_fz_ev_per_angstrom"}, FORCES_CSV.name)
    return df[df["case"].isin(CASE_ORDER) & df["method"].isin(MODEL_ORDER)].copy()


def ordered_values(df, case_column, value_column, method=None):
    values = []
    for case in CASE_ORDER:
        part = df[df[case_column] == case]
        if method is not None:
            part = part[part["method"] == method]
        if len(part) != 1:
            raise ValueError(f"Expected one row for case={case}, method={method}, column={value_column}; got {len(part)}")
        values.append(float(part[value_column].iloc[0]))
    return np.asarray(values, dtype=float)


def force_component_means(df, method, delta):
    columns = ["delta_fx_ev_per_angstrom", "delta_fy_ev_per_angstrom", "delta_fz_ev_per_angstrom"] if delta else ["fx_ev_per_angstrom", "fy_ev_per_angstrom", "fz_ev_per_angstrom"]
    values = []
    for case in CASE_ORDER:
        part = df[(df["case"] == case) & (df["method"] == method)]
        if part.empty:
            raise ValueError(f"No force rows for case={case}, method={method}")
        values.append(float(np.mean(np.abs(part[columns].to_numpy(dtype=float)))))
    return np.asarray(values, dtype=float)


def add_headroom(ax, values, fraction=0.35):
    maximum = float(np.nanmax(values))
    if maximum > 0:
        ax.set_ylim(0.0, maximum * (1.0 + fraction))


def style_axes(ax, axr, x, xticklabels, right_log=False):
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if not right_log:
        axr.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.tick_params(axis="x", which="major", direction="in", top=True, bottom=True, width=1.1, zorder=100)
    ax.tick_params(axis="x", which="minor", direction="in", top=True, bottom=True, width=1.1, zorder=100)
    ax.tick_params(axis="y", which="major", direction="in", left=True, right=False, width=1.1, colors=LEFT_AXIS_COLOR, zorder=100)
    ax.tick_params(axis="y", which="minor", direction="in", left=True, right=False, width=1.1, colors=LEFT_AXIS_COLOR, zorder=100)
    axr.tick_params(axis="y", which="major", direction="in", left=False, right=True, width=1.1, colors=RIGHT_AXIS_COLOR, zorder=100)
    axr.tick_params(axis="y", which="minor", direction="in", left=False, right=True, width=1.1, colors=RIGHT_AXIS_COLOR, zorder=100)

    ax.spines["left"].set_color(LEFT_AXIS_COLOR)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(RIGHT_AXIS_COLOR)
    axr.spines["right"].set_linewidth(1.1)
    axr.spines["left"].set_visible(False)
    axr.spines["top"].set_visible(False)
    axr.spines["bottom"].set_visible(False)

    ax.yaxis.label.set_color(LEFT_AXIS_COLOR)
    axr.yaxis.label.set_color(RIGHT_AXIS_COLOR)
    axr.patch.set_visible(False)

    for spine in ax.spines.values():
        spine.set_zorder(100)
    for spine in axr.spines.values():
        spine.set_zorder(100)


def plot_four_bars(ax, x, left_1, left_2, right_1, right_2, left_ylabel, right_ylabel, xticklabels, panel_label, right_3=None, right_log=False):
    axr = ax.twinx()
    if right_3 is None:
        width = 0.19
        offsets = np.asarray([-1.5, -0.5, 0.5, 1.5]) * width
    else:
        width = 0.152
        offsets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]) * width

    ax.bar(x + offsets[0], left_1, width=width, color=BAR_COLORS[0], edgecolor="none", zorder=1)
    ax.bar(x + offsets[1], left_2, width=width, color=BAR_COLORS[1], edgecolor="none", zorder=1)
    axr.bar(x + offsets[2], right_1, width=width, color=BAR_COLORS[2], edgecolor="none", zorder=1)
    axr.bar(x + offsets[3], right_2, width=width, color=BAR_COLORS[3], edgecolor="none", zorder=1)
    if right_3 is not None:
        axr.bar(x + offsets[4], right_3, width=width, color=BAR_COLORS[4], edgecolor="none", zorder=1)

    ax.set_ylabel(left_ylabel, labelpad=1.5)
    axr.set_ylabel(right_ylabel, labelpad=1.5)

    style_axes(ax, axr, x, xticklabels, right_log=right_log)
    add_headroom(ax, np.concatenate((left_1, left_2)), 0.22)
    right_values = (right_1, right_2) if right_3 is None else (right_1, right_2, right_3)
    if right_log:
        positive = np.asarray(right_values, dtype=float)
        positive = positive[positive > 0]
        axr.set_yscale("log")
        axr.set_ylim(float(np.min(positive)) / 2.5, float(np.max(positive)) * 1.5)
    else:
        add_headroom(axr, np.concatenate(right_values), 0.22)

    ax.text(0.035, 0.965, f"{panel_label})", transform=ax.transAxes, ha="left", va="top", fontsize=FS, zorder=900)


def make_figure(property_data, timing_data, force_data):
    x = np.arange(len(CASE_ORDER))
    angles = np.asarray([21.79, 13.17, 9.43])
    xticklabels = [rf"{angle:.2f}$^\circ$" for angle in angles]

    charge_aiden = np.abs(ordered_values(property_data, "case", "smooth_charge_error_e", "aiden"))
    charge_charge3net = np.abs(ordered_values(property_data, "case", "smooth_charge_error_e", "charge3net"))
    time_aiden = ordered_values(timing_data, "sample", "aiden_full_grid_inference_seconds")
    time_charge3net = ordered_values(timing_data, "sample", "charge3net_official_full_grid_per_sample_seconds")
    time_dft = ordered_values(timing_data, "sample", "dft_scf_wall_seconds")

    energy_aiden = np.abs(ordered_values(property_data, "case", "energy_fixed_density_ev", "aiden"))
    energy_charge3net = np.abs(ordered_values(property_data, "case", "energy_fixed_density_ev", "charge3net"))
    energy_err_aiden = np.abs(ordered_values(property_data, "case", "energy_error_mev_per_atom", "aiden"))
    energy_err_charge3net = np.abs(ordered_values(property_data, "case", "energy_error_mev_per_atom", "charge3net"))

    force_aiden = force_component_means(force_data, "aiden", False)
    force_charge3net = force_component_means(force_data, "charge3net", False)
    force_err_aiden = force_component_means(force_data, "aiden", True)
    force_err_charge3net = force_component_means(force_data, "charge3net", True)

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    plt.subplots_adjust(left=0.070, right=0.985, bottom=0.145, top=0.815, wspace=0.44)

    plot_four_bars(axes[0], x, charge_aiden, charge_charge3net, time_aiden, time_charge3net, r"$|\Delta N_{\rm e}|$ [Unitless]", "Inference time [s]", xticklabels, "a", right_3=time_dft, right_log=True)
    plot_four_bars(axes[1], x, energy_aiden, energy_charge3net, energy_err_aiden, energy_err_charge3net, r"$|E|$ [eV]", r"$|\Delta E|$ [meV/atom]", xticklabels, "b")
    plot_four_bars(axes[2], x, force_aiden, force_charge3net, force_err_aiden, force_err_charge3net, r"$\langle |F_{i\alpha}| \rangle_{i,\alpha}$ [eV/$\mathrm{\AA}$]", r"$\langle |\Delta F_{i\alpha}| \rangle$ [eV/$\mathrm{\AA}$]", xticklabels, "c")

    legend_handles = [
        Patch(facecolor=BAR_COLORS[0], edgecolor="none", label="AIDEN"),
        Patch(facecolor=BAR_COLORS[1], edgecolor="none", label="ChargE3Net"),
        Patch(facecolor=BAR_COLORS[2], edgecolor="none", label="AIDEN"),
        Patch(facecolor=BAR_COLORS[3], edgecolor="none", label="ChargE3Net"),
        Patch(facecolor=BAR_COLORS[4], edgecolor="none", label="DFT")
    ]

    legend = fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=5, frameon=False, handlelength=1.25, handleheight=0.8, handletextpad=0.45, columnspacing=1.25, borderaxespad=0.0)
    legend.set_zorder(1000)

    return fig


def save_figure(fig):
    pdf_path = OUTPUT_BASENAME.with_suffix(".pdf")
    jpg_path = OUTPUT_BASENAME.with_suffix(".jpg")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(jpg_path, dpi=DPI, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 95})
    print(f"Saved: {pdf_path.resolve()}")
    print(f"Saved: {jpg_path.resolve()}")
    # fig.savefig('tbg-all.pdf', bbox_inches="tight")
    if EXTRA_FIG_DIR is not None:
        EXTRA_FIG_DIR.mkdir(parents=True, exist_ok=True)
        extra_pdf_path = EXTRA_FIG_DIR / "tbg-all.pdf"
        fig.savefig(extra_pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {extra_pdf_path.resolve()}")

def main():
    apply_style()
    property_data = load_property_data()
    timing_data = load_timing_data()
    force_data = load_force_data()
    fig = make_figure(property_data, timing_data, force_data)
    save_figure(fig)
    plt.show()


if __name__ == "__main__":
    main()