#!/usr/bin/env python3

import glob
import json
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

mpl.use("Agg")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 9.00

ROOT = os.path.dirname(os.path.abspath(__file__))
EXTRA_FIG_DIR = "/Users/yuxuanzeng/Desktop/iclr-temp/figs"

SYSTEMS = [
    ("water", "Water"),
    ("almg", "AlMg"),
    ("asi_011", "a-Si"),
]

METHODS = ["aiden", "charge3net", "sad"]
BLUES = ["#1f77b4", "#5fa8d8", "#a8cbe9"]
ORANGES = ["#ff7f0e", "#f9a25a", "#fccc9d"]


def find_one(pattern):
    hits = glob.glob(os.path.join(ROOT, pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def read_chgcar(path):
    with open(path) as f:
        lines = f.readlines()

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) * scale for x in lines[2 + i].split()[:3]] for i in range(3)])
    volume = abs(np.linalg.det(lattice))

    is_ints = lambda t: t and all(s.lstrip("-").isdigit() for s in t)

    cnt_line, pos_start = (5, 7) if is_ints(lines[5].split()) else (6, 8)
    n_at = sum(int(x) for x in lines[cnt_line].split())
    i = pos_start + n_at

    while not (len(lines[i].split()) == 3 and is_ints(lines[i].split())):
        i += 1

    ngx, ngy, ngz = map(int, lines[i].split())
    n = ngx * ngy * ngz

    vals = np.empty(n)
    k = 0

    with open(path) as f:
        for _ in range(i + 1):
            f.readline()

        for line in f:
            toks = line.split()

            if toks:
                vals[k:k + len(toks)] = np.array(toks, dtype=np.float64)
                k += len(toks)

            if k >= n:
                break

    rho = vals.reshape((ngx, ngy, ngz), order="F") / volume
    return lattice, volume, (ngx, ngy, ngz), rho


def parse_forces(outcar):
    txt = open(outcar).read()
    blk = txt.split("TOTAL-FORCE (eV/Angst)")[-1]
    F = []

    for ln in blk.splitlines()[2:]:
        t = ln.split()

        if len(t) == 6:
            F.append([float(x) for x in t[3:6]])
        else:
            break

    return np.array(F)


def parse_energy(outcar):
    e = None

    for line in open(outcar):
        m = re.search(r"free\s+energy\s+TOTEN\s*=\s*([-0-9.Ee+]+)", line)

        if m:
            e = float(m.group(1))

    return e


def style_axis(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.25)

    ax.minorticks_on()

    ax.tick_params(
        which="both",
        direction="in",
        width=1.25,
        labelsize=9.00,
        top=True,
        right=True,
        bottom=True,
        left=True,
    )


def main():
    bars = {}

    with open(os.path.join(ROOT, "core_metrics_compact.csv")) as f:
        header = f.readline().strip().split(",")

        for row in csv_rows(f, header):
            bars[(row["sample"], row["method"])] = row

    csv_check = {
        "density_nmae_percent": {},
        "force_component_mae_ev_per_ang": {},
        "energy_abs_error_mev_per_atom": {},
    }

    sel = json.load(open(os.path.join(ROOT, "analysis/atomic_slices/slice_selection.json")))
    plane_ix = {s["sample"]: s["grid_index"] for s in sel["selections"]}

    slice_stats = {}
    data = {}

    for key, label in SYSTEMS:
        ref_dir = os.path.dirname(
            os.path.dirname(
                find_one(f"downloads/reference/ref_{key}/*/extracted/results/scf/CHGCAR")
            )
        )

        aiden_npz = np.load(
            find_one(
                f"downloads/gpu/gpu_{key}/*/extracted/results/{key}/aiden/full_grid_prediction.npz"
            )
        )

        lattice, volume, grid, rho_ref = read_chgcar(os.path.join(ref_dir, "scf/CHGCAR"))
        _, _, grid_sad, rho_sad = read_chgcar(os.path.join(ref_dir, "sad/CHGCAR"))

        rho_aiden = aiden_npz["density_pred_original"].reshape(
            aiden_npz["grid_shape"], order="F"
        )

        rho_c3n = np.load(
            find_one(
                f"downloads/gpu/gpu_{key}/*/extracted/results/{key}/charge3net/full_grid_prediction.npy"
            )
        )

        assert (
            grid_sad == grid
            and tuple(aiden_npz["grid_shape"]) == grid
            and rho_c3n.shape == grid
        ), key

        for mname, arr in [
            ("aiden_merged", rho_aiden),
            ("charge3net", rho_c3n),
            ("sad", rho_sad),
        ]:
            nmae = np.abs(arr - rho_ref).sum() / np.abs(rho_ref).sum() * 100
            ref_val = float(bars[(key, mname)]["density_nmae_percent"])

            csv_check["density_nmae_percent"][(key, mname)] = (nmae, ref_val)

            assert abs(nmae - ref_val) / ref_val < 1e-3, (
                key,
                mname,
                nmae,
                ref_val,
            )

        nat = int(
            sum(
                json.load(
                    open(
                        find_one(
                            f"downloads/reference/ref_{key}/*/extracted/manifest.json"
                        )
                    )
                )["counts"]
            )
        )

        F_ref = parse_forces(os.path.join(ref_dir, "scf/OUTCAR"))
        E_ref = parse_energy(os.path.join(ref_dir, "scf/OUTCAR"))

        prop = {
            "aiden_merged": find_one(
                f"downloads/properties/prop_aiden_{key}/*/extracted/results/energy_force/OUTCAR"
            ),
            "charge3net": find_one(
                f"downloads/properties/prop_charge3net_{key}/*/extracted/results/energy_force/OUTCAR"
            ),
            "sad": os.path.join(ref_dir, "sad_energy_force/OUTCAR"),
        }

        for mname, p in prop.items():
            F = parse_forces(p)
            mae = np.abs(F - F_ref).mean()
            ref_val = float(bars[(key, mname)]["force_component_mae_ev_per_ang"])

            csv_check["force_component_mae_ev_per_ang"][(key, mname)] = (
                mae,
                ref_val,
            )

            e_mev = abs(parse_energy(p) - E_ref) * 1000.0 / nat
            ref_e = float(bars[(key, mname)]["energy_abs_error_mev_per_atom"])

            csv_check["energy_abs_error_mev_per_atom"][(key, mname)] = (
                e_mev,
                ref_e,
            )

        ix = plane_ix[key]

        ref2d = rho_ref[ix]
        pos = ref2d[ref2d > 0]

        rho_vmin = max(np.percentile(pos, 5.0), 1e-4)

        denom = np.maximum(
            np.abs(ref2d),
            np.percentile(pos, 10.0),
        )

        planes = {
            "ref": ref2d,
            "aiden_infer": rho_aiden[ix],
        }

        for mname, arr in [
            ("aiden_merged", rho_aiden),
            ("charge3net", rho_c3n),
            ("sad", rho_sad),
        ]:
            err = np.abs(arr[ix] - ref2d)
            planes[mname] = err / denom * 100.0

            slice_stats[(key, mname)] = {
                "slice_nmae_percent": err.sum() / np.abs(ref2d).sum() * 100,
                "mean_abs_error_e_per_ang3": err.mean(),
                "var_abs_error_e6": err.var(),
            }

        data[key] = {
            "label": label,
            "lattice": lattice,
            "planes": planes,
            "rho_vmin": rho_vmin,
        }

    print("=== verification: recomputed from raw data vs core_metrics_compact.csv ===")

    for metric, entries in csv_check.items():
        for (key, mname), (mine, ref) in entries.items():
            flag = "OK " if abs(mine - ref) / ref < 2e-3 else "?? "

            print(
                f"  {flag}{key:8s} {mname:12s} "
                f"{metric:34s} raw={mine:.6g}  csv={ref:.6g}"
            )

    print("\n=== slice-plane statistics (printed per error panel) ===")

    for (key, mname), st in slice_stats.items():
        print(
            f"  {key:8s} {mname:12s} "
            f"NMAE={st['slice_nmae_percent']:.4f}%  "
            f"mean|drho|={st['mean_abs_error_e_per_ang3']:.4e} e/A^3  "
            f"var={st['var_abs_error_e6']:.4e} (e/A^3)^2"
        )

    with open(os.path.join(os.getcwd(), "ood-summary-slice-stats.json"), "w") as f:
        json.dump(
            {f"{k[0]}/{k[1]}": v for k, v in slice_stats.items()},
            f,
            indent=2,
        )

    density_fig_w, density_fig_h = 8.2, 3.5
    fig = plt.figure(figsize=(density_fig_w, density_fig_h))

    density_gap = 0.055
    L, T, B = 0.55, 0.16, 0.24

    panel = (density_fig_h - T - B - 2 * density_gap) / 3
    block_w = 5 * panel + 4 * density_gap
    block_h = 3 * panel + 2 * density_gap

    cbar_w = 0.13
    cbar_x = L + block_w + density_gap
    cbar_h = (block_h - density_gap) / 2

    X2F = lambda x: x / density_fig_w
    Y2F = lambda y: y / density_fig_h

    def rect(x, y, w, h):
        return [
            X2F(x),
            Y2F(y),
            X2F(w),
            Y2F(h),
        ]

    rho_vmin = min(d["rho_vmin"] for d in data.values())
    rho_vmax = max(d["planes"]["ref"].max() for d in data.values())

    err_values = np.concatenate(
        [
            data[key]["planes"][m].ravel()
            for key, _ in SYSTEMS
            for m in (
                "aiden_merged",
                "charge3net",
                "sad",
            )
        ]
    )

    err_vmin = max(
        np.percentile(err_values, 10.0),
        np.finfo(float).tiny,
    )

    err_vmax = np.percentile(
        err_values,
        99.5,
    )

    rho_norm = LogNorm(
        vmin=rho_vmin,
        vmax=rho_vmax,
    )

    err_norm = LogNorm(
        vmin=err_vmin,
        vmax=err_vmax,
    )

    method_labels = [
        "DFT Ref.",
        "AIDEN Infer.",
        "AIDEN Err.",
        "ChargE3Net Err.",
        "SAD Err.",
    ]

    im_rho = None
    im_err = None

    for r, (key, _) in enumerate(SYSTEMS):
        d = data[key]
        lat = d["lattice"]

        ly = np.linalg.norm(lat[1])
        lz = np.linalg.norm(lat[2])

        ytop = B + block_h - r * (panel + density_gap)
        axs = []

        for c in range(5):
            ax = fig.add_axes(
                rect(
                    L + c * (panel + density_gap),
                    ytop - panel,
                    panel,
                    panel,
                )
            )

            axs.append(ax)

        ref2d = d["planes"]["ref"]

        im_rho = axs[0].imshow(
            np.clip(
                ref2d,
                d["rho_vmin"],
                None,
            ),
            origin="lower",
            extent=(0, lz, 0, ly),
            aspect="equal",
            cmap="GnBu",
            norm=rho_norm,
            interpolation="nearest",
        )

        im_rho = axs[1].imshow(
            np.clip(
                d["planes"]["aiden_infer"],
                d["rho_vmin"],
                None,
            ),
            origin="lower",
            extent=(0, lz, 0, ly),
            aspect="equal",
            cmap="GnBu",
            norm=rho_norm,
            interpolation="nearest",
        )

        dmaps = [
            d["planes"][m]
            for m in (
                "aiden_merged",
                "charge3net",
                "sad",
            )
        ]

        for c, dm in enumerate(
            dmaps,
            start=2,
        ):
            im_err = axs[c].imshow(
                np.clip(
                    dm,
                    err_vmin,
                    err_vmax,
                ),
                origin="lower",
                extent=(0, lz, 0, ly),
                aspect="equal",
                cmap="Spectral_r",
                norm=err_norm,
                interpolation="nearest",
            )

        for c, ax in enumerate(axs):
            style_axis(ax)

            ax.tick_params(
                labelbottom=False,
                labelleft=(c == 0),
            )

            if c > 0:
                ax.sharey(axs[0])

            if r == len(SYSTEMS) - 1:
                ax.set_xlabel(
                    method_labels[c],
                    labelpad=1,
                )

        axs[0].set_ylabel(
            f"{d['label']}\n"
            f"$y$ [$\\mathrm{{\\AA}}$]"
        )

    for i, (im, lab) in enumerate(
        [
            (
                im_rho,
                "$\\rho(\\mathbf{r})$ [e/$\\mathrm{\\AA}^3$]",
            ),
            (
                im_err,
                "$\\delta_{\\rho}$ [%]",
            ),
        ]
    ):
        cby = B + (1 - i) * (cbar_h + density_gap)

        cax = fig.add_axes(
            rect(
                cbar_x,
                cby,
                cbar_w,
                cbar_h,
            )
        )

        cb = fig.colorbar(
            im,
            cax=cax,
        )

        cb.outline.set_linewidth(1.25)

        cb.locator = LogLocator(
            base=10,
            subs=(1.0,),
        )

        cb.formatter = LogFormatterMathtext(
            base=10,
            labelOnlyBase=True,
        )

        cb.update_ticks()

        cb.ax.yaxis.set_minor_locator(
            LogLocator(
                base=10,
                subs=np.arange(2, 10),
            )
        )

        cb.ax.yaxis.set_ticks_position("both")

        cb.ax.tick_params(
            axis="y",
            which="major",
            direction="in",
            width=1.25,
            labelsize=10,
            left=True,
            right=True,
            labelleft=False,
            labelright=True,
        )

        cb.ax.tick_params(
            axis="y",
            which="minor",
            direction="in",
            width=1.0,
            left=True,
            right=True,
        )

        cb.set_label(lab)

    for ext, kw in (
        ("pdf", {}),
        ("jpg", {"dpi": 1200}),
    ):
        for out_dir in (
            os.getcwd(),
            EXTRA_FIG_DIR,
        ):
            out = os.path.join(
                out_dir,
                f"ood-density-summary.{ext}",
            )

            fig.savefig(
                out,
                bbox_inches="tight",
                **kw,
            )

            print(f"saved {out}")

    plt.close(fig)

    bar_fig_w, bar_fig_h = 7.2, 1.95
    fig = plt.figure(figsize=(bar_fig_w, bar_fig_h))

    L, R, B, bar_h = (
        0.48,
        0.24,
        0.27,
        1.06,
    )

    chart_gap = 0.90
    chart_w = (bar_fig_w - L - R - 2 * chart_gap) / 3

    X2F = lambda x: x / bar_fig_w
    Y2F = lambda y: y / bar_fig_h

    keys = [k for k, _ in SYSTEMS]

    w = 0.21

    # 第一张图每个体系的 6 根柱子严格连续排列
    dual_bar_w = 0.15
    dual_bar_step = 0.15

    centers = np.arange(3)
    offs = (np.arange(6) - 2.5) * dual_bar_step

    def style_bar_axis(ax):
        style_axis(ax)

        ax.xaxis.set_minor_locator(
            NullLocator()
        )

        ax.tick_params(
            axis="x",
            which="minor",
            top=False,
            bottom=False,
        )

        ax.tick_params(
            axis="x",
            which="major",
            top=True,
            bottom=True,
        )

    ax1 = fig.add_axes(
        rect(
            L,
            B,
            chart_w,
            bar_h,
        )
    )

    ax1r = ax1.twinx()

    for i, key in enumerate(keys):
        nmae = [
            float(
                bars[(key, m)][
                    "density_nmae_percent"
                ]
            )
            for m in (
                "aiden_merged",
                "charge3net",
                "sad",
            )
        ]

        tsec = [
            float(
                bars[(key, m)][
                    "density_time_seconds"
                ]
            )
            for m in (
                "aiden_merged",
                "charge3net",
                "sad",
            )
        ]

        ax1.bar(
            centers[i] + offs[:3],
            nmae,
            width=dual_bar_w,
            color=BLUES,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            zorder=2,
        )

        ax1r.bar(
            centers[i] + offs[3:],
            tsec,
            width=dual_bar_w,
            bottom=2,
            color=ORANGES,
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            zorder=2,
        )

    ax1.set_xlim(
        -0.75,
        2.75,
    )

    ax1.set_xticks(
        centers,
        [
            data[k]["label"]
            for k in keys
        ],
    )

    ax1.tick_params(
        axis="x",
        labelsize=9,
        pad=1,
    )

    ax1.set_ylabel(
        "NMAE $\\varepsilon_{\\rho}$ [%]"
    )

    ax1r.set_ylabel(
        "Infer. time [s]"
    )

    ax1r.set_yscale("log")

    ax1r.set_ylim(
        2,
        3000,
    )

    style_bar_axis(ax1)
    style_axis(ax1r)

    ax1.tick_params(
        axis="y",
        which="both",
        colors="#1f77b4",
        labelsize=9.00,
        right=False,
        labelright=False,
    )

    ax1r.tick_params(
        axis="y",
        which="both",
        colors="#ff7f0e",
        labelsize=9.00,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
    )

    ax1r.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
        top=False,
        labeltop=False,
    )

    ax1.spines["left"].set_color(
        "#1f77b4"
    )

    ax1r.spines["right"].set_color(
        "#ff7f0e"
    )

    ax1.yaxis.label.set_color(
        "#1f77b4"
    )

    ax1r.yaxis.label.set_color(
        "#ff7f0e"
    )

    ax1.spines["right"].set_visible(
        False
    )

    for side in (
        "left",
        "top",
        "bottom",
    ):
        ax1r.spines[side].set_visible(
            False
        )

    ax1r.spines["right"].set_visible(
        True
    )

    # ------------------------------------------------------------
    # 关键修改：
    # 在 twin axes 全部内容画完后，以 figure-level artist
    # 重新绘制第一张图的下 x 轴和主刻度。
    # 它们位于两个 Axes 的所有柱子之上，不会再被橙色柱遮挡。
    # ------------------------------------------------------------
    axis_overlay_zorder = 1000

    fig.add_artist(
        Line2D(
            [0.0, 1.0],
            [0.0, 0.0],
            transform=ax1.transAxes,
            color="black",
            linewidth=1.25,
            solid_capstyle="butt",
            clip_on=False,
            zorder=axis_overlay_zorder,
        )
    )

    tick_length_axes = (3.5 / 72.0) / bar_h

    x0, x1 = ax1.get_xlim()

    for x in centers:
        x_axes = (x - x0) / (x1 - x0)

        fig.add_artist(
            Line2D(
                [x_axes, x_axes],
                [0.0, tick_length_axes],
                transform=ax1.transAxes,
                color="black",
                linewidth=1.25,
                solid_capstyle="butt",
                clip_on=False,
                zorder=axis_overlay_zorder + 1,
            )
        )

    ax1.text(
        0.035,
        0.955,
        "a)",
        transform=ax1.transAxes,
        ha="left",
        va="top",
    )

    for j, (ylab, field) in enumerate(
        [
            (
                "$|\\Delta E|$ [meV/atom]",
                "energy_abs_error_mev_per_atom",
            ),
            (
                "$\\langle |\\Delta F_{i\\alpha}| \\rangle$ [eV/$\\mathrm{\\AA}$]",
                "force_component_mae_ev_per_ang",
            ),
        ]
    ):
        ax = fig.add_axes(
            rect(
                L + (j + 1) * (chart_w + chart_gap),
                B,
                chart_w,
                bar_h,
            )
        )

        for i, key in enumerate(keys):
            v = [
                float(
                    bars[(key, m)][field]
                )
                for m in (
                    "aiden_merged",
                    "charge3net",
                    "sad",
                )
            ]

            ax.bar(
                centers[i] + (np.arange(3) - 1) * w,
                v,
                width=w,
                color=BLUES,
                edgecolor="none",
            )

        ax.set_yscale("log")

        ax.set_xlim(
            -0.82,
            2.82,
        )

        ax.set_xticks(
            centers,
            [
                data[k]["label"]
                for k in keys
            ],
        )

        ax.tick_params(
            axis="x",
            labelsize=9,
            pad=1,
        )

        ax.set_ylabel(
            ylab
        )

        style_bar_axis(ax)

        ax.text(
            0.035,
            0.955,
            f"{'bc'[j]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

    legend_handles = [
        Patch(
            facecolor=BLUES[0],
            edgecolor="none",
            label="AIDEN",
        ),
        Patch(
            facecolor=BLUES[1],
            edgecolor="none",
            label="ChargE3Net",
        ),
        Patch(
            facecolor=BLUES[2],
            edgecolor="none",
            label="SAD",
        ),
        Patch(
            facecolor=ORANGES[0],
            edgecolor="none",
            label="AIDEN",
        ),
        Patch(
            facecolor=ORANGES[1],
            edgecolor="none",
            label="ChargE3Net",
        ),
        Patch(
            facecolor=ORANGES[2],
            edgecolor="none",
            label="SAD",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.8),
        ncol=6,
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        handleheight=0.8,
        handletextpad=0.4,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    for ext, kw in (
        ("pdf", {}),
        ("jpg", {"dpi": 1200}),
    ):
        for out_dir in (
            os.getcwd(),
            EXTRA_FIG_DIR,
        ):
            out = os.path.join(
                out_dir,
                f"ood-bar-summary.{ext}",
            )

            fig.savefig(
                out,
                bbox_inches="tight",
                **kw,
            )

            print(f"saved {out}")

    plt.close(fig)


def csv_rows(f, header):
    for line in f:
        parts = line.strip().split(",")

        if len(parts) == len(header):
            yield dict(
                zip(
                    header,
                    parts,
                )
            )


if __name__ == "__main__":
    main()
