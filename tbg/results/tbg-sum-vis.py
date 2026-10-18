import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext, AutoMinorLocator
from skimage.measure import marching_cubes
from ase.calculators.vasp import VaspChargeDensity
from ase.io import read as ase_read
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.linewidth'] = 1.15
mpl.rcParams['lines.linewidth'] = 1.15
mpl.rcParams['xtick.major.width'] = 1.15
mpl.rcParams['ytick.major.width'] = 1.15
mpl.rcParams['xtick.minor.width'] = 1.15
mpl.rcParams['ytick.minor.width'] = 1.15
mpl.rcParams['pdf.fonttype'] = 42

READ_CHARGE_DENSITY = True

MAIN_CBAR_VMIN = 1e-4
MAIN_CBAR_VMAX = 1e1

prediction_file = 'prediction.chgcar'
target_file = 'target.chgcar'

labels = [r'a) $\rho(\mathbf{r})$', r'b) $\hat{\rho}(\mathbf{r})$', r'c) $\varepsilon_{\rho}$']

rotation = '60x,30y,30z'
cmap_main = plt.get_cmap('viridis')
cmap_diff = plt.get_cmap('Spectral')

iso_alpha_main = 0.25
iso_alpha_diff = 0.30
LW = 1.2

step_main = 1
step_diff = 1

figsize = (8.75, 4.2)

cbar_width = 0.02
cbar_stack_frac = 0.90
cbar_gap_frac = 0.035
cbar_x_gap = 0.010

axis_inset_width = 0.075
axis_inset_height = 0.12
axis_inset_y_shift = 0.01

error_levels = np.array([0.5, 5.0, 10.0])

BASE_DIR = Path(__file__).resolve().parent
PROPERTY_CSV = BASE_DIR / 'property_metrics.csv'
TIMING_CSV = BASE_DIR / 'inference_timing.csv'
FORCES_CSV = BASE_DIR / 'forces_per_atom.csv'
EXTRA_FIG_DIR = None

CASE_ORDER = ('tbg_m01_21p79deg', 'tbg_m02_13p17deg', 'tbg_m03_9p43deg')
MODEL_ORDER = ('aiden', 'charge3net')

BAR_COLORS = ('#245297', '#537CB0', '#85C17E', '#A4D09D', '#C8E6C9')
LEFT_AXIS_COLOR = '#3C67A4'
RIGHT_AXIS_COLOR = '#95C98E'


def rotation_matrix(spec):
    R = np.eye(3)
    for item in spec.replace(' ', '').split(','):
        if item == '':
            continue
        ang = np.deg2rad(float(item[:-1]))
        ax = item[-1].lower()
        if ax == 'x':
            A = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ang), -np.sin(ang)], [0.0, np.sin(ang), np.cos(ang)]])
        elif ax == 'y':
            A = np.array([[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]])
        else:
            A = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
        R = A @ R
    return R


def project(points, R):
    q = np.asarray(points) @ R.T
    return q[:, :2], q[:, 2]


def read_chgcar(path):
    v = VaspChargeDensity(path)
    return v.atoms[-1], np.asarray(v.chg[-1], dtype=float)


def read_chgcar_geometry(path):
    return ase_read(path, format='vasp')


def get_cell_corners(cell):
    f = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    return f @ cell


def choose_log_levels(vmin, vmax, n=4):
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0 or vmax <= vmin:
        raise ValueError('Invalid positive range for logarithmic isosurfaces.')
    bounds = np.geomspace(vmin, vmax, n + 2)
    return bounds[1:-1]


def choose_main_levels(rho_target, rho_prediction):
    target_pos = rho_target[np.isfinite(rho_target) & (rho_target > 0.0)]
    pred_pos = rho_prediction[np.isfinite(rho_prediction) & (rho_prediction > 0.0)]
    common_min = max(target_pos.min(), pred_pos.min())
    common_max = min(target_pos.max(), pred_pos.max())
    if common_max <= common_min:
        raise ValueError('Target and prediction have no valid common positive density interval.')
    levels = choose_log_levels(common_min, common_max, 4)
    return common_min, common_max, levels


def calculate_relative_error(rho_target, rho_prediction):
    if rho_target.shape != rho_prediction.shape:
        raise ValueError(f'Target and prediction grids differ: {rho_target.shape} vs {rho_prediction.shape}')
    finite_target = np.abs(rho_target[np.isfinite(rho_target)])
    finite_target = finite_target[finite_target > 0.0]
    if finite_target.size == 0:
        raise ValueError('Target density contains no valid nonzero values.')
    rho_floor = max(np.finfo(float).eps, finite_target.max() * 1e-12)
    denominator = np.maximum(np.abs(rho_target), rho_floor)
    error = 100.0 * np.abs(rho_prediction - rho_target) / denominator
    error[~np.isfinite(error)] = 0.0
    return error, rho_floor


def draw_unit_cell(ax, atoms, R):
    corners = get_cell_corners(atoms.cell.array)
    xy, z = project(corners, R)
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    depths = [0.5 * (z[i] + z[j]) for i, j in edges]
    for k in np.argsort(depths):
        i, j = edges[k]
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color='0.15', lw=0.8, ls=(0, (6, 6)), solid_capstyle='butt', zorder=2)


def draw_atoms(ax, atoms, R, radius_scale):
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    xy, z = project(pos, R)
    for i in np.argsort(z):
        r = covalent_radii[nums[i]] * radius_scale
        circ = plt.Circle((xy[i, 0], xy[i, 1]), r, facecolor=jmol_colors[nums[i]], edgecolor='black', lw=0.6, zorder=20 + z[i])
        ax.add_patch(circ)


def extract_surface(rho, level, step_size):
    if not np.isfinite(level):
        return None
    if level <= np.nanmin(rho) or level >= np.nanmax(rho):
        return None
    try:
        return marching_cubes(rho, level=level, step_size=step_size, allow_degenerate=False)
    except RuntimeError:
        try:
            return marching_cubes(rho, level=level, step_size=1, allow_degenerate=False)
        except RuntimeError:
            return None


def draw_iso(ax, atoms, rho, levels, norm, cmap, R, alpha, step_size):
    shape = np.array(rho.shape, dtype=float) - 1.0
    polys_all = []
    colors_all = []
    depths_all = []

    for level in levels:
        result = extract_surface(rho, level, step_size)
        if result is None:
            print(f'No surface found for level = {level:.6g}')
            continue

        verts, faces, _, _ = result
        frac = verts / shape
        xyz = frac @ atoms.cell.array
        q = xyz @ R.T
        polys = q[faces][:, :, :2]
        depths = q[faces][:, :, 2].mean(axis=1)

        for poly, depth in zip(polys, depths):
            polys_all.append(poly)
            colors_all.append(cmap(norm(level)))
            depths_all.append(depth)

    if len(polys_all) == 0:
        return

    order = np.argsort(depths_all)
    polygons = [polys_all[i] for i in order]
    colors = [colors_all[i] for i in order]
    pc = PolyCollection(polygons, facecolors=colors, edgecolors='none', alpha=alpha, zorder=5)
    ax.add_collection(pc)


def add_xyz_inset(fig, ax_left, ax_right, R):
    pos0 = ax_left.get_position()
    pos1 = ax_right.get_position()
    x_center = 0.5 * (pos0.x1 + pos1.x0)
    y0 = min(pos0.y0, pos1.y0) + axis_inset_y_shift
    x0 = x_center - 0.5 * axis_inset_width
    ax_inset = fig.add_axes([x0, y0, axis_inset_width, axis_inset_height])
    ax_inset.set_aspect('equal')
    ax_inset.axis('off')

    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    xy, _ = project(basis, R)
    lens = np.linalg.norm(xy, axis=1)
    scale = 0.42 / np.max(lens)

    colors = ['red', 'green', 'blue']
    texts = [r'$x$', r'$y$', r'$z$']

    for vec, color, text in zip(xy, colors, texts):
        end = scale * vec
        ax_inset.annotate('', xy=(end[0], end[1]), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle='-|>', lw=LW, color=color, shrinkA=0.0, shrinkB=0.0, mutation_scale=12))
        ax_inset.text(1.12 * end[0], 1.12 * end[1], text, color='black', ha='center', va='center', fontsize=10)

    ax_inset.plot(0.0, 0.0, 'o', ms=3.0, color='0.4')
    ax_inset.set_xlim(-0.48, 0.48)
    ax_inset.set_ylim(-0.48, 0.48)


def require_columns(df, columns, name):
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f'{name} is missing columns: {", ".join(missing)}')


def load_property_data():
    df = pd.read_csv(PROPERTY_CSV)
    require_columns(df, {'case', 'method', 'smooth_charge_error_e', 'energy_fixed_density_ev', 'energy_error_mev_per_atom'}, PROPERTY_CSV.name)
    return df[df['case'].isin(CASE_ORDER) & df['method'].isin(MODEL_ORDER)].copy()


def load_timing_data():
    df = pd.read_csv(TIMING_CSV)
    require_columns(df, {'sample', 'aiden_full_grid_inference_seconds', 'charge3net_official_full_grid_per_sample_seconds', 'dft_scf_wall_seconds'}, TIMING_CSV.name)
    return df[df['sample'].isin(CASE_ORDER)].copy()


def load_force_data():
    df = pd.read_csv(FORCES_CSV)
    require_columns(df, {'case', 'method', 'fx_ev_per_angstrom', 'fy_ev_per_angstrom', 'fz_ev_per_angstrom', 'delta_fx_ev_per_angstrom', 'delta_fy_ev_per_angstrom', 'delta_fz_ev_per_angstrom'}, FORCES_CSV.name)
    return df[df['case'].isin(CASE_ORDER) & df['method'].isin(MODEL_ORDER)].copy()


def ordered_values(df, case_column, value_column, method=None):
    values = []
    for case in CASE_ORDER:
        part = df[df[case_column] == case]
        if method is not None:
            part = part[part['method'] == method]
        if len(part) != 1:
            raise ValueError(f'Expected one row for case={case}, method={method}, column={value_column}; got {len(part)}')
        values.append(float(part[value_column].iloc[0]))
    return np.asarray(values, dtype=float)


def force_component_means(df, method, delta):
    columns = ['delta_fx_ev_per_angstrom', 'delta_fy_ev_per_angstrom', 'delta_fz_ev_per_angstrom'] if delta else ['fx_ev_per_angstrom', 'fy_ev_per_angstrom', 'fz_ev_per_angstrom']
    values = []
    for case in CASE_ORDER:
        part = df[(df['case'] == case) & (df['method'] == method)]
        if part.empty:
            raise ValueError(f'No force rows for case={case}, method={method}')
        values.append(float(np.mean(np.abs(part[columns].to_numpy(dtype=float)))))
    return np.asarray(values, dtype=float)


def add_headroom(ax, values, fraction=0.22):
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

    ax.tick_params(axis='x', which='major', direction='in', top=True, bottom=True, width=1.15, labelsize=10, zorder=100)
    ax.tick_params(axis='x', which='minor', direction='in', top=True, bottom=True, width=1.15, zorder=100)
    ax.tick_params(axis='y', which='major', direction='in', left=True, right=False, width=1.15, colors=LEFT_AXIS_COLOR, labelsize=10, zorder=100)
    ax.tick_params(axis='y', which='minor', direction='in', left=True, right=False, width=1.15, colors=LEFT_AXIS_COLOR, zorder=100)
    axr.tick_params(axis='y', which='major', direction='in', left=False, right=True, width=1.15, colors=RIGHT_AXIS_COLOR, labelsize=10, zorder=100)
    axr.tick_params(axis='y', which='minor', direction='in', left=False, right=True, width=1.15, colors=RIGHT_AXIS_COLOR, zorder=100)

    ax.spines['left'].set_color(LEFT_AXIS_COLOR)
    ax.spines['left'].set_linewidth(1.15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(1.15)
    ax.spines['bottom'].set_linewidth(1.15)

    axr.spines['right'].set_visible(True)
    axr.spines['right'].set_color(RIGHT_AXIS_COLOR)
    axr.spines['right'].set_linewidth(1.15)
    axr.spines['left'].set_visible(False)
    axr.spines['top'].set_visible(False)
    axr.spines['bottom'].set_visible(False)

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

    ax.bar(x + offsets[0], left_1, width=width, color=BAR_COLORS[0], edgecolor='none', zorder=1)
    ax.bar(x + offsets[1], left_2, width=width, color=BAR_COLORS[1], edgecolor='none', zorder=1)
    axr.bar(x + offsets[2], right_1, width=width, color=BAR_COLORS[2], edgecolor='none', zorder=1)
    axr.bar(x + offsets[3], right_2, width=width, color=BAR_COLORS[3], edgecolor='none', zorder=1)
    if right_3 is not None:
        axr.bar(x + offsets[4], right_3, width=width, color=BAR_COLORS[4], edgecolor='none', zorder=1)

    ax.set_ylabel(left_ylabel, labelpad=1.5, fontsize=10)
    axr.set_ylabel(right_ylabel, labelpad=1.5, fontsize=10)

    style_axes(ax, axr, x, xticklabels, right_log=right_log)
    add_headroom(ax, np.concatenate((left_1, left_2)))
    right_values = (right_1, right_2) if right_3 is None else (right_1, right_2, right_3)
    if right_log:
        positive = np.asarray(right_values, dtype=float)
        positive = positive[positive > 0]
        axr.set_yscale('log')
        axr.set_ylim(float(np.min(positive)) / 2.5, float(np.max(positive)) * 1.5)
    else:
        add_headroom(axr, np.concatenate(right_values))

    ax.text(0.035, 0.965, f'{panel_label})', transform=ax.transAxes, ha='left', va='top', fontsize=10, zorder=900)


def draw_bar_row(fig, row_spec, property_data, timing_data, force_data):
    x = np.arange(len(CASE_ORDER))
    angles = np.asarray([21.79, 13.17, 9.43])
    xticklabels = [rf'{angle:.2f}$^\circ$' for angle in angles]

    charge_aiden = np.abs(ordered_values(property_data, 'case', 'smooth_charge_error_e', 'aiden'))
    charge_charge3net = np.abs(ordered_values(property_data, 'case', 'smooth_charge_error_e', 'charge3net'))
    time_aiden = ordered_values(timing_data, 'sample', 'aiden_full_grid_inference_seconds')
    time_charge3net = ordered_values(timing_data, 'sample', 'charge3net_official_full_grid_per_sample_seconds')
    time_dft = ordered_values(timing_data, 'sample', 'dft_scf_wall_seconds')

    energy_aiden = np.abs(ordered_values(property_data, 'case', 'energy_fixed_density_ev', 'aiden'))
    energy_charge3net = np.abs(ordered_values(property_data, 'case', 'energy_fixed_density_ev', 'charge3net'))
    energy_err_aiden = np.abs(ordered_values(property_data, 'case', 'energy_error_mev_per_atom', 'aiden'))
    energy_err_charge3net = np.abs(ordered_values(property_data, 'case', 'energy_error_mev_per_atom', 'charge3net'))

    force_aiden = force_component_means(force_data, 'aiden', False)
    force_charge3net = force_component_means(force_data, 'charge3net', False)
    force_err_aiden = force_component_means(force_data, 'aiden', True)
    force_err_charge3net = force_component_means(force_data, 'charge3net', True)

    gs_bottom = row_spec.subgridspec(1, 3, wspace=0.44)
    axes = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]

    plot_four_bars(axes[0], x, charge_aiden, charge_charge3net, time_aiden, time_charge3net, r'$|\Delta N_{\rm e}|$ [Unitless]', 'Inference time [s]', xticklabels, 'd', right_3=time_dft, right_log=True)
    plot_four_bars(axes[1], x, energy_aiden, energy_charge3net, energy_err_aiden, energy_err_charge3net, r'$|E|$ [eV]', r'$|\Delta E|$ [meV/atom]', xticklabels, 'e')
    plot_four_bars(axes[2], x, force_aiden, force_charge3net, force_err_aiden, force_err_charge3net, r'$\langle |F_{i\alpha}| \rangle_{i,\alpha}$ [eV/$\mathrm{\AA}$]', r'$\langle |\Delta F_{i\alpha}| \rangle$ [eV/$\mathrm{\AA}$]', xticklabels, 'f')

    legend_handles = [
        Patch(facecolor=BAR_COLORS[0], edgecolor="none", label="AIDEN"),
        Patch(facecolor=BAR_COLORS[1], edgecolor="none", label="ChargE3Net"),
        Patch(facecolor=BAR_COLORS[2], edgecolor="none", label="AIDEN"),
        Patch(facecolor=BAR_COLORS[3], edgecolor="none", label="ChargE3Net"),
        Patch(facecolor=BAR_COLORS[4], edgecolor="none", label="DFT")
    ]

    return axes, legend_handles


def add_aligned_top_labels(fig, top_axes, bottom_axes, top_labels):
    fig.canvas.draw()
    for ax_top, ax_bottom, label in zip(top_axes, bottom_axes, top_labels):
        x_display = ax_bottom.transAxes.transform((0.035, 0.965))[0]
        y_display = ax_top.transAxes.transform((0.02, 0.98))[1]
        x_figure, y_figure = fig.transFigure.inverted().transform((x_display, y_display))
        fig.text(x_figure, y_figure, label, ha='left', va='top', fontsize=10, fontweight='normal', zorder=900)


def process_sample(sample_dir, output_dir, property_data, timing_data, force_data):
    pred_path = os.path.join(sample_dir, prediction_file)
    target_path = os.path.join(sample_dir, target_file)

    R = rotation_matrix(rotation)

    if READ_CHARGE_DENSITY:
        atoms_pred, rho_pred = read_chgcar(pred_path)
        atoms_target, rho_target = read_chgcar(target_path)

        main_min, main_max, main_levels = choose_main_levels(rho_target, rho_pred)
        rho_error, rho_floor = calculate_relative_error(rho_target, rho_pred)

        print(f'Processing: {sample_dir}')
        print('Shared target/prediction range:')
        print(f'Min: {main_min:.8e}')
        print(f'Max: {main_max:.8e}')
        print('Shared target/prediction isosurfaces:')
        for i, level in enumerate(main_levels, 1):
            print(f'Level {i}: {level:.8e}')
        print('Relative-error field:')
        print(f'Min: {rho_error.min():.8e} %')
        print(f'Max: {rho_error.max():.8e} %')
        print(f'Density denominator floor: {rho_floor:.8e}')
        print('Relative-error isosurfaces:')
        for level in error_levels:
            print(f'{level:.1f} %')
    else:
        atoms_pred = read_chgcar_geometry(pred_path)
        atoms_target = read_chgcar_geometry(target_path)

        rho_pred = None
        rho_target = None
        rho_error = None
        main_levels = None

        main_min = MAIN_CBAR_VMIN
        main_max = MAIN_CBAR_VMAX

        print(f'Processing: {sample_dir}')
        print('READ_CHARGE_DENSITY = False')
        print('Charge-density grids are not loaded.')

    main_norm = LogNorm(vmin=main_min, vmax=main_max)
    error_norm = LogNorm(vmin=error_levels.min(), vmax=error_levels.max())

    corners_target = get_cell_corners(atoms_target.cell.array)
    corners_pred = get_cell_corners(atoms_pred.cell.array)

    xy_target, _ = project(corners_target, R)
    xy_pred, _ = project(corners_pred, R)
    all_xy = np.vstack([xy_target, xy_pred])

    xmin = all_xy[:, 0].min()
    xmax = all_xy[:, 0].max()
    ymin = all_xy[:, 1].min()
    ymax = all_xy[:, 1].max()

    span = max(xmax - xmin, ymax - ymin)
    pad = 0.10 * span
    radius_scale = 0.060 * span

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(2, 1, left=0.015, right=0.90, bottom=0.070, top=0.985, height_ratios=(3.0, 1.55), hspace=0.10)

    gs_top = outer[0].subgridspec(1, 3, wspace=0.002)
    axes = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.axis('off')

    if READ_CHARGE_DENSITY:
        draw_iso(axes[0], atoms_target, rho_target, main_levels, main_norm, cmap_main, R, iso_alpha_main, step_main)

    draw_unit_cell(axes[0], atoms_target, R)
    draw_atoms(axes[0], atoms_target, R, radius_scale)

    if READ_CHARGE_DENSITY:
        draw_iso(axes[1], atoms_pred, rho_pred, main_levels, main_norm, cmap_main, R, iso_alpha_main, step_main)

    draw_unit_cell(axes[1], atoms_pred, R)
    draw_atoms(axes[1], atoms_pred, R, radius_scale)

    if READ_CHARGE_DENSITY:
        draw_iso(axes[2], atoms_target, rho_error, error_levels, error_norm, cmap_diff, R, iso_alpha_diff, step_diff)

    draw_unit_cell(axes[2], atoms_target, R)
    draw_atoms(axes[2], atoms_target, R, radius_scale)

    add_xyz_inset(fig, axes[0], axes[1], R)

    pos3 = axes[2].get_position()
    stack_height = pos3.height * cbar_stack_frac
    gap_height = pos3.height * cbar_gap_frac
    single_height = 0.5 * (stack_height - gap_height)
    stack_y0 = pos3.y0 + 0.5 * (pos3.height - stack_height)
    cbar_x0 = pos3.x1 + cbar_x_gap

    cax_main = fig.add_axes([cbar_x0, stack_y0 + single_height + gap_height, cbar_width, single_height])
    cax_diff = fig.add_axes([cbar_x0, stack_y0, cbar_width, single_height])

    sm_main = mpl.cm.ScalarMappable(norm=main_norm, cmap=cmap_main)
    sm_main.set_array([])
    cb_main = fig.colorbar(sm_main, cax=cax_main)
    cb_main.locator = LogLocator(base=10.0)
    cb_main.formatter = LogFormatterMathtext(base=10.0)
    cb_main.update_ticks()
    cb_main.set_label(r'$\rho(\mathbf{r})$, $\hat{\rho}(\mathbf{r})$ [${\rm e}/{\rm \AA}^3$]', fontsize=10)
    cb_main.ax.yaxis.set_ticks_position('both')
    cb_main.ax.tick_params(which='major', direction='in', left=True, right=True, labelleft=False, labelright=True, length=4, width=LW, labelsize=10)
    cb_main.ax.tick_params(which='minor', direction='in', left=True, right=True, labelleft=False, labelright=False, length=2.5, width=LW)
    cb_main.outline.set_linewidth(LW)

    sm_diff = mpl.cm.ScalarMappable(norm=error_norm, cmap=cmap_diff)
    sm_diff.set_array([])
    cb_diff = fig.colorbar(sm_diff, cax=cax_diff)
    cb_diff.locator = LogLocator(base=10.0)
    cb_diff.formatter = LogFormatterMathtext(base=10.0)
    cb_diff.update_ticks()
    cb_diff.set_label(r'$\varepsilon_{\rho}$ [%]', fontsize=10)
    cb_diff.ax.yaxis.set_ticks_position('both')
    cb_diff.ax.tick_params(which='major', direction='in', left=True, right=True, labelleft=False, labelright=True, length=4, width=LW, labelsize=10)
    cb_diff.ax.tick_params(which='minor', direction='in', left=True, right=True, labelleft=False, labelright=False, length=2.5, width=LW)
    cb_diff.outline.set_linewidth(LW)

    bar_axes, legend_handles = draw_bar_row(fig, outer[1], property_data, timing_data, force_data)
    add_aligned_top_labels(fig, axes, bar_axes, labels)

    legend = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.4575, -0.05), ncol=5, frameon=False, handlelength=1.25, handleheight=0.8, handletextpad=0.45, columnspacing=1.25, borderaxespad=0.0, fontsize=10)
    legend.set_zorder(1000)

    sample_name = os.path.basename(sample_dir.rstrip(os.sep))
    jpg_path = os.path.join(output_dir, f'chg-vis-sum.jpg')
    pdf_path = os.path.join(output_dir, f'chg-vis-sum.pdf')

    fig.savefig(jpg_path, dpi=1200, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    EXTRA_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(EXTRA_FIG_DIR, 'chg-vis-sum.jpg'), dpi=1200, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(EXTRA_FIG_DIR, 'chg-vis-sum.pdf'), bbox_inches='tight', pad_inches=0.02)

    plt.show()

def main():
    base_dir = '.'
    output_dir = 'view-res'
    os.makedirs(output_dir, exist_ok=True)

    property_data = load_property_data()
    timing_data = load_timing_data()
    force_data = load_force_data()

    for entry in os.listdir(base_dir):
        sample_path = os.path.join(base_dir, entry)

        if not os.path.isdir(sample_path):
            continue

        if not os.path.isfile(os.path.join(sample_path, prediction_file)):
            continue

        if not os.path.isfile(os.path.join(sample_path, target_file)):
            continue

        process_sample(sample_path, output_dir, property_data, timing_data, force_data)


if __name__ == '__main__':
    main()