from __future__ import annotations

SHOW_WORLD_AXES = False

import argparse
import itertools
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results-gaas" / ".matplotlib"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogFormatterMathtext, LogLocator  # noqa: E402

from ase.calculators.vasp import VaspChargeDensity  # noqa: E402
from ase.data import covalent_radii  # noqa: E402
from ase.data.colors import jmol_colors  # noqa: E402


# Visual settings parallel to the first row of tbg-sum-vis.py.
CAMERA_ROTATION = "60x,30y,30z"
CMAP = plt.get_cmap("viridis")
EPS_CMAP = plt.get_cmap("Spectral")
ISO_ALPHA = 0.27
LINE_WIDTH = 1.15
NUM_ISOSURFACES = 4
# Error fields reach machine-zero at near-perfect points; isosurface levels
# only span this many decades below each field's own maximum.
EPS_DYNAMIC_WINDOW = 1.0e4
FIGSIZE = (7.3, 2.7)
PANEL_NAMES = ("original", "translation", "proper_rotation", "inversion")
PANEL_TITLES = (
    "a) Original",
    "b) Translation",
    "c) Rotation",
    "d) Inversion",
)
CELL_EDGES = (
    (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
    (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
)
CELL_CORNER_FRAC = np.asarray(
    (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ),
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density-dir",
        type=Path,
        default=ROOT / "results-gaas" / "densities",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results-gaas" / "eq-gaas.png",
    )
    parser.add_argument(
        "--extra-figs-dir",
        type=Path,
        default=None,
        help="additionally copy the saved PNG/PDF into this directory",
    )
    parser.add_argument(
        "--world-axes",
        action="store_true",
        default=SHOW_WORLD_AXES,
        help="draw a shared large world Cartesian frame behind every panel",
    )
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def rotation_matrix(specification: str) -> np.ndarray:
    rotation = np.eye(3)
    for item in specification.replace(" ", "").split(","):
        angle = np.deg2rad(float(item[:-1]))
        axis = item[-1].lower()
        if axis == "x":
            update = np.asarray(
                ((1, 0, 0), (0, np.cos(angle), -np.sin(angle)), (0, np.sin(angle), np.cos(angle)))
            )
        elif axis == "y":
            update = np.asarray(
                ((np.cos(angle), 0, np.sin(angle)), (0, 1, 0), (-np.sin(angle), 0, np.cos(angle)))
            )
        elif axis == "z":
            update = np.asarray(
                ((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1))
            )
        else:
            raise ValueError(f"unknown camera axis {axis!r}")
        rotation = update @ rotation
    return rotation


def project(points: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.asarray(points) @ camera.T
    return transformed[..., :2], transformed[..., 2]


def read_density(path: Path):
    payload = VaspChargeDensity(str(path))
    if not payload.atoms or not payload.chg:
        raise ValueError(f"ASE found no density in {path}")
    atoms = payload.atoms[-1]
    density = np.asarray(payload.chg[-1], dtype=float)
    if density.ndim != 3 or not np.isfinite(density).all():
        raise ValueError(f"invalid density grid in {path}: {density.shape}")
    return atoms, density


def common_isosurface_levels(densities: list[np.ndarray]) -> tuple[float, float, np.ndarray]:
    positive = [field[np.isfinite(field) & (field > 0.0)] for field in densities]
    if any(values.size == 0 for values in positive):
        raise ValueError("all density grids must contain positive values")
    lower = max(float(values.min()) for values in positive)
    upper = min(float(values.max()) for values in positive)
    if upper <= lower:
        raise ValueError("density grids have no shared positive interval")
    bounds = np.geomspace(lower, upper, NUM_ISOSURFACES + 2)
    return lower, upper, bounds[1:-1]


def align_density_to_reference(
    field: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, tuple | None]:
    """Undo an exact periodic grid roll before a pointwise comparison.

    The translated CHGCAR stores the reference density rolled on the periodic
    grid; the rotation and inversion CHGCARs already share the reference's
    fractional indices.  A correlation-optimal integer roll is applied only
    when it halves the direct L1 difference, so aligned fields stay untouched.
    """
    direct = float(np.abs(field - reference).sum())
    spectrum = np.fft.rfftn(reference) * np.conj(np.fft.rfftn(field))
    correlation = np.fft.irfftn(spectrum, s=field.shape, axes=(0, 1, 2))
    lag = np.unravel_index(int(np.argmax(correlation)), field.shape)
    shift = tuple(
        int(index) if index <= size // 2 else int(index) - int(size)
        for index, size in zip(lag, field.shape)
    )
    rolled = np.roll(field, shift, axis=(0, 1, 2))
    if float(np.abs(rolled - reference).sum()) < 0.5 * direct:
        return rolled, shift
    return field, None


def error_isosurface_levels(field: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-spaced isosurface levels for one error field, plus its norm floor.

    The floor clamps each panel's level range to ``EPS_DYNAMIC_WINDOW`` decades
    below the field maximum so machine-zero points never stretch the scale.
    """
    positive = field[field > 0.0]
    if positive.size == 0:
        raise ValueError("error field has no positive values")
    upper = float(positive.max())
    lower = max(float(positive.min()), upper / EPS_DYNAMIC_WINDOW)
    return np.geomspace(lower, upper, NUM_ISOSURFACES + 2)[1:-1], lower


def _interpolate_edge(
    point_a: np.ndarray,
    point_b: np.ndarray,
    value_a: np.ndarray,
    value_b: np.ndarray,
    level: float,
) -> np.ndarray:
    denominator = value_b - value_a
    denominator = np.where(np.abs(denominator) > 1.0e-30, denominator, 1.0e-30)
    fraction = np.clip((level - value_a) / denominator, 0.0, 1.0)
    return point_a + fraction[:, None] * (point_b - point_a)


def marching_tetrahedra(field: np.ndarray, level: float) -> np.ndarray:
    """Dependency-free isosurface triangles in grid-index coordinates."""
    shape = np.asarray(field.shape, dtype=int)
    if np.any(shape < 2) or not float(np.min(field)) < level < float(np.max(field)):
        return np.empty((0, 3, 3), dtype=float)

    offsets = CELL_CORNER_FRAC
    corner_values = np.stack(
        [
            field[
                offset[0] : offset[0] + shape[0] - 1,
                offset[1] : offset[1] + shape[1] - 1,
                offset[2] : offset[2] + shape[2] - 1,
            ]
            for offset in offsets.astype(int)
        ],
        axis=-1,
    ).reshape(-1, 8)
    origins = np.stack(
        np.meshgrid(
            np.arange(shape[0] - 1),
            np.arange(shape[1] - 1),
            np.arange(shape[2] - 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3).astype(float)

    # Six tetrahedra sharing the cube's 0--7 body diagonal.
    tetrahedra = (
        (0, 1, 4, 7), (0, 4, 2, 7), (0, 2, 6, 7),
        (0, 6, 3, 7), (0, 3, 5, 7), (0, 5, 1, 7),
    )
    triangles: list[np.ndarray] = []
    for tetrahedron in tetrahedra:
        tetrahedron = np.asarray(tetrahedron, dtype=int)
        values = corner_values[:, tetrahedron]
        inside = values >= level
        counts = inside.sum(axis=1)
        points = origins[:, None, :] + offsets[tetrahedron][None, :, :]

        # One vertex on one side of the surface gives one triangle.  Handling
        # both 1-in and 1-out cases covers counts 1 and 3.
        for singleton_inside in (True, False):
            target_count = 1 if singleton_inside else 3
            singleton_mask = inside if singleton_inside else ~inside
            for singleton in range(4):
                selected = (counts == target_count) & singleton_mask[:, singleton]
                if not np.any(selected):
                    continue
                others = [index for index in range(4) if index != singleton]
                triangle = np.stack(
                    [
                        _interpolate_edge(
                            points[selected, singleton],
                            points[selected, other],
                            values[selected, singleton],
                            values[selected, other],
                            level,
                        )
                        for other in others
                    ],
                    axis=1,
                )
                triangles.append(triangle)

        # Two-in/two-out gives a quadrilateral, split consistently into two triangles.
        for inside_pair in itertools.combinations(range(4), 2):
            outside_pair = tuple(index for index in range(4) if index not in inside_pair)
            selected = (counts == 2) & inside[:, inside_pair[0]] & inside[:, inside_pair[1]]
            if not np.any(selected):
                continue
            p00 = _interpolate_edge(
                points[selected, inside_pair[0]], points[selected, outside_pair[0]],
                values[selected, inside_pair[0]], values[selected, outside_pair[0]],
                level,
            )
            p01 = _interpolate_edge(
                points[selected, inside_pair[0]], points[selected, outside_pair[1]],
                values[selected, inside_pair[0]], values[selected, outside_pair[1]],
                level,
            )
            p10 = _interpolate_edge(
                points[selected, inside_pair[1]], points[selected, outside_pair[0]],
                values[selected, inside_pair[1]], values[selected, outside_pair[0]],
                level,
            )
            p11 = _interpolate_edge(
                points[selected, inside_pair[1]], points[selected, outside_pair[1]],
                values[selected, inside_pair[1]], values[selected, outside_pair[1]],
                level,
            )
            triangles.append(np.stack((p00, p01, p11), axis=1))
            triangles.append(np.stack((p00, p11, p10), axis=1))

    return np.concatenate(triangles, axis=0) if triangles else np.empty((0, 3, 3))


def centered_cell_corners(atoms) -> np.ndarray:
    cell = np.asarray(atoms.cell.array, dtype=float)
    return CELL_CORNER_FRAC @ cell - 0.5 * cell.sum(axis=0)


def draw_isosurfaces(
    axis,
    atoms,
    density: np.ndarray,
    levels: np.ndarray,
    normalization: LogNorm,
    camera: np.ndarray,
    cmap=None,
    alpha=None,
    rasterized: bool = True,
) -> None:
    if cmap is None:
        cmap = CMAP
    if alpha is None:
        alpha = ISO_ALPHA
    cell = np.asarray(atoms.cell.array, dtype=float)
    center = 0.5 * cell.sum(axis=0)
    denominator = np.asarray(density.shape, dtype=float) - 1.0
    projected_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for level in levels:
        triangles = marching_tetrahedra(density, float(level))
        if not triangles.size:
            continue
        fractional = triangles / denominator[None, None, :]
        cartesian = fractional @ cell - center
        projected, depth = project(cartesian, camera)
        projected_parts.append(projected)
        depth_parts.append(depth.mean(axis=1))
        color_parts.append(
            np.repeat(np.asarray([cmap(normalization(level))]), projected.shape[0], axis=0)
        )
    if not projected_parts:
        return
    polygons = np.concatenate(projected_parts, axis=0)
    depths = np.concatenate(depth_parts)
    colors = np.concatenate(color_parts, axis=0)
    order = np.argsort(depths)
    collection = PolyCollection(
        polygons[order],
        facecolors=colors[order],
        edgecolors="none",
        alpha=alpha,
        zorder=5,
        rasterized=rasterized,
    )
    axis.add_collection(collection)


def draw_unit_cell(axis, atoms, camera: np.ndarray) -> None:
    projected, depth = project(centered_cell_corners(atoms), camera)
    edge_depth = [0.5 * (depth[left] + depth[right]) for left, right in CELL_EDGES]
    for edge_index in np.argsort(edge_depth):
        left, right = CELL_EDGES[edge_index]
        axis.plot(
            projected[[left, right], 0],
            projected[[left, right], 1],
            color="0.15",
            linewidth=0.85,
            linestyle=(0, (6, 5)),
            solid_capstyle="butt",
            zorder=10,
        )


def draw_atoms(axis, atoms, camera: np.ndarray, radius_scale: float) -> None:
    cell = np.asarray(atoms.cell.array, dtype=float)
    centered = np.asarray(atoms.positions) - 0.5 * cell.sum(axis=0)
    projected, depth = project(centered, camera)
    for atom_index in np.argsort(depth):
        atomic_number = int(atoms.numbers[atom_index])
        circle = plt.Circle(
            projected[atom_index],
            covalent_radii[atomic_number] * radius_scale,
            facecolor=jmol_colors[atomic_number],
            edgecolor="black",
            linewidth=0.65,
            zorder=20 + depth[atom_index],
        )
        axis.add_patch(circle)


def draw_world_axes(axis, camera: np.ndarray, length: float) -> None:
    """Large identical lab-frame axes, deliberately behind the crystal."""
    basis = np.eye(3)
    projected, _ = project(basis, camera)
    projected *= length / np.linalg.norm(projected, axis=1)[:, None]
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    labels = (r"$X$", r"$Y$", r"$Z$")
    for vector, color, label in zip(projected, colors, labels):
        axis.annotate(
            "",
            xy=vector,
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>", linewidth=2.0, color=color, alpha=0.28,
                mutation_scale=14, shrinkA=0.0, shrinkB=0.0,
            ),
            zorder=0,
        )
        axis.text(
            1.06 * vector[0], 1.06 * vector[1], label,
            color=color, alpha=0.60, fontsize=11, ha="center", va="center", zorder=15,
        )


def add_crystal_axes(axis, atoms, camera: np.ndarray) -> None:
    """Small lower-left xyz triad following the panel's crystal directions."""
    inset = axis.inset_axes((0.015, 0.015, 0.22, 0.22), zorder=100)
    inset.set_aspect("equal")
    inset.axis("off")
    lattice = np.asarray(atoms.cell.array, dtype=float)
    crystal_basis = lattice / np.linalg.norm(lattice, axis=1)[:, None]
    projected, _ = project(crystal_basis, camera)
    maximum = max(float(np.linalg.norm(projected, axis=1).max()), 1.0e-12)
    projected *= 0.42 / maximum
    for vector, color, label in zip(
        projected,
        ("red", "green", "blue"),
        (r"$x$", r"$y$", r"$z$"),
    ):
        inset.annotate(
            "",
            xy=vector,
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>", linewidth=LINE_WIDTH, color=color,
                mutation_scale=10, shrinkA=0.0, shrinkB=0.0,
            ),
        )
        inset.text(
            1.15 * vector[0], 1.15 * vector[1], label,
            color="black", ha="center", va="center", fontsize=8.5,
        )
    inset.plot(0.0, 0.0, "o", markersize=2.4, color="0.4")
    inset.set_xlim(-0.52, 0.52)
    inset.set_ylim(-0.52, 0.52)


def add_atom_legend(figure, atoms) -> None:
    unique_numbers = list(dict.fromkeys(int(number) for number in atoms.numbers))
    handles = [
        Line2D(
            (0,), (0,), marker="o", linestyle="none", markersize=7,
            markerfacecolor=jmol_colors[number], markeredgecolor="black",
            markeredgewidth=0.6, label=atoms[number_index].symbol,
        )
        for number_index, number in (
            (next(index for index, value in enumerate(atoms.numbers) if int(value) == number), number)
            for number in unique_numbers
        )
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.002),
        ncol=len(handles),
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.1,
        fontsize=9,
    )


class DecadeMathtextFormatter(LogFormatterMathtext):
    """Log mathtext labels on only every *step*-th decade (e.g. 10^-8, 10^-6)."""

    def __init__(self, *, base: float = 10.0, step: int = 2, **kwargs) -> None:
        super().__init__(base=base, **kwargs)
        self.step = int(step)

    def __call__(self, value, pos=None):
        if value > 0.0:
            exponent = round(float(np.log10(value)))
            if exponent % self.step == 0 and abs(10.0**exponent - value) <= 1.0e-6 * value:
                return super().__call__(value, pos)
        return ""


def add_log_colorbar(figure, rect, cmap, normalization: LogNorm, label: str,
                     label_every_nth_decade: int = 1):
    axis = figure.add_axes(rect)
    scalar = mpl.cm.ScalarMappable(norm=normalization, cmap=cmap)
    scalar.set_array([])
    colorbar = figure.colorbar(scalar, cax=axis)
    colorbar.ax.yaxis.set_ticks_position("both")
    colorbar.ax.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,), numticks=100)
    )
    colorbar.ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2.0, 10.0), numticks=100)
    )
    colorbar.ax.tick_params(
        which="major", direction="in", left=True, right=True,
        labelleft=False, labelright=True, width=1.2, length=4,
    )
    colorbar.ax.tick_params(
        which="minor", direction="in", left=True, right=True,
        width=1.2, length=2.5,
    )
    formatter = LogFormatterMathtext(base=10.0)
    if label_every_nth_decade > 1:
        formatter = DecadeMathtextFormatter(base=10.0, step=label_every_nth_decade)
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.outline.set_linewidth(1.2)
    colorbar.set_label(label, fontsize=10)
    return colorbar


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    density_dir = args.density_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output.parent / ".matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    atoms_all, densities = [], []
    for name in PANEL_NAMES:
        atoms, density = read_density(density_dir / f"{name}.CHGCAR")
        atoms_all.append(atoms)
        densities.append(density)
    shapes = {field.shape for field in densities}
    if len(shapes) != 1:
        raise ValueError(f"density grid shapes differ: {sorted(shapes)}")

    reference = densities[0]
    reference_floor = float(np.abs(reference).min())
    if reference_floor <= 0.0:
        raise ValueError("reference density vanishes on the grid; relative error undefined")
    error_fields: list[np.ndarray] = []
    grid_rolls: dict[str, tuple | None] = {}
    for name, field in zip(PANEL_NAMES[1:], densities[1:]):
        aligned, shift = align_density_to_reference(field, reference)
        error_fields.append(100.0 * np.abs(aligned - reference) / np.abs(reference))
        grid_rolls[name] = shift

    rho_minimum, rho_maximum, rho_levels = common_isosurface_levels([reference])
    eps_levels_and_floors = [error_isosurface_levels(field) for field in error_fields]
    panel_levels = [rho_levels] + [levels for levels, _ in eps_levels_and_floors]
    eps_minimum = min(floor for _, floor in eps_levels_and_floors)
    eps_maximum = max(float(field.max()) for field in error_fields)
    rho_norm = LogNorm(vmin=rho_minimum, vmax=rho_maximum)
    eps_norm = LogNorm(vmin=eps_minimum, vmax=eps_maximum)
    camera = rotation_matrix(CAMERA_ROTATION)

    all_projected = np.concatenate(
        [project(centered_cell_corners(atoms), camera)[0] for atoms in atoms_all], axis=0
    )
    x_min, y_min = all_projected.min(axis=0)
    x_max, y_max = all_projected.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min))
    pad = 0.11 * span
    radius_scale = 0.055 * span

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.15,
            "lines.linewidth": 1.15,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.minor.width": 1.15,
            "ytick.minor.width": 1.15,
            "pdf.fonttype": 42,
        }
    )
    try:
        mpl.font_manager.findfont("Times New Roman", fallback_to_default=False)
    except Exception as error:
        raise RuntimeError(
            "Times New Roman is missing from the matplotlib font cache; "
            f"delete {os.environ.get('MPLCONFIGDIR')}/fontlist-*.json and rerun "
            "to rebuild the cache (matplotlib otherwise silently falls back "
            "to DejaVu Sans)"
        ) from error
    figure, axes = plt.subplots(1, 4, figsize=FIGSIZE)
    figure.subplots_adjust(left=0.008, right=0.925, bottom=0.11, top=0.91, wspace=0.01)
    display = [
        (reference, rho_levels, rho_norm, CMAP),
        *[
            (field, panel_levels[index + 1], eps_norm, EPS_CMAP)
            for index, field in enumerate(error_fields)
        ],
    ]
    for axis, atoms, (field, field_levels, norm, cmap), title in zip(
        axes, atoms_all, display, PANEL_TITLES
    ):
        axis.set_aspect("equal")
        axis.set_xlim(x_min - pad, x_max + pad)
        axis.set_ylim(y_min - pad, y_max + pad)
        axis.axis("off")
        axis.set_title(title, pad=2.0)
        if args.world_axes:
            draw_world_axes(axis, camera, 0.50 * span)
        draw_isosurfaces(axis, atoms, field, field_levels, norm, camera, cmap)
        draw_unit_cell(axis, atoms, camera)
        draw_atoms(axis, atoms, camera, radius_scale)
        add_crystal_axes(axis, atoms, camera)

    add_log_colorbar(
        figure, (0.94, 0.52, 0.014, 0.34), CMAP, rho_norm,
        r"$\rho(\mathbf{r})$ [${\rm e}/{\rm \AA}^3$]",
    )
    add_log_colorbar(
        figure, (0.94, 0.13, 0.014, 0.34), EPS_CMAP, eps_norm,
        r"$\delta_{\rho}(\mathbf{r})$ [%]",
    )
    add_atom_legend(figure, atoms_all[0])
    if args.world_axes:
        figure.text(
            0.01, 0.02,
            "small xyz: crystal frame; large XYZ: shared world frame",
            fontsize=8.5, color="0.3", ha="left",
        )

    png_path = args.output.resolve()
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    if args.extra_figs_dir is not None:
        args.extra_figs_dir.mkdir(parents=True, exist_ok=True)
        for saved_path in (png_path, pdf_path):
            shutil.copyfile(saved_path, args.extra_figs_dir / saved_path.name)
            print(f"Copied to {args.extra_figs_dir / saved_path.name}")
    plt.show()
    summary_path = png_path.parent / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("outputs", {})["ase_eq_gaas"] = {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "world_axes": bool(args.world_axes),
            "rho_levels_e_per_angstrom3": rho_levels.tolist(),
            "error_levels_percent": {
                name: field_levels.tolist()
                for name, field_levels in zip(PANEL_NAMES[1:], panel_levels[1:])
            },
            "grid_rolls_undone": {k: list(v) for k, v in grid_rolls.items() if v is not None},
            "min_reference_abs_density": reference_floor,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"ASE {__import__('ase').__version__}")
    print(f"Density grid: {next(iter(shapes))}")
    print(f"min |rho_a| on grid = {reference_floor:.6g} e/Ang^3 (relative error defined everywhere)")
    print("rho isosurface levels [e/Ang^3]: " + ", ".join(f"{value:.8g}" for value in rho_levels))
    for name, field, field_levels in zip(PANEL_NAMES[1:], error_fields, panel_levels[1:]):
        roll = grid_rolls[name]
        roll_text = f" (grid roll {roll} undone)" if roll is not None else ""
        print(
            f"{name}{roll_text}: eps max {field.max():.3e} %, "
            "levels [%] " + ", ".join(f"{value:.3e}" for value in field_levels)
        )
    print(f"World axes: {args.world_axes}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
