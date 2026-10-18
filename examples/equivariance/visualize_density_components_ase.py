#!/usr/bin/env python3
"""Infer rho_init, rho_env, and their total for a molecule and export four
annotation-free single-panel figures: structure, env, init, sum."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
sys.path[:0] = [str(REPO_ROOT)]
os.environ.setdefault("AIDEN_GRAPH_BACKEND", "numpy")
os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / "results-h2o-components" / ".matplotlib")
)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from ase import Atoms  # noqa: E402
from aiden.data import load_element_descriptors  # noqa: E402
from aiden.fast_infer import (  # noqa: E402
    FastAidenInference,
    GPUProbeGraphBuilder,
    iter_regular_grid_queries,
)
from aiden.model import Aiden  # noqa: E402
from run_equivariance import read_cif, structure_sample  # noqa: E402
from visualize_equivariance_ase import (  # noqa: E402
    draw_atoms,
    draw_isosurfaces,
    marching_tetrahedra,
    project,
    rotation_matrix,
)


FIGSIZE = (3.0, 3.0)
# Frontal view: camera along the molecular z axis, molecule rotated so the
# oxygen sits on top and the two hydrogens spread symmetrically below.
CAMERA_ROTATION = "-90z"
NUM_ISOSURFACES = 4
# Density decays to machine noise far from the molecule; isosurface levels
# only span this many decades below each field's own maximum.
DENSITY_LEVEL_WINDOW = 1.0e4
# The dashed gray envelope in the "sum" panel runs one decade further out.
OUTLINE_WINDOW = 1.0e5
VIEW_MARGIN = 2.6   # Angstrom of empty space around the molecule in each figure
ATOM_SCALE = 1.15   # ball radius in Angstrom per unit covalent radius
INIT_CMAP_NAME = "summer"
ENV_CMAP_NAME = "autumn"
ENV_NEGATIVE_CMAP_NAME = "Blues"
PANELS = ("structure", "env", "init", "sum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", type=Path, default=ROOT / "H2O.cif")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "aiden-pbe.pt")
    parser.add_argument("--element-csv", type=Path, default=REPO_ROOT / "data" / "elem-des.csv")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results-h2o-components"
    )
    parser.add_argument("--grid", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--query-block-size", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="redraw figures from an existing density NPZ without model inference",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode_components(
    fast: FastAidenInference,
    prepared,
    query_frac: torch.Tensor,
    probe_edge_atom: torch.Tensor,
    probe_edge_query: torch.Tensor,
    probe_edge_vec: torch.Tensor,
    probe_edge_dist: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaled ``rho_init`` and ``rho_env`` for one probe block."""
    decoder = fast.decoder
    num_queries = int(query_frac.shape[0])
    valid = (
        (probe_edge_dist < decoder.orbital_cutoff)
        & (probe_edge_dist >= 0.0)
        & (probe_edge_query >= 0)
        & (probe_edge_query < num_queries)
    )
    atom_index = probe_edge_atom[valid].long()
    query_index = probe_edge_query[valid].long()
    edge_vector = probe_edge_vec[valid]
    distance = probe_edge_dist[valid]

    radial_s = decoder.gto.radial_values(distance)[:, 0]
    rho_init_normalized = decoder.initial_guess(
        prepared.atom_z,
        atom_index,
        query_index,
        radial_s,
        num_queries,
    )
    edge_factors = fast._evaluate_factors(
        prepared,
        atom_index,
        edge_vector,
        distance,
    )
    left = edge_factors.new_zeros((num_queries, decoder.num_orbitals))
    right = edge_factors.new_zeros((num_queries, decoder.num_orbitals))
    left.index_add_(0, query_index, edge_factors[:, 0])
    right.index_add_(0, query_index, edge_factors[:, 1])
    if decoder.use_abs:
        left = torch.nn.functional.smooth_l1_loss(
            left * decoder.abs_scale,
            torch.zeros_like(left),
            reduction="none",
        ) / decoder.abs_scale
    rho_env_normalized = (left * right).sum(dim=-1)
    scale = fast.model.density_scale.to(rho_init_normalized)
    return rho_init_normalized * scale, rho_env_normalized * scale


@torch.inference_mode()
def predict_full_grid_components(
    fast: FastAidenInference,
    prepared,
    grid: tuple[int, int, int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    builder = GPUProbeGraphBuilder(
        prepared.atom_frac,
        prepared.lattice,
        fast.decoder.orbital_cutoff,
        strict_order=True,
    )
    initial_parts: list[torch.Tensor] = []
    environment_parts: list[torch.Tensor] = []
    for query_frac in iter_regular_grid_queries(
        grid,
        block_size,
        prepared.atom_frac.device,
        prepared.atom_frac.dtype,
    ):
        atom_index, query_index, edge_vector, distance = builder.build(query_frac)
        rho_init, rho_env = decode_components(
            fast,
            prepared,
            query_frac,
            atom_index,
            query_index,
            edge_vector,
            distance,
        )
        initial_parts.append(rho_init)
        environment_parts.append(rho_env)
    return torch.cat(initial_parts), torch.cat(environment_parts)


def windowed_levels(field: np.ndarray) -> tuple[LogNorm, np.ndarray]:
    """Log-spaced isosurface levels within a fixed window below the maximum."""
    positive = np.asarray(field)[np.asarray(field) > 0.0]
    if positive.size == 0:
        raise ValueError("density component contains no positive values")
    upper = float(positive.max())
    lower = max(float(positive.min()), upper / DENSITY_LEVEL_WINDOW)
    if not upper > lower:
        raise ValueError("density component has no positive dynamic range")
    bounds = np.geomspace(lower, upper, NUM_ISOSURFACES + 2)
    return LogNorm(vmin=lower, vmax=upper), bounds[1:-1]


def molecule_limits(atoms: Atoms, camera: np.ndarray) -> tuple[float, float, float, float]:
    """Axis limits from the projected atoms plus a fixed margin in Angstrom.

    Using the atoms (rather than the vacuum-box corners) keeps the molecule
    large in frame, and the margin guarantees the atom balls are never cut.
    """
    cell = np.asarray(atoms.cell.array, dtype=float)
    centered = np.asarray(atoms.positions) - 0.5 * cell.sum(axis=0)
    projected, _ = project(centered, camera)
    x_min, y_min = projected.min(axis=0)
    x_max, y_max = projected.max(axis=0)
    return (
        x_min - VIEW_MARGIN,
        x_max + VIEW_MARGIN,
        y_min - VIEW_MARGIN,
        y_max + VIEW_MARGIN,
    )


def env_layer_list(
    fields: dict[str, np.ndarray],
    positive_alpha: float | None = None,
    negative_alpha: float = 0.24,
) -> list[tuple]:
    layers = [("surface", fields["rho_env"], ENV_CMAP_NAME, positive_alpha)]
    if np.any(fields["rho_env"] < 0.0):
        layers.append(
            ("surface", -fields["rho_env"], ENV_NEGATIVE_CMAP_NAME, negative_alpha)
        )
    return layers


def panel_layers(name: str, fields: dict[str, np.ndarray]) -> list[tuple]:
    """Rendering layers per panel; "sum" overlays both components' colors.

    In "sum" the environment head is drawn more translucently than the
    initial head so both component colors stay visible through the overlay.
    """
    if name == "structure":
        return []
    if name == "init":
        return [("surface", fields["rho_init"], INIT_CMAP_NAME, None)]
    if name == "env":
        return env_layer_list(fields)
    return [
        ("surface", fields["rho_init"], INIT_CMAP_NAME, 0.50),
        *env_layer_list(fields, positive_alpha=0.12, negative_alpha=0.12),
        ("outline", fields["rho_total"], None, None),
    ]


def projected_field_triangles(
    atoms: Atoms,
    field: np.ndarray,
    level: float,
    camera: np.ndarray,
) -> np.ndarray:
    """Isosurface triangles of ``field`` at ``level``, projected to 2D."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    center = 0.5 * cell.sum(axis=0)
    denominator = np.asarray(field.shape, dtype=float) - 1.0
    triangles = marching_tetrahedra(field, float(level))
    if not triangles.size:
        return np.empty((0, 3, 2))
    fractional = triangles / denominator[None, None, :]
    cartesian = fractional @ cell - center
    projected, _ = project(cartesian, camera)
    return projected


def outline_level(field: np.ndarray) -> float:
    return float(field[field > 0.0].max()) / OUTLINE_WINDOW


def draw_outline_silhouette(
    axis,
    projected: np.ndarray,
    limits: tuple[float, float, float, float],
) -> None:
    """One closed dashed gray curve around the projected outer isosurface.

    Rasterizes the outer isosurface triangles, then traces the per-row
    extremes of the union, so a single silhouette outline replaces the
    (far too dense to read as dashed) surface wireframe.
    """
    from PIL import Image, ImageDraw

    x_min, x_max, y_min, y_max = limits
    pixels = 1600
    scale_x = (pixels - 1) / (x_max - x_min)
    scale_y = (pixels - 1) / (y_max - y_min)
    polygons = np.stack(
        (
            (projected[..., 0] - x_min) * scale_x,
            (y_max - projected[..., 1]) * scale_y,
        ),
        axis=-1,
    )
    raster = Image.new("1", (pixels, pixels), 0)
    painter = ImageDraw.Draw(raster)
    for polygon in polygons:
        painter.polygon([tuple(vertex) for vertex in polygon], fill=1)
    mask = np.asarray(raster)
    filled_rows = np.flatnonzero(mask.any(axis=1))
    if filled_rows.size == 0:
        return
    left = np.argmax(mask, axis=1)
    right = pixels - 1 - np.argmax(mask[:, ::-1], axis=1)

    def to_data(row: int, column: int) -> tuple[float, float]:
        return (
            x_min + column / scale_x,
            y_max - row / scale_y,
        )

    outline = [to_data(row, left[row]) for row in filled_rows]
    outline += [to_data(row, right[row]) for row in filled_rows[::-1]]
    outline.append(outline[0])
    axis.plot(
        [point[0] for point in outline],
        [point[1] for point in outline],
        color="0.45",
        linewidth=0.7,
        linestyle=(0, (3.0, 2.4)),
        solid_capstyle="butt",
        zorder=8,
    )


def draw_component_figure(
    atoms: Atoms,
    layers: list[tuple],
    output_png: Path,
    dpi: int,
) -> Path:
    camera = rotation_matrix(CAMERA_ROTATION)
    x_min, x_max, y_min, y_max = molecule_limits(atoms, camera)
    outline_projected = None
    for kind, magnitude, _, _ in layers:
        if kind == "outline":
            outline_projected = projected_field_triangles(
                atoms, magnitude, outline_level(magnitude), camera
            )
    if outline_projected is not None and outline_projected.size:
        outline_pad = 0.4
        x_min = min(x_min, float(outline_projected[..., 0].min()) - outline_pad)
        x_max = max(x_max, float(outline_projected[..., 0].max()) + outline_pad)
        y_min = min(y_min, float(outline_projected[..., 1].min()) - outline_pad)
        y_max = max(y_max, float(outline_projected[..., 1].max()) + outline_pad)

    figure, axis = plt.subplots(figsize=FIGSIZE)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    axis.set_aspect("equal")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.axis("off")
    for kind, magnitude, cmap_name, alpha in layers:
        if kind == "outline":
            draw_outline_silhouette(
                axis, outline_projected, (x_min, x_max, y_min, y_max)
            )
            continue
        normalization, levels = windowed_levels(magnitude)
        draw_isosurfaces(
            axis,
            atoms,
            magnitude,
            levels,
            normalization,
            camera,
            cmap=plt.get_cmap(cmap_name),
            alpha=alpha,
            rasterized=False,
        )
    draw_atoms(axis, atoms, camera, ATOM_SCALE)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    figure.savefig(
        output_png,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
    return output_pdf


def draw_all_figures(
    atoms: Atoms,
    fields: dict[str, np.ndarray],
    output_dir: Path,
    dpi: int,
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for name in PANELS:
        output_png = output_dir / f"{name}.png"
        output_pdf = draw_component_figure(
            atoms, panel_layers(name, fields), output_png, dpi
        )
        outputs[name] = {"png": str(output_png), "pdf": str(output_pdf)}
        print(f"Figure saved: {output_png} and {output_pdf}", flush=True)
    return outputs


def component_statistics(field: np.ndarray, volume: float) -> dict[str, float]:
    return {
        "min_e_per_angstrom3": float(field.min()),
        "max_e_per_angstrom3": float(field.max()),
        "mean_e_per_angstrom3": float(field.mean(dtype=np.float64)),
        "integrated_electrons": float(field.mean(dtype=np.float64) * volume),
    }


def main() -> None:
    args = parse_args()
    grid = tuple(int(value) for value in args.grid)
    if any(value <= 1 for value in grid):
        raise ValueError("each grid dimension must be greater than one")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    block_size = args.query_block_size or (65536 if device.type == "cuda" else 2048)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(output_dir / ".matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )

    if args.render_only:
        density_path = output_dir / "h2o_density_components.npz"
        with np.load(density_path) as archive:
            archive_grid = tuple(int(value) for value in archive["grid_shape"])
            expected_size = int(np.prod(archive_grid))
            rho_init_flat = np.asarray(archive["rho_init"], dtype=np.float32)
            rho_env_flat = np.asarray(archive["rho_env"], dtype=np.float32)
            rho_total_flat = np.asarray(archive["rho_total"], dtype=np.float32)
            if any(array.size != expected_size for array in (rho_init_flat, rho_env_flat, rho_total_flat)):
                raise ValueError("density array size does not match stored grid_shape")
            atoms = Atoms(
                numbers=np.asarray(archive["atomic_numbers"], dtype=np.int64),
                scaled_positions=np.asarray(archive["atom_frac"], dtype=float),
                cell=np.asarray(archive["lattice"], dtype=float),
                pbc=True,
            )
        fields = {
            "rho_init": rho_init_flat.reshape(archive_grid, order="F"),
            "rho_env": rho_env_flat.reshape(archive_grid, order="F"),
            "rho_total": rho_total_flat.reshape(archive_grid, order="F"),
        }
        figures = draw_all_figures(atoms, fields, output_dir, args.dpi)
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["render_dpi"] = args.dpi
            summary["figures"] = figures
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        print(
            f"Figures complete from {density_path}: grid={archive_grid}", flush=True
        )
        plt.show()
        return

    crystal = read_cif(args.cif.resolve())
    symbol_to_z, symbol_to_feature, _, feature_names, _ = load_element_descriptors(
        args.element_csv.resolve()
    )
    try:
        checkpoint = torch.load(
            args.checkpoint.resolve(), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
    model_args: dict[str, Any] = checkpoint["model_args"]
    if len(feature_names) != int(model_args["atom_feat_dim"]):
        raise RuntimeError("element-descriptor width does not match checkpoint")

    model = Aiden(**model_args)
    state_key = "ema_model" if "ema_model" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval()
    fast = FastAidenInference(model)
    sample = structure_sample(
        crystal,
        symbol_to_z,
        symbol_to_feature,
        model_args,
        device,
    )

    start = time.perf_counter()
    with torch.inference_mode():
        prepared = fast.prepare(sample)
        rho_init_tensor, rho_env_tensor = predict_full_grid_components(
            fast, prepared, grid, block_size
        )
    synchronize(device)
    inference_seconds = time.perf_counter() - start

    rho_init_flat = rho_init_tensor.cpu().numpy().astype(np.float32, copy=False)
    rho_env_flat = rho_env_tensor.cpu().numpy().astype(np.float32, copy=False)
    rho_total_flat = rho_init_flat + rho_env_flat
    fields = {
        "rho_init": rho_init_flat.reshape(grid, order="F"),
        "rho_env": rho_env_flat.reshape(grid, order="F"),
        "rho_total": rho_total_flat.reshape(grid, order="F"),
    }

    density_path = output_dir / "h2o_density_components.npz"
    np.savez_compressed(
        density_path,
        rho_init=rho_init_flat,
        rho_env=rho_env_flat,
        rho_total=rho_total_flat,
        grid_shape=np.asarray(grid, dtype=np.int64),
        lattice=crystal.lattice.astype(np.float64),
        atom_frac=crystal.frac.astype(np.float64),
        atomic_numbers=np.asarray(
            [symbol_to_z[symbol] for symbol in crystal.symbols], dtype=np.int64
        ),
        density_units=np.asarray("electron/angstrom^3"),
    )
    print(
        f"Inference complete: grid={grid}, seconds={inference_seconds:.3f}; "
        f"saved {density_path}",
        flush=True,
    )

    atoms = Atoms(
        symbols=list(crystal.symbols),
        scaled_positions=crystal.frac,
        cell=crystal.lattice,
        pbc=True,
    )
    figures = draw_all_figures(atoms, fields, output_dir, args.dpi)
    volume = abs(float(np.linalg.det(crystal.lattice)))
    reconstruction_error = float(
        np.max(np.abs(rho_total_flat - (rho_init_flat + rho_env_flat)))
    )
    summary = {
        "system": crystal.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_state": state_key,
        "device": str(device),
        "grid_shape": list(grid),
        "inference_seconds": inference_seconds,
        "component_definition": {
            "rho_init": "model decoder pred_initial (element-only initial-density head)",
            "rho_env": "model decoder pred_learned (environment-dependent head)",
            "rho_total": "rho_init + rho_env",
        },
        "rho_init": component_statistics(rho_init_flat, volume),
        "rho_env": component_statistics(rho_env_flat, volume),
        "rho_total": component_statistics(rho_total_flat, volume),
        "max_abs_total_reconstruction_error": reconstruction_error,
        "panel_order": list(PANELS),
        "figure_has_annotations": False,
        "unit_cell_box_drawn": False,
        "camera_rotation": CAMERA_ROTATION,
        "render_dpi": args.dpi,
        "density_surfaces_rasterized_in_pdf": False,
        "panel_colormaps": {
            "init": INIT_CMAP_NAME,
            "env_positive": ENV_CMAP_NAME,
            "env_negative": ENV_NEGATIVE_CMAP_NAME,
            "sum": f"{INIT_CMAP_NAME} + {ENV_CMAP_NAME} overlay, dashed gray total-density outline",
        },
        "figsize_inches": list(FIGSIZE),
        "view_margin_angstrom": VIEW_MARGIN,
        "figures": figures,
        "density_npz": str(density_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    plt.show()


if __name__ == "__main__":
    main()
