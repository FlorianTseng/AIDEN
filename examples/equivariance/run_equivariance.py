#!/usr/bin/env python3
"""Infer a crystal charge density under translation, rotation, and inversion.

The model implementation and the encoder-cached decoder come from the
``aiden`` package at the repository root.  Density arrays use the reference code's VASP order (x index
fastest) and units of electron / Angstrom^3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AIDEN_GRAPH_BACKEND", "numpy")

from aiden.data import build_atom_graph_np, load_element_descriptors  # noqa: E402
from aiden.fast_infer import FastAidenInference  # noqa: E402
from aiden.model import Aiden  # noqa: E402


@dataclass(frozen=True)
class Crystal:
    name: str
    lattice: np.ndarray
    frac: np.ndarray
    symbols: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cif", type=Path, default=ROOT / "GaAs_mp-2534_symmetrized.cif"
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "aiden-pbe.pt")
    parser.add_argument("--element-csv", type=Path, default=REPO_ROOT / "data" / "elem-des.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results-gaas")
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=(32, 32, 32),
        metavar=("NX", "NY", "NZ"),
        help="regular fractional density grid (default: 32 32 32)",
    )
    parser.add_argument(
        "--translation-grid-steps",
        type=int,
        nargs=3,
        default=(3, -5, 7),
        metavar=("SX", "SY", "SZ"),
        help="translation as integer grid steps, enabling an exact np.roll comparison",
    )
    parser.add_argument("--rotation-angle-deg", type=float, default=37.0)
    parser.add_argument(
        "--rotation-axis", type=float, nargs=3, default=(1.0, 2.0, -1.0)
    )
    parser.add_argument(
        "--query-block-size",
        type=int,
        default=0,
        help="0 chooses 65536 on CUDA and 2048 on CPU",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda")
    )
    parser.add_argument(
        "--equivariance-tolerance",
        type=float,
        default=1.0e-4,
        help="maximum normalized L1 error counted as a pass (default: 0.01%%)",
    )
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def _cif_number(value: str) -> float:
    """Parse a CIF number, ignoring a parenthesized uncertainty."""
    return float(re.sub(r"\([^)]*\)$", "", value))


def _tokenize_cif(text: str) -> list[str]:
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _parse_cif_tables(
    text: str,
) -> tuple[dict[str, str], list[tuple[list[str], list[list[str]]]]]:
    tokens = _tokenize_cif(text)
    scalars: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()
        if lower == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                tags.append(tokens[index].lower())
                index += 1
            if not tags:
                raise ValueError("encountered CIF loop_ without column names")
            values: list[str] = []
            while index < len(tokens):
                candidate = tokens[index]
                candidate_lower = candidate.lower()
                if (
                    candidate.startswith("_")
                    or candidate_lower == "loop_"
                    or candidate_lower.startswith("data_")
                ):
                    break
                values.append(candidate)
                index += 1
            if len(values) % len(tags):
                raise ValueError(
                    f"CIF loop with {len(tags)} columns has {len(values)} values"
                )
            rows = [
                values[start : start + len(tags)]
                for start in range(0, len(values), len(tags))
            ]
            loops.append((tags, rows))
            continue
        if token.startswith("_"):
            if index + 1 >= len(tokens):
                raise ValueError(f"missing value for CIF tag {token}")
            scalars[lower] = tokens[index + 1]
            index += 2
            continue
        index += 1
    return scalars, loops


def _cell_from_parameters(scalars: dict[str, str]) -> np.ndarray:
    a = _cif_number(scalars["_cell_length_a"])
    b = _cif_number(scalars["_cell_length_b"])
    c = _cif_number(scalars["_cell_length_c"])
    alpha = math.radians(_cif_number(scalars["_cell_angle_alpha"]))
    beta = math.radians(_cif_number(scalars["_cell_angle_beta"]))
    gamma = math.radians(_cif_number(scalars["_cell_angle_gamma"]))
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        raise ValueError("invalid CIF cell: gamma is collinear")
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * math.cos(gamma), b * sin_gamma, 0.0])
    c_x = c * math.cos(beta)
    c_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    c_vec = np.array([c_x, c_y, math.sqrt(max(c_z_sq, 0.0))])
    return np.vstack((a_vec, b_vec, c_vec))


def _custom_lattice(
    loops: list[tuple[list[str], list[list[str]]]],
) -> np.ndarray | None:
    required = {
        "_aiden_lattice_vector_id",
        "_aiden_lattice_vector_x",
        "_aiden_lattice_vector_y",
        "_aiden_lattice_vector_z",
    }
    for tags, rows in loops:
        if not required.issubset(tags):
            continue
        lookup = {tag: tags.index(tag) for tag in required}
        ordered = sorted(rows, key=lambda row: int(row[lookup["_aiden_lattice_vector_id"]]))
        if len(ordered) != 3:
            raise ValueError("custom _eqproof lattice loop must contain three vectors")
        return np.asarray(
            [
                [
                    _cif_number(row[lookup["_aiden_lattice_vector_x"]]),
                    _cif_number(row[lookup["_aiden_lattice_vector_y"]]),
                    _cif_number(row[lookup["_aiden_lattice_vector_z"]]),
                ]
                for row in ordered
            ],
            dtype=np.float64,
        )
    return None


def _symmetry_value(expression: str, point: np.ndarray) -> float:
    expression = expression.lower().replace(" ", "").replace("−", "-")
    values = {"x": point[0], "y": point[1], "z": point[2]}
    answer = 0.0
    pieces = re.findall(r"([+-]?)([^+-]+)", expression)
    if not pieces:
        raise ValueError(f"unsupported CIF symmetry expression: {expression!r}")
    for sign_text, body in pieces:
        sign = -1.0 if sign_text == "-" else 1.0
        if body in values:
            value = float(values[body])
        elif "/" in body:
            value = float(Fraction(body))
        else:
            value = float(body)
        answer += sign * value
    return answer


def _apply_symmetry(operation: str, point: np.ndarray) -> np.ndarray:
    components = [part.strip() for part in operation.split(",")]
    if len(components) != 3:
        raise ValueError(f"invalid CIF symmetry operation: {operation!r}")
    return np.asarray([_symmetry_value(part, point) for part in components]) % 1.0


def _element_symbol(value: str) -> str:
    match = re.match(r"[A-Z][a-z]?", value.strip())
    if match is None:
        raise ValueError(f"cannot extract element from CIF symbol {value!r}")
    return match.group(0)


def read_cif(path: Path) -> Crystal:
    scalars, loops = _parse_cif_tables(path.read_text(encoding="utf-8"))
    lattice = _custom_lattice(loops)
    if lattice is None:
        lattice = _cell_from_parameters(scalars)

    atom_tags: list[str] | None = None
    atom_rows: list[list[str]] | None = None
    symmetry_operations: list[str] = []
    symop_names = (
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    )
    for tags, rows in loops:
        if "_atom_site_fract_x" in tags:
            atom_tags, atom_rows = tags, rows
        for name in symop_names:
            if name in tags:
                symmetry_operations.extend(row[tags.index(name)] for row in rows)
    if atom_tags is None or atom_rows is None:
        raise ValueError(f"no fractional atom-site loop found in {path}")
    if not symmetry_operations:
        symmetry_operations = ["x,y,z"]

    symbol_tag = (
        "_atom_site_type_symbol"
        if "_atom_site_type_symbol" in atom_tags
        else "_atom_site_label"
    )
    indices = {
        tag: atom_tags.index(tag)
        for tag in (
            symbol_tag,
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        )
    }
    symbols: list[str] = []
    coordinates: list[np.ndarray] = []
    for row in atom_rows:
        symbol = _element_symbol(row[indices[symbol_tag]])
        point = np.asarray(
            [
                _cif_number(row[indices["_atom_site_fract_x"]]),
                _cif_number(row[indices["_atom_site_fract_y"]]),
                _cif_number(row[indices["_atom_site_fract_z"]]),
            ]
        )
        for operation in symmetry_operations:
            transformed = _apply_symmetry(operation, point)
            duplicate = False
            for old_symbol, old_point in zip(symbols, coordinates):
                delta = transformed - old_point
                delta -= np.rint(delta)
                if old_symbol == symbol and np.max(np.abs(delta)) < 1.0e-7:
                    duplicate = True
                    break
            if not duplicate:
                symbols.append(symbol)
                coordinates.append(transformed)
    frac = np.asarray(coordinates, dtype=np.float64)
    if abs(np.linalg.det(lattice)) < 1.0e-10:
        raise ValueError("CIF lattice is singular")
    return Crystal(path.stem, lattice, frac, tuple(symbols))


def _cell_parameters(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(lattice, axis=1)

    def angle(left: np.ndarray, right: np.ndarray) -> float:
        cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))

    angles = np.asarray(
        [
            angle(lattice[1], lattice[2]),
            angle(lattice[0], lattice[2]),
            angle(lattice[0], lattice[1]),
        ]
    )
    return lengths, angles


def write_cif(
    path: Path,
    crystal: Crystal,
    transform_name: str,
    transform: np.ndarray,
    translation_frac: np.ndarray,
) -> None:
    lengths, angles = _cell_parameters(crystal.lattice)
    lines = [
        f"data_{re.sub(r'[^A-Za-z0-9_]', '_', crystal.name)}",
        f"_audit_creation_method 'aiden equivariance {transform_name} transform'",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        f"_cell_length_a {lengths[0]:.12f}",
        f"_cell_length_b {lengths[1]:.12f}",
        f"_cell_length_c {lengths[2]:.12f}",
        f"_cell_angle_alpha {angles[0]:.12f}",
        f"_cell_angle_beta {angles[1]:.12f}",
        f"_cell_angle_gamma {angles[2]:.12f}",
        f"_aiden_transform_name '{transform_name}'",
        f"_aiden_transform_determinant {np.linalg.det(transform):.12f}",
        "_aiden_translation_fractional "
        f"'{translation_frac[0]:.12f} {translation_frac[1]:.12f} {translation_frac[2]:.12f}'",
        "loop_",
        "_aiden_lattice_vector_id",
        "_aiden_lattice_vector_x",
        "_aiden_lattice_vector_y",
        "_aiden_lattice_vector_z",
    ]
    for index, vector in enumerate(crystal.lattice, start=1):
        lines.append(
            f"{index} {vector[0]:.12f} {vector[1]:.12f} {vector[2]:.12f}"
        )
    lines.extend(
        [
            "loop_",
            "_aiden_transform_row_id",
            "_aiden_transform_x",
            "_aiden_transform_y",
            "_aiden_transform_z",
        ]
    )
    for index, row in enumerate(transform, start=1):
        lines.append(f"{index} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}")
    lines.extend(
        [
            "loop_",
            "_atom_site_type_symbol",
            "_atom_site_label",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ]
    )
    counters: dict[str, int] = {}
    for symbol, point in zip(crystal.symbols, crystal.frac):
        counters[symbol] = counters.get(symbol, 0) + 1
        lines.append(
            f"{symbol} {symbol}{counters[symbol]} "
            f"{point[0]:.12f} {point[1]:.12f} {point[2]:.12f} 1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _species_groups(symbols: Iterable[str]) -> tuple[list[str], list[list[int]]]:
    species: list[str] = []
    groups: list[list[int]] = []
    for atom_index, symbol in enumerate(symbols):
        if symbol not in species:
            species.append(symbol)
            groups.append([])
        groups[species.index(symbol)].append(atom_index)
    return species, groups


def write_poscar(path: Path, crystal: Crystal) -> None:
    species, groups = _species_groups(crystal.symbols)
    lines = [f"{crystal.name} (aiden equivariance exact oriented cell)", "1.0"]
    lines.extend("  " + "  ".join(f"{value:.12f}" for value in row) for row in crystal.lattice)
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(len(group)) for group in groups))
    lines.append("Direct")
    for group in groups:
        for index in group:
            lines.append("  " + "  ".join(f"{value:.12f}" for value in crystal.frac[index]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chgcar(path: Path, crystal: Crystal, density: np.ndarray, grid: tuple[int, int, int]) -> None:
    species, groups = _species_groups(crystal.symbols)
    lines = [f"{crystal.name} AIDEN predicted charge density", "1.0"]
    lines.extend("  " + "  ".join(f"{value:.12f}" for value in row) for row in crystal.lattice)
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(str(len(group)) for group in groups))
    lines.append("Direct")
    for group in groups:
        for index in group:
            lines.append("  " + "  ".join(f"{value:.12f}" for value in crystal.frac[index]))
    lines.extend(("", f"  {grid[0]}  {grid[1]}  {grid[2]}"))
    raw = np.asarray(density, dtype=np.float64) * abs(np.linalg.det(crystal.lattice))
    for start in range(0, raw.size, 5):
        lines.append(" ".join(f"{value:18.11E}" for value in raw[start : start + 5]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def axis_angle_rotation(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1.0e-12:
        raise ValueError("rotation axis must be nonzero")
    x, y, z = axis / norm
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis / norm, axis / norm) + sine * cross


def make_variants(
    original: Crystal,
    grid: tuple[int, int, int],
    translation_steps: tuple[int, int, int],
    rotation: np.ndarray,
) -> tuple[dict[str, Crystal], dict[str, dict[str, Any]]]:
    shift = np.asarray(translation_steps, dtype=np.float64) / np.asarray(grid, dtype=np.float64)
    identity = np.eye(3)
    inversion = -np.eye(3)
    variants = {
        "original": original,
        "translation": Crystal(
            f"{original.name}_translation",
            original.lattice.copy(),
            (original.frac + shift) % 1.0,
            original.symbols,
        ),
        "proper_rotation": Crystal(
            f"{original.name}_proper_rotation",
            original.lattice @ rotation.T,
            original.frac.copy(),
            original.symbols,
        ),
        "inversion": Crystal(
            f"{original.name}_inversion",
            original.lattice @ inversion.T,
            original.frac.copy(),
            original.symbols,
        ),
    }
    metadata = {
        "original": {"matrix": identity, "translation_frac": np.zeros(3)},
        "translation": {"matrix": identity, "translation_frac": shift},
        "proper_rotation": {"matrix": rotation, "translation_frac": np.zeros(3)},
        "inversion": {"matrix": inversion, "translation_frac": np.zeros(3)},
    }
    return variants, metadata


def structure_sample(
    crystal: Crystal,
    symbol_to_z: dict[str, int],
    symbol_to_feature: dict[str, np.ndarray],
    model_args: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    missing = sorted(set(crystal.symbols).difference(symbol_to_z))
    if missing:
        raise ValueError(f"element descriptors missing for {missing}")
    lattice = crystal.lattice.astype(np.float32)
    frac = crystal.frac.astype(np.float32)
    edge_src, edge_dst, edge_vec, edge_dist = build_atom_graph_np(
        frac,
        lattice,
        cutoff=float(model_args["atom_cutoff"]),
        exclude_self=True,
        max_neighbors=int(model_args["max_atom_neighbors"]),
    )
    arrays: dict[str, Any] = {
        "name": crystal.name,
        "lattice": lattice,
        "atom_frac": frac,
        "atom_z": np.asarray([symbol_to_z[symbol] for symbol in crystal.symbols], dtype=np.int64),
        "atom_features": np.asarray(
            [symbol_to_feature[symbol] for symbol in crystal.symbols], dtype=np.float32
        ),
        "atom_edge_src": edge_src,
        "atom_edge_dst": edge_dst,
        "atom_edge_vec": edge_vec,
        "atom_edge_dist": edge_dist,
    }
    return {
        key: torch.from_numpy(value).to(device) if isinstance(value, np.ndarray) else value
        for key, value in arrays.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def density_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    absolute = np.abs(delta)
    denominator = max(float(np.abs(expected).sum()), 1.0e-30)
    peak = max(float(np.abs(expected).max()), 1.0e-30)
    centered_actual = actual.astype(np.float64) - float(np.mean(actual))
    centered_expected = expected.astype(np.float64) - float(np.mean(expected))
    corr_denom = math.sqrt(
        float(np.dot(centered_actual, centered_actual))
        * float(np.dot(centered_expected, centered_expected))
    )
    correlation = (
        float(np.dot(centered_actual, centered_expected) / corr_denom)
        if corr_denom > 0.0
        else 1.0
    )
    return {
        "mae_e_per_angstrom3": float(absolute.mean()),
        "rmse_e_per_angstrom3": float(np.sqrt(np.mean(delta * delta))),
        "max_abs_e_per_angstrom3": float(absolute.max()),
        "normalized_l1": float(absolute.sum() / denominator),
        "max_abs_over_reference_peak": float(absolute.max() / peak),
        "pearson_correlation": correlation,
    }


def classify_group(results: dict[str, dict[str, Any]]) -> tuple[str, str]:
    translation = bool(results["translation"]["passes"])
    rotation = bool(results["proper_rotation"]["passes"])
    inversion = bool(results["inversion"]["passes"])
    if not rotation:
        return "inconclusive", "proper-rotation test failed, so none of E(3), SE(3), O(3), SO(3) is supported"
    lookup = {
        (True, True): ("E(3)", "translation + proper rotations + inversion/reflections"),
        (True, False): ("SE(3)", "translation + proper rotations, but not inversion/reflections"),
        (False, True): ("O(3)", "proper rotations + inversion/reflections, but not translation"),
        (False, False): ("SO(3)", "proper rotations only"),
    }
    return lookup[(translation, inversion)]


def save_plot(
    output_path: Path,
    predictions: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    grid: tuple[int, int, int],
) -> str | None:
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    names = ("translation", "proper_rotation", "inversion")
    figure, axes = plt.subplots(3, 3, figsize=(9.2, 8.0), constrained_layout=True)
    for row, name in enumerate(names):
        predicted = predictions[name].reshape(grid, order="F")[:, :, grid[2] // 2]
        reference = expected[name].reshape(grid, order="F")[:, :, grid[2] // 2]
        difference = predicted - reference
        low = min(float(predicted.min()), float(reference.min()))
        high = max(float(predicted.max()), float(reference.max()))
        axes[row, 0].imshow(reference.T, origin="lower", vmin=low, vmax=high, cmap="viridis")
        axes[row, 1].imshow(predicted.T, origin="lower", vmin=low, vmax=high, cmap="viridis")
        error_scale = max(float(np.max(np.abs(difference))), 1.0e-12)
        axes[row, 2].imshow(
            difference.T,
            origin="lower",
            vmin=-error_scale,
            vmax=error_scale,
            cmap="coolwarm",
        )
        axes[row, 2].set_xlabel(f"max |difference| = {error_scale:.2e}")
        axes[row, 0].set_ylabel(name.replace("_", " "))
    for column, title in enumerate(("expected", "inferred", "difference")):
        axes[0, column].set_title(title)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return str(output_path)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# AIDEN {summary['system']} equivariance test",
        "",
        f"Result: **{summary['classification']}** under the definitions used below.",
        "",
        "| transform | determinant | normalized L1 | max/peak | correlation | pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name in ("translation", "proper_rotation", "inversion"):
        result = summary["comparisons"][name]
        metric = result["metrics"]
        lines.append(
            f"| {name} | {result['transform_determinant']:.0f} | "
            f"{metric['normalized_l1']:.6e} | "
            f"{metric['max_abs_over_reference_peak']:.6e} | "
            f"{metric['pearson_correlation']:.9f} | "
            f"{'yes' if result['passes'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Practical pass threshold: normalized L1 <= {summary['equivariance_tolerance']:.3e} "
            f"({100.0 * summary['equivariance_tolerance']:.4g}% for this run); raw errors are "
            "reported so stricter thresholds can be applied.",
            "",
            "Definitions: E(3) = translations + O(3); SE(3) = translations + SO(3); "
            "O(3) includes proper rotations and inversion/reflections; SO(3) includes only proper rotations.",
            "",
            "The translated reference is an exact periodic grid roll. The proper-rotation and inversion "
            "tests transform the complete Cartesian lattice and atoms together, so their expected scalar "
            "density has the same fractional-grid indices.",
            "",
            "CIF note: standard cell lengths/angles do not retain an absolute Cartesian orientation or "
            "handedness. Generated CIFs therefore include a documented `_aiden_lattice_vector_*` loop; "
            "the accompanying POSCAR files store the same exact oriented cells in a standard format.",
            "",
            f"This is a numerical test on this checkpoint and {summary['system']}, not a formal proof for every structure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    grid = tuple(int(value) for value in args.grid)
    translation_steps = tuple(int(value) for value in args.translation_grid_steps)
    if any(value <= 0 for value in grid):
        raise ValueError("all grid dimensions must be positive")
    if args.equivariance_tolerance <= 0.0:
        raise ValueError("--equivariance-tolerance must be positive")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    block_size = args.query_block_size or (65536 if device.type == "cuda" else 2048)
    if block_size <= 0:
        raise ValueError("query block size must be positive")

    output_dir = args.output_dir.resolve()
    structure_dir = output_dir / "structures"
    density_dir = output_dir / "densities"
    structure_dir.mkdir(parents=True, exist_ok=True)
    density_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    original = read_cif(args.cif.resolve())
    rotation = axis_angle_rotation(np.asarray(args.rotation_axis), args.rotation_angle_deg)
    variants, transform_metadata = make_variants(
        original, grid, translation_steps, rotation
    )
    print(
        f"Expanded {args.cif.name}: {len(original.symbols)} atoms; "
        f"grid={grid}; device={device}; block={block_size}",
        flush=True,
    )

    # Write, then read each CIF back.  The custom lattice loop preserves the
    # exact oriented/left-handed tensors that ordinary CIF cell metrics omit.
    for name, crystal in variants.items():
        metadata = transform_metadata[name]
        cif_path = structure_dir / f"{name}.cif"
        write_cif(
            cif_path,
            crystal,
            name,
            metadata["matrix"],
            metadata["translation_frac"],
        )
        write_poscar(structure_dir / f"{name}.vasp", crystal)
        reloaded = read_cif(cif_path)
        if not (
            np.allclose(reloaded.lattice, crystal.lattice, atol=2.0e-10)
            and np.allclose(reloaded.frac, crystal.frac, atol=2.0e-10)
            and reloaded.symbols == crystal.symbols
        ):
            raise RuntimeError(f"generated CIF round-trip failed for {name}")
        variants[name] = reloaded

    symbol_to_z, symbol_to_feature, _, feature_names, _ = load_element_descriptors(
        args.element_csv.resolve()
    )
    try:
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
    model_args = checkpoint.get("model_args")
    if not isinstance(model_args, dict):
        raise RuntimeError("checkpoint is missing model_args")
    if len(feature_names) != int(model_args["atom_feat_dim"]):
        raise RuntimeError(
            f"element feature width {len(feature_names)} != model width {model_args['atom_feat_dim']}"
        )
    state_key = "model" if args.raw_weights or "ema_model" not in checkpoint else "ema_model"
    model_load_start = time.perf_counter()
    model = Aiden(**model_args)
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval()
    fast = FastAidenInference(model)
    synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    predictions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    integrals: dict[str, float] = {}
    for name in ("original", "translation", "proper_rotation", "inversion"):
        crystal = variants[name]
        print(f"Inferring {name} ...", flush=True)
        start = time.perf_counter()
        sample = structure_sample(
            crystal, symbol_to_z, symbol_to_feature, model_args, device
        )
        with torch.inference_mode():
            prepared = fast.prepare(sample)
            density = fast.predict_full_grid(prepared, grid, block_size)
        synchronize(device)
        elapsed = time.perf_counter() - start
        density_np = density.detach().cpu().numpy().astype(np.float32, copy=False)
        predictions[name] = density_np
        timings[name] = elapsed
        volume = abs(float(np.linalg.det(crystal.lattice)))
        integrals[name] = float(np.mean(density_np, dtype=np.float64) * volume)
        np.savez_compressed(
            density_dir / f"{name}.npz",
            density_pred=density_np,
            grid_shape=np.asarray(grid, dtype=np.int64),
            lattice=crystal.lattice.astype(np.float64),
            atom_frac=crystal.frac.astype(np.float64),
            atomic_numbers=np.asarray([symbol_to_z[s] for s in crystal.symbols], dtype=np.int64),
            transform_matrix=transform_metadata[name]["matrix"].astype(np.float64),
            translation_frac=transform_metadata[name]["translation_frac"].astype(np.float64),
            density_units=np.asarray("electron/angstrom^3"),
        )
        write_chgcar(density_dir / f"{name}.CHGCAR", crystal, density_np, grid)
        print(
            f"  {elapsed:.2f} s; integrated density={integrals[name]:.6f} e",
            flush=True,
        )
        del density, prepared, sample

    original_grid = predictions["original"].reshape(grid, order="F")
    expected = {
        "translation": np.roll(
            original_grid,
            shift=translation_steps,
            axis=(0, 1, 2),
        ).reshape(-1, order="F"),
        "proper_rotation": predictions["original"],
        "inversion": predictions["original"],
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for name in ("translation", "proper_rotation", "inversion"):
        metrics = density_metrics(predictions[name], expected[name])
        comparisons[name] = {
            "metrics": metrics,
            "passes": bool(metrics["normalized_l1"] <= args.equivariance_tolerance),
            "expected_mapping": (
                f"rho_variant[i,j,k] = rho_original[(i-({translation_steps[0]})) mod {grid[0]}, "
                f"(j-({translation_steps[1]})) mod {grid[1]}, "
                f"(k-({translation_steps[2]})) mod {grid[2]}]"
                if name == "translation"
                else "same fractional-grid index after transforming the full Cartesian cell"
            ),
            "transform_determinant": float(np.linalg.det(transform_metadata[name]["matrix"])),
        }
    classification, classification_detail = classify_group(comparisons)

    plot_path = save_plot(
        output_dir / "equivariance_comparison.png", predictions, expected, grid
    )
    summary: dict[str, Any] = {
        "status": "ok",
        "system": original.name,
        "classification": classification,
        "classification_detail": classification_detail,
        "definitions": {
            "E(3)": "translations + O(3)",
            "SE(3)": "translations + SO(3)",
            "O(3)": "proper rotations + inversion/reflections, without asserted translations",
            "SO(3)": "proper rotations only, without asserted translations or reflections",
        },
        "equivariance_tolerance": float(args.equivariance_tolerance),
        "input_cif": str(args.cif.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight_state": state_key,
        "model_name": model_args.get("model_name"),
        "device": str(device),
        "torch_version": torch.__version__,
        "atoms": len(original.symbols),
        "species_counts": {
            symbol: original.symbols.count(symbol) for symbol in dict.fromkeys(original.symbols)
        },
        "grid_shape": list(grid),
        "grid_points": int(np.prod(grid)),
        "query_block_size": int(block_size),
        "translation_grid_steps": list(translation_steps),
        "translation_fractional": transform_metadata["translation"]["translation_frac"].tolist(),
        "proper_rotation_axis": list(args.rotation_axis),
        "proper_rotation_angle_degrees": float(args.rotation_angle_deg),
        "proper_rotation_matrix": rotation.tolist(),
        "comparisons": comparisons,
        "integrated_density_electrons": integrals,
        "timing_seconds": {
            "model_load": model_load_seconds,
            "per_variant_structure_graph_encode_decode": timings,
            "total": time.perf_counter() - total_start,
        },
        "outputs": {
            "structure_dir": str(structure_dir),
            "density_dir": str(density_dir),
            "comparison_plot": plot_path,
        },
        "caveat": (
            f"Numerical evidence for this checkpoint on {original.name}; not a formal architecture-level proof. "
            "Generated CIFs use custom _eqproof lattice-vector tags because standard CIF metrics do "
            "not preserve global Cartesian orientation/handedness."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
