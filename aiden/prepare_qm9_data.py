#!/usr/bin/env python3
"""Convert the supplied QM9 CHGCAR HDF5 files for Aiden training.

The source files contain molecular geometries in VASP vacuum boxes.  Aiden's
existing dataset interface builds periodic neighbour graphs, whereas the QM9
task must not use molecular periodic images.  This converter therefore
translates every molecule and its labelled probe points together into the
middle of a large cubic *isolation cell*.  Relative atom--atom and atom--probe
positions (and thus density labels) are unchanged, while all periodic images
are farther away than the training cutoffs.

The result uses the existing ``aiden_packed_h5_v1`` schema, so no model or
dataset code changes are needed.  The original VASP cell and its physical
volume are retained as metadata; the stored ``lattice`` is deliberately the
synthetic isolation cell used exclusively by the existing graph construction.

Example
-------
python prepare_qm9_data.py \
  --input-root data/qm9-h5 \
  --output-root data/aiden-qm9
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np


PACKED_H5_SCHEMA = "aiden_packed_h5_v1"
GEOMETRY_MODE = "nonperiodic_isolation_cell_v1"


SYMBOL_BY_ATOMIC_NUMBER = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert QM9 CHGCAR HDF5 into Aiden packed HDF5 with "
            "non-periodic molecular geometry."
        )
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/qm9-h5"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/aiden-qm9"),
    )
    parser.add_argument(
        "--isolation-cell-length",
        type=float,
        default=64.0,
        help="Side length of the synthetic cubic cell in Angstrom (default: 64).",
    )
    parser.add_argument(
        "--atom-cutoff",
        type=float,
        default=4.0,
        help="Cutoff used to precompute atom graph edges (default: 4 Angstrom).",
    )
    parser.add_argument("--max-atom-neighbors", type=int, default=100)
    parser.add_argument(
        "--max-structures",
        type=int,
        default=0,
        help="Convert at most this many structures; 0 converts every record.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace invalid or incompatible completed output shards.",
    )
    return parser.parse_args()


def _natural_group_key(key: str) -> tuple[int, str]:
    """Sort source keys such as ``10-chgcar_random`` numerically."""
    prefix, _, _ = key.partition("-")
    try:
        return int(prefix), key
    except ValueError:
        return math.inf, key


def _read_singleton(array: np.ndarray, name: str, ndim: int) -> np.ndarray:
    """Remove the source file's leading singleton sample dimension."""
    value = np.asarray(array)
    expected_ndim = ndim + 1
    if value.ndim != expected_ndim or value.shape[0] != 1:
        raise ValueError(
            f"{name} must have shape (1, ...), received {tuple(value.shape)}"
        )
    return value[0]


def _species_and_counts(atomic_numbers: np.ndarray) -> tuple[list[str], np.ndarray]:
    species: list[str] = []
    counts: list[int] = []
    for number in atomic_numbers.astype(np.int64).tolist():
        symbol = SYMBOL_BY_ATOMIC_NUMBER.get(int(number))
        if symbol is None:
            raise ValueError(f"unsupported atomic number in source data: {number}")
        if symbol in species:
            counts[species.index(symbol)] += 1
        else:
            species.append(symbol)
            counts.append(1)
    return species, np.asarray(counts, dtype=np.int64)


def _nonperiodic_atom_graph(
    atom_cart: np.ndarray,
    cutoff: float,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the encoder graph without image atoms.

    The returned source/destination convention and edge vector match the
    existing ``build_atom_graph_np`` implementation: ``edge_vec`` points from
    source atom to destination atom.
    """
    if cutoff <= 0.0:
        raise ValueError("atom cutoff must be positive")
    if max_neighbors <= 0:
        raise ValueError("max atom neighbors must be positive")

    positions = np.asarray(atom_cart, dtype=np.float64)
    n_atoms = int(positions.shape[0])
    source_parts: list[np.ndarray] = []
    destination_parts: list[np.ndarray] = []
    vector_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    for destination in range(n_atoms):
        vectors = positions[destination][None, :] - positions
        distances = np.linalg.norm(vectors, axis=1)
        valid = (distances > 1.0e-8) & (distances < float(cutoff))
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            continue
        # Stable sort makes tied distances deterministic and agrees with the
        # source-index order used by the existing vectorised graph backend.
        order = np.argsort(distances[candidates], kind="stable")
        candidates = candidates[order[:max_neighbors]]
        source_parts.append(candidates.astype(np.int64, copy=False))
        destination_parts.append(
            np.full(candidates.shape, destination, dtype=np.int64)
        )
        vector_parts.append(vectors[candidates].astype(np.float32, copy=False))
        distance_parts.append(distances[candidates].astype(np.float32, copy=False))

    if not source_parts:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.concatenate(source_parts),
        np.concatenate(destination_parts),
        np.concatenate(vector_parts),
        np.concatenate(distance_parts),
    )


def _graph_version(cutoff: float, max_neighbors: int) -> str:
    return (
        f"periodic_radius_v2_cutoff={float(cutoff):.8g}"
        f"_max_neighbors={int(max_neighbors)}"
    )


def _attr_as_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _valid_completed_shard(
    path: Path,
    start: int,
    count: int,
    graph_version: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r", swmr=True) as handle:
            return (
                str(handle.attrs.get("schema", "")) == PACKED_H5_SCHEMA
                and bool(handle.attrs.get("complete", False))
                and int(handle.attrs.get("start", -1)) == int(start)
                and int(handle.attrs.get("count", -1)) == int(count)
                and str(handle.attrs.get("atom_graph_version", "")) == graph_version
                and len(handle) == int(count)
            )
    except OSError:
        return False


def _require_isolation(
    atom_cart: np.ndarray,
    probe_cart: np.ndarray,
    cell_length: float,
    cutoff: float,
    source_name: str,
) -> np.ndarray:
    all_positions = np.concatenate([atom_cart, probe_cart], axis=0)
    coordinate_min = all_positions.min(axis=0)
    coordinate_max = all_positions.max(axis=0)
    span = coordinate_max - coordinate_min
    min_image_separation = float(cell_length) - float(span.max())
    if min_image_separation <= 2.0 * float(cutoff):
        raise ValueError(
            f"isolation cell is too small for {source_name}: its smallest "
            f"periodic-image separation lower bound is {min_image_separation:.4g} "
            f"Angstrom, while 2 * atom cutoff is {2.0 * cutoff:.4g}."
        )
    midpoint = 0.5 * (coordinate_min + coordinate_max)
    return np.full(3, 0.5 * float(cell_length), dtype=np.float64) - midpoint


def _write_record(
    destination: h5py.Group,
    source: h5py.Group,
    source_file: Path,
    source_key: str,
    isolation_cell_length: float,
    atom_cutoff: float,
    max_atom_neighbors: int,
    graph_version: str,
) -> None:
    atom_cart_source = _read_singleton(source["atom_pos"], "atom_pos", ndim=2)
    atomic_numbers = _read_singleton(source["atom_type"], "atom_type", ndim=1)
    source_lattice = _read_singleton(source["cell"], "cell", ndim=2)
    density = _read_singleton(source["chg"], "chg", ndim=1)
    query_cart_source = _read_singleton(source["probe_pos"], "probe_pos", ndim=2)
    probe_shape = np.asarray(source["probe_shape"], dtype=np.int64)

    if atom_cart_source.shape[-1] != 3 or query_cart_source.shape[-1] != 3:
        raise ValueError(f"{source_file.name}:{source_key} contains non-3D coordinates")
    if density.ndim != 1 or query_cart_source.shape[0] != density.size:
        raise ValueError(
            f"{source_file.name}:{source_key} has {query_cart_source.shape[0]} "
            f"probe positions but {density.size} density values"
        )
    if probe_shape.shape != (4,) or int(probe_shape[0]) != 1:
        raise ValueError(
            f"{source_file.name}:{source_key} has invalid probe_shape "
            f"{probe_shape.tolist()}"
        )
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    if atomic_numbers.ndim != 1 or atom_cart_source.shape[0] != atomic_numbers.size:
        raise ValueError(f"{source_file.name}:{source_key} atom coordinate/type mismatch")
    if np.any(atomic_numbers <= 0) or np.any(atomic_numbers > 118):
        raise ValueError(f"{source_file.name}:{source_key} has invalid atomic numbers")
    if not np.isfinite(atom_cart_source).all() or not np.isfinite(query_cart_source).all():
        raise ValueError(f"{source_file.name}:{source_key} contains non-finite coordinates")
    if not np.isfinite(density).all():
        raise ValueError(f"{source_file.name}:{source_key} contains non-finite density")

    translation = _require_isolation(
        atom_cart_source,
        query_cart_source,
        isolation_cell_length,
        atom_cutoff,
        f"{source_file.name}:{source_key}",
    )
    atom_cart = np.asarray(atom_cart_source + translation, dtype=np.float32)
    query_cart = np.asarray(query_cart_source + translation, dtype=np.float32)
    lattice = np.eye(3, dtype=np.float32) * float(isolation_cell_length)
    inverse_lattice = np.linalg.inv(lattice.astype(np.float64))
    frac_coords = np.asarray(atom_cart @ inverse_lattice, dtype=np.float32)
    query_frac = np.asarray(query_cart @ inverse_lattice, dtype=np.float32)
    if np.any(frac_coords < 0.0) or np.any(frac_coords >= 1.0):
        raise RuntimeError(f"{source_file.name}:{source_key} escaped its isolation cell")
    if np.any(query_frac < 0.0) or np.any(query_frac >= 1.0):
        raise RuntimeError(f"{source_file.name}:{source_key} probes escaped their isolation cell")

    source_edge, destination_edge, edge_vec, edge_dist = _nonperiodic_atom_graph(
        atom_cart,
        cutoff=atom_cutoff,
        max_neighbors=max_atom_neighbors,
    )
    species, counts = _species_and_counts(atomic_numbers)
    source_volume = float(abs(np.linalg.det(np.asarray(source_lattice, dtype=np.float64))))
    if source_volume <= 0.0:
        raise ValueError(f"{source_file.name}:{source_key} has non-positive source volume")

    destination.attrs["name"] = f"qm9:{source_file.name}:{source_key}"
    destination.attrs["species_json"] = json.dumps(species)
    # atom_features is intentionally omitted.  CHGCARRhoDataset derives it
    # from the caller-provided elem-des.csv, avoiding a duplicate feature schema.
    destination.attrs["feature_schema_version"] = ""
    destination.attrs["atom_graph_version"] = graph_version
    destination.attrs["source_file"] = source_file.name
    destination.attrs["source_group"] = source_key
    destination.attrs["source_lattice"] = np.asarray(source_lattice, dtype=np.float32)
    destination.attrs["source_volume"] = source_volume
    destination.attrs["geometry_mode"] = GEOMETRY_MODE
    destination.attrs["coordinate_translation"] = translation.astype(np.float32)
    destination.attrs["source_chgcar"] = _attr_as_text(
        source.attrs.get("file_source"),
        f"{source_file.name}:{source_key}",
    )
    destination.create_dataset("lattice", data=lattice, compression=None)
    destination.create_dataset("counts", data=counts, compression=None)
    destination.create_dataset("frac_coords", data=frac_coords, compression=None)
    destination.create_dataset("atomic_numbers", data=atomic_numbers, compression=None)
    destination.create_dataset("atom_cart", data=atom_cart, compression=None)
    destination.create_dataset("grid_shape", data=probe_shape[1:], compression=None)
    destination.create_dataset("density", data=np.asarray(density, dtype=np.float32), compression=None)
    destination.create_dataset("volume", data=np.asarray(source_volume, dtype=np.float32), compression=None)
    destination.create_dataset("all_query_frac", data=query_frac, compression=None)
    destination.create_dataset("all_query_cart", data=query_cart, compression=None)
    destination.create_dataset("atom_edge_src", data=source_edge, compression=None)
    destination.create_dataset("atom_edge_dst", data=destination_edge, compression=None)
    destination.create_dataset("atom_edge_vec", data=edge_vec, compression=None)
    destination.create_dataset("atom_edge_dist", data=edge_dist, compression=None)


def _write_shard(
    source_path: Path,
    source_keys: Sequence[str],
    output_path: Path,
    start: int,
    args: argparse.Namespace,
    graph_version: str,
) -> None:
    count = len(source_keys)
    if _valid_completed_shard(output_path, start, count, graph_version):
        print(f"[reuse] {output_path.name}: {count} structures", flush=True)
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists but is not a compatible completed shard; "
            "rerun with --overwrite after inspecting it."
        )

    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(source_path, "r", swmr=True) as source_handle, h5py.File(
            temporary,
            "w",
            libver="latest",
        ) as output_handle:
            output_handle.attrs["schema"] = PACKED_H5_SCHEMA
            output_handle.attrs["start"] = int(start)
            output_handle.attrs["count"] = int(count)
            output_handle.attrs["atom_graph_version"] = graph_version
            output_handle.attrs["geometry_mode"] = GEOMETRY_MODE
            output_handle.attrs["isolation_cell_length_angstrom"] = float(
                args.isolation_cell_length
            )
            output_handle.attrs["complete"] = False
            for local_index, source_key in enumerate(source_keys):
                global_index = start + local_index
                destination = output_handle.create_group(f"{global_index:08d}")
                _write_record(
                    destination,
                    source_handle[source_key],
                    source_path,
                    source_key,
                    isolation_cell_length=float(args.isolation_cell_length),
                    atom_cutoff=float(args.atom_cutoff),
                    max_atom_neighbors=int(args.max_atom_neighbors),
                    graph_version=graph_version,
                )
            output_handle.attrs["complete"] = True
            output_handle.flush()
        os.replace(temporary, output_path)
    except Exception:
        # Keep the partial file for diagnosis.  It is never mistaken for a
        # finished shard, and the next run replaces it safely.
        raise
    print(f"[written] {output_path.name}: {count} structures", flush=True)


def _source_file_plan(
    source_paths: Iterable[Path],
    max_structures: int,
) -> list[tuple[Path, list[str]]]:
    remaining = math.inf if max_structures == 0 else int(max_structures)
    plan: list[tuple[Path, list[str]]] = []
    for source_path in source_paths:
        if remaining <= 0:
            break
        with h5py.File(source_path, "r", swmr=True) as handle:
            keys = sorted(handle.keys(), key=_natural_group_key)
        if remaining != math.inf:
            keys = keys[: int(remaining)]
        if keys:
            plan.append((source_path, keys))
            remaining -= len(keys)
    return plan


def main() -> None:
    args = parse_args()
    if args.isolation_cell_length <= 0.0:
        raise ValueError("--isolation-cell-length must be positive")
    if args.atom_cutoff <= 0.0:
        raise ValueError("--atom-cutoff must be positive")
    if args.max_atom_neighbors <= 0:
        raise ValueError("--max-atom-neighbors must be positive")
    if args.max_structures < 0:
        raise ValueError("--max-structures must be non-negative")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    source_paths = sorted(input_root.glob("*.h5"))
    if not source_paths:
        raise FileNotFoundError(f"no .h5 files found under {input_root}")
    if output_root == input_root:
        raise ValueError("output root must differ from input root")
    output_root.mkdir(parents=True, exist_ok=True)

    plan = _source_file_plan(source_paths, int(args.max_structures))
    expected_structures = sum(len(keys) for _, keys in plan)
    if expected_structures == 0:
        raise RuntimeError("source plan contains no structures")
    graph_version = _graph_version(args.atom_cutoff, args.max_atom_neighbors)
    print(
        "Converting QM9 data "
        f"| structures={expected_structures} input_shards={len(plan)} "
        f"| isolation_cell={args.isolation_cell_length:.8g} A "
        f"| graph={graph_version}",
        flush=True,
    )

    shards: list[dict[str, object]] = []
    start = 0
    for shard_index, (source_path, source_keys) in enumerate(plan):
        output_path = output_root / f"shard-{shard_index:05d}.h5"
        _write_shard(
            source_path,
            source_keys,
            output_path,
            start,
            args,
            graph_version,
        )
        shards.append(
            {
                "file": output_path.name,
                "start": int(start),
                "count": int(len(source_keys)),
                "source_file": source_path.name,
            }
        )
        start += len(source_keys)

    manifest = {
        "schema": PACKED_H5_SCHEMA,
        "source_root": str(input_root),
        "num_structures": int(expected_structures),
        "atom_cutoff": float(args.atom_cutoff),
        "max_atom_neighbors": int(args.max_atom_neighbors),
        "atom_graph_version": graph_version,
        "geometry_mode": GEOMETRY_MODE,
        "isolation_cell_length_angstrom": float(args.isolation_cell_length),
        "source_density_units": "e/Angstrom^3 (validated from source CHGCAR values)",
        "shards": shards,
    }
    temporary_manifest = output_root / "manifest.json.partial"
    manifest_path = output_root / "manifest.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        f"QM9 packed dataset ready | structures={expected_structures} "
        f"shards={len(shards)} root={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
