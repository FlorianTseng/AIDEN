from __future__ import annotations

import json
import os
import tarfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class ParsedChgcar:
    name: str
    lattice: np.ndarray  # (3, 3)
    species: List[str]
    counts: List[int]
    frac_coords: np.ndarray  # (N, 3)
    atomic_numbers: np.ndarray  # (N,)
    atom_features: np.ndarray  # (N, F)
    grid_shape: Tuple[int, int, int]
    density: np.ndarray  # (nx*ny*nz,), in e/Ang^3
    volume: float
    atom_cart: Optional[np.ndarray] = None        # (N, 3)
    all_query_frac: Optional[np.ndarray] = None   # (n_grid, 3)
    all_query_cart: Optional[np.ndarray] = None   # (n_grid, 3)
    sample_probs: Optional[np.ndarray] = None
    atom_edge_src: Optional[np.ndarray] = None    # (E,)
    atom_edge_dst: Optional[np.ndarray] = None    # (E,)
    atom_edge_vec: Optional[np.ndarray] = None    # (E, 3)
    atom_edge_dist: Optional[np.ndarray] = None   # (E,)
    atom_graph_version: Optional[str] = None


PACKED_H5_SCHEMA = "aiden_packed_h5_v1"


@dataclass(frozen=True)
class PackedH5Record:
    shard_path: Path
    group_key: str
    order: int


InputRecord = Union[Path, PackedH5Record]


FEATURE_COLUMNS = [
    "atomic_number",
    "period",
    "group",
    "n_valence",
    "n_s_valence",
    "n_p_valence",
    "n_d_valence",
    "n_f_valence",
    "electronegativity",
    "atomic_radius",
    "covalent_radius",
    "first_ionization_energy",
    "electron_affinity",
    "dipole_polarizability",
    "ground_state_magmom",
]

FEATURE_SCHEMA_VERSION = "onehot_plus_elemdesc_concat_then_minmax_v1"


def load_element_descriptors(
    csv_path: str | Path,
) -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[int, np.ndarray], List[str], List[str]]:
    df = pd.read_csv(csv_path)
    expected = {"element", *FEATURE_COLUMNS}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"elem-des.csv is missing columns: {sorted(missing)}")

    df = df.copy()
    df["element"] = df["element"].astype(str).str.strip()

    if df["element"].duplicated().any():
        dup = df.loc[df["element"].duplicated(), "element"].tolist()
        raise ValueError(f"Duplicate element symbols found in elem-des.csv: {dup}")

    elements = df["element"].tolist()
    num_elements = len(elements)

    onehot = np.eye(num_elements, dtype=np.float32)
    desc = df[FEATURE_COLUMNS].astype(np.float32).to_numpy()

    feat_raw = np.concatenate([onehot, desc], axis=1)
    col_min = feat_raw.min(axis=0, keepdims=True)
    col_max = feat_raw.max(axis=0, keepdims=True)
    denom = np.where((col_max - col_min) > 0.0, (col_max - col_min), 1.0)
    feat_norm = ((feat_raw - col_min) / denom).astype(np.float32)

    onehot_feature_names = [f"onehot_{sym}" for sym in elements]
    feature_names = onehot_feature_names + FEATURE_COLUMNS.copy()

    symbol_to_z: Dict[str, int] = {}
    symbol_to_feat: Dict[str, np.ndarray] = {}
    z_to_feat: Dict[int, np.ndarray] = {}

    for i, row in df.iterrows():
        symbol = str(row["element"]).strip()
        z = int(row["atomic_number"])
        feat = feat_norm[i]
        symbol_to_z[symbol] = z
        symbol_to_feat[symbol] = feat
        z_to_feat[z] = feat

    return symbol_to_z, symbol_to_feat, z_to_feat, feature_names, elements


def _find_chgcar_member(tf: tarfile.TarFile) -> tarfile.TarInfo:
    members = [m for m in tf.getmembers() if m.isfile()]
    if not members:
        raise ValueError("No regular file found inside tar.gz archive.")
    for member in members:
        lower = member.name.lower()
        if "chgcar" in lower or lower.endswith(".vasp"):
            return member
    return members[0]


def _read_text_from_path(path: Path) -> str:
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(path, mode="r:gz") as tf:
            member = _find_chgcar_member(tf)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Failed to extract {member.name} from {path}")
            return extracted.read().decode("utf-8", errors="ignore")

    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, mode="rt", encoding="utf-8", errors="ignore") as fh:
            return fh.read()

    return path.read_text(encoding="utf-8", errors="ignore")


def parse_chgcar_text(
    text: str,
    name: str,
    symbol_to_z: Dict[str, int],
    symbol_to_feat: Dict[str, np.ndarray],
) -> ParsedChgcar:
    lines = text.splitlines()
    if len(lines) < 10:
        raise ValueError(f"CHGCAR content for {name} is too short.")

    idx = 1  # skip CHGCAR comment line

    scale = float(lines[idx].split()[0])
    idx += 1

    lattice = []
    for _ in range(3):
        lattice.append([float(x) for x in lines[idx].split()[:3]])
        idx += 1
    lattice = np.asarray(lattice, dtype=np.float64) * scale

    species = lines[idx].split()
    idx += 1

    counts = [int(x) for x in lines[idx].split()]
    idx += 1

    if len(species) != len(counts):
        raise ValueError(f"Species/count mismatch in {name}")

    coord_mode = lines[idx].strip().lower()
    if coord_mode.startswith("s"):
        idx += 1
        coord_mode = lines[idx].strip().lower()
    idx += 1

    direct = coord_mode.startswith("d")

    n_atoms = sum(counts)
    coords = []
    for _ in range(n_atoms):
        toks = lines[idx].split()
        coords.append([float(toks[0]), float(toks[1]), float(toks[2])])
        idx += 1
    coords = np.asarray(coords, dtype=np.float64)

    if direct:
        frac_coords = coords % 1.0
    else:
        frac_coords = np.linalg.solve(lattice.T, coords.T).T % 1.0

    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"Missing FFT grid dimensions in {name}")

    grid_toks = lines[idx].split()
    if len(grid_toks) < 3:
        raise ValueError(f"Invalid FFT grid line in {name}: {lines[idx]!r}")
    nx, ny, nz = [int(x) for x in grid_toks[:3]]
    n_grid = nx * ny * nz
    idx += 1

    raw_vals: List[float] = []
    while idx < len(lines) and len(raw_vals) < n_grid:
        stripped = lines[idx].strip()
        if stripped:
            raw_vals.extend(float(x) for x in stripped.split())
        idx += 1

    if len(raw_vals) < n_grid:
        raise ValueError(f"Density block truncated in {name}: expected {n_grid}, got {len(raw_vals)}")

    density = np.asarray(raw_vals[:n_grid], dtype=np.float32)
    volume = float(abs(np.linalg.det(lattice)))
    if volume <= 0:
        raise ValueError(f"Non-positive cell volume in {name}")

    density = density / volume

    atomic_numbers: List[int] = []
    atom_features: List[np.ndarray] = []
    for symbol, count in zip(species, counts):
        if symbol not in symbol_to_z:
            raise KeyError(f"Element {symbol!r} from {name} not found in elem-des.csv")
        z = symbol_to_z[symbol]
        feat = symbol_to_feat[symbol]
        atomic_numbers.extend([z] * count)
        atom_features.extend([feat] * count)

    return ParsedChgcar(
        name=name,
        lattice=lattice.astype(np.float32),
        species=species,
        counts=counts,
        frac_coords=frac_coords.astype(np.float32),
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int64),
        atom_features=np.asarray(atom_features, dtype=np.float32),
        grid_shape=(nx, ny, nz),
        density=density.astype(np.float32),
        volume=volume,
    )


def parse_chgcar_file(
    path: str | Path,
    symbol_to_z: Dict[str, int],
    symbol_to_feat: Dict[str, np.ndarray],
) -> ParsedChgcar:
    path = Path(path)
    text = _read_text_from_path(path)
    return parse_chgcar_text(text, path.name, symbol_to_z, symbol_to_feat)


def flat_indices_to_frac(indices: np.ndarray, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = grid_shape
    ix = indices % nx
    iy = (indices // nx) % ny
    iz = indices // (nx * ny)
    frac = np.stack([ix / nx, iy / ny, iz / nz], axis=-1).astype(np.float32)
    return frac


def _pbc_diff_frac_np(dst_frac: np.ndarray, src_frac: np.ndarray) -> np.ndarray:
    diff = dst_frac - src_frac
    return diff - np.round(diff)


_TORCH_CLUSTER_RADIUS = None
_TORCH_CLUSTER_IMPORT_ATTEMPTED = False


def _get_torch_cluster_radius():
    """Import torch-cluster lazily inside each DataLoader/preprocess worker."""
    global _TORCH_CLUSTER_RADIUS, _TORCH_CLUSTER_IMPORT_ATTEMPTED
    if not _TORCH_CLUSTER_IMPORT_ATTEMPTED:
        _TORCH_CLUSTER_IMPORT_ATTEMPTED = True
        try:
            from torch_cluster import radius

            _TORCH_CLUSTER_RADIUS = radius
        except (ImportError, OSError):
            _TORCH_CLUSTER_RADIUS = None
    return _TORCH_CLUSTER_RADIUS


def graph_backend_name() -> str:
    requested = os.environ.get("AIDEN_GRAPH_BACKEND", "auto").lower()
    if requested not in {"auto", "torch_cluster", "numpy"}:
        raise ValueError(
            "AIDEN_GRAPH_BACKEND must be auto, torch_cluster, or numpy"
        )
    if requested == "numpy":
        return "numpy-vectorized"
    available = _get_torch_cluster_radius() is not None
    if requested == "torch_cluster" and not available:
        raise RuntimeError(
            "AIDEN_GRAPH_BACKEND=torch_cluster was requested but "
            "torch-cluster could not be imported"
        )
    return "torch_cluster" if available else "numpy-vectorized"


def _periodic_images_np(
    atom_frac: np.ndarray,
    lattice: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    inverse_lattice = np.linalg.inv(lattice)
    reciprocal_column_norm = np.linalg.norm(inverse_lattice, axis=0)
    bounds = np.ceil(
        1.0 + float(cutoff) * reciprocal_column_norm
    ).astype(np.int64)
    ranges = [
        np.arange(-int(bound), int(bound) + 1, dtype=np.int64)
        for bound in bounds
    ]
    shift_grid = np.meshgrid(*ranges, indexing="ij")
    shifts = np.stack(
        [component.reshape(-1) for component in shift_grid],
        axis=-1,
    )
    num_atoms = int(atom_frac.shape[0])
    image_frac = (
        atom_frac[None, :, :] + shifts[:, None, :]
    ).reshape(-1, 3)
    image_atom = np.broadcast_to(
        np.arange(num_atoms, dtype=np.int64)[None, :],
        (shifts.shape[0], num_atoms),
    ).reshape(-1).copy()
    return image_frac, image_atom


def _rank_within_sorted_groups(index: torch.Tensor) -> torch.Tensor:
    if index.numel() == 0:
        return index
    starts = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=index.device),
            index[1:] != index[:-1],
        ]
    )
    group_ids = starts.cumsum(dim=0) - 1
    start_positions = torch.nonzero(starts, as_tuple=False).flatten()
    return (
        torch.arange(index.numel(), device=index.device)
        - start_positions[group_ids]
    )


def _periodic_radius_graph_torch_cluster_np(
    center_frac: np.ndarray,
    atom_frac: np.ndarray,
    lattice: np.ndarray,
    cutoff: float,
    max_neighbors: Optional[int],
    exclude_zero: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius = _get_torch_cluster_radius()
    if radius is None:
        raise RuntimeError("torch-cluster radius backend is unavailable")

    center_frac = np.asarray(center_frac, dtype=np.float32)
    atom_frac = np.asarray(atom_frac, dtype=np.float32)
    lattice = np.asarray(lattice, dtype=np.float32)
    image_frac, image_atom_np = _periodic_images_np(
        atom_frac.astype(np.float64),
        lattice.astype(np.float64),
        cutoff,
    )
    center_cart = torch.from_numpy(center_frac @ lattice)
    image_cart = torch.from_numpy(
        image_frac.astype(np.float32, copy=False) @ lattice
    )
    image_atom = torch.from_numpy(image_atom_np)

    requested = int(max_neighbors or 0)
    if requested > 0:
        # Over-request candidates, then perform an exact distance sort and
        # truncation per destination.
        search_neighbors = min(
            int(image_cart.shape[0]),
            max(5 * (requested + int(exclude_zero)), requested + 1),
        )
    else:
        search_neighbors = int(image_cart.shape[0])

    center_index, image_index = radius(
        image_cart,
        center_cart,
        float(cutoff),
        max_num_neighbors=max(search_neighbors, 1),
    )
    edge_vector = center_cart.index_select(
        0, center_index
    ) - image_cart.index_select(0, image_index)
    distance = torch.linalg.norm(edge_vector, dim=-1)
    valid = distance < float(cutoff)
    if exclude_zero:
        valid &= distance > 1.0e-8
    center_index = center_index[valid]
    image_index = image_index[valid]
    edge_vector = edge_vector[valid]
    distance = distance[valid]

    if center_index.numel():
        # Sort by destination first and distance second without a Python loop
        # per centre.
        sort_key = (
            center_index.to(torch.float64)
            + distance.to(torch.float64)
            / max(float(cutoff) * 2.0, 1.0e-12)
        )
        order = torch.argsort(sort_key, stable=True)
        center_index = center_index[order]
        image_index = image_index[order]
        edge_vector = edge_vector[order]
        distance = distance[order]
        if requested > 0:
            rank = _rank_within_sorted_groups(center_index)
            keep = rank < requested
            center_index = center_index[keep]
            image_index = image_index[keep]
            edge_vector = edge_vector[keep]
            distance = distance[keep]

    return (
        image_atom.index_select(0, image_index).numpy().astype(
            np.int64, copy=False
        ),
        center_index.numpy().astype(np.int64, copy=False),
        edge_vector.numpy().astype(np.float32, copy=False),
        distance.numpy().astype(np.float32, copy=False),
    )


def _periodic_radius_graph_vectorized_np(
    center_frac: np.ndarray,
    atom_frac: np.ndarray,
    lattice: np.ndarray,
    cutoff: float,
    max_neighbors: Optional[int],
    exclude_zero: bool,
    center_block_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dependency-free fallback, vectorized over blocks rather than centers."""
    center_frac = np.asarray(center_frac, dtype=np.float64)
    atom_frac = np.asarray(atom_frac, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)
    image_frac, image_atom = _periodic_images_np(
        atom_frac,
        lattice,
        cutoff,
    )
    requested = int(max_neighbors or 0)
    selection_width = (
        min(
            image_frac.shape[0],
            requested + int(exclude_zero),
        )
        if requested > 0
        else image_frac.shape[0]
    )

    atom_parts: List[np.ndarray] = []
    center_parts: List[np.ndarray] = []
    vector_parts: List[np.ndarray] = []
    distance_parts: List[np.ndarray] = []
    block_size = max(int(center_block_size), 1)
    for start in range(0, center_frac.shape[0], block_size):
        centers = center_frac[start : start + block_size]
        query_to_atom = (
            image_frac[None, :, :] - centers[:, None, :]
        ) @ lattice
        distance_all = np.linalg.norm(query_to_atom, axis=-1)
        order = np.argsort(distance_all, axis=1, kind="stable")
        order = order[:, :selection_width]
        selected_distance = np.take_along_axis(
            distance_all,
            order,
            axis=1,
        )
        selected_query_to_atom = np.take_along_axis(
            query_to_atom,
            order[..., None],
            axis=1,
        )
        valid = selected_distance < float(cutoff)
        if exclude_zero:
            valid &= selected_distance > 1.0e-8
        if requested > 0:
            valid &= np.cumsum(valid, axis=1) <= requested

        local_center = np.broadcast_to(
            np.arange(
                start,
                start + centers.shape[0],
                dtype=np.int64,
            )[:, None],
            order.shape,
        )
        atom_parts.append(image_atom[order][valid])
        center_parts.append(local_center[valid])
        vector_parts.append((-selected_query_to_atom)[valid])
        distance_parts.append(selected_distance[valid])

    return (
        np.concatenate(atom_parts).astype(np.int64, copy=False),
        np.concatenate(center_parts).astype(np.int64, copy=False),
        np.concatenate(vector_parts).astype(np.float32, copy=False),
        np.concatenate(distance_parts).astype(np.float32, copy=False),
    )


def _periodic_radius_graph_np(
    center_frac: np.ndarray,
    atom_frac: np.ndarray,
    lattice: np.ndarray,
    cutoff: float,
    max_neighbors: Optional[int],
    exclude_zero: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    backend = graph_backend_name()
    if backend == "torch_cluster":
        return _periodic_radius_graph_torch_cluster_np(
            center_frac,
            atom_frac,
            lattice,
            cutoff,
            max_neighbors,
            exclude_zero,
        )
    return _periodic_radius_graph_vectorized_np(
        center_frac,
        atom_frac,
        lattice,
        cutoff,
        max_neighbors,
        exclude_zero,
    )


def build_atom_graph_np(
    atom_frac: np.ndarray,
    lattice: np.ndarray,
    cutoff: float,
    exclude_self: bool = True,
    max_neighbors: Optional[int] = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic atom graph with compiled radius search."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    return _periodic_radius_graph_np(
        np.asarray(atom_frac),
        np.asarray(atom_frac),
        np.asarray(lattice),
        cutoff,
        max_neighbors,
        exclude_self,
    )


def build_probe_graph_np(
    atom_frac: np.ndarray,
    lattice: np.ndarray,
    query_frac: np.ndarray,
    cutoff: float,
    max_neighbors: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic atom->probe graph using the compiled radius-search path."""
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    return _periodic_radius_graph_np(
        np.asarray(query_frac),
        np.asarray(atom_frac),
        np.asarray(lattice),
        cutoff,
        max_neighbors,
        False,
    )


def collate_rho(batch: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    return list(batch)


class CHGCARRhoDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        elem_csv: str | Path,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        num_queries: int = 2048,
        cache_dir: Optional[str | Path] = None,
        file_list_json: Optional[str | Path] = None,
        max_files: Optional[int] = None,
        fixed_queries: bool = True,
        atom_cutoff: float = 5.0,
        max_atom_neighbors: int = 100,
        query_cutoff: float = 6.0,
        max_probe_neighbors: int = 300,
        in_memory_cache_size: int = 0,
        use_sample_probs: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = str(split)
        self.seed = int(seed)
        self.num_queries = int(num_queries)
        self.fixed_queries = bool(fixed_queries)
        self.atom_cutoff = float(atom_cutoff)
        self.max_atom_neighbors = int(max_atom_neighbors)
        self.query_cutoff = float(query_cutoff)
        self.max_probe_neighbors = int(max_probe_neighbors)
        self.atom_graph_version = (
            f"periodic_radius_v2_cutoff={self.atom_cutoff:.8g}"
            f"_max_neighbors={self.max_atom_neighbors}"
        )
        self.use_sample_probs = bool(use_sample_probs)
        self.in_memory_cache_size = max(int(in_memory_cache_size), 0)
        self._memory_cache: "OrderedDict[int, ParsedChgcar]" = OrderedDict()
        self._stochastic_query_rng: Optional[np.random.Generator] = None
        self._packed_handles: Dict[Path, object] = {}
        self._packed_handle_pid = os.getpid()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        (
            self.symbol_to_z,
            self.symbol_to_feat,
            self.z_to_feat,
            self.feature_names,
            self.element_order,
        ) = load_element_descriptors(elem_csv)
        self.feature_dim = len(self.feature_names)

        all_files = self._discover_files(file_list_json=file_list_json)
        if max_files is not None:
            all_files = all_files[:max_files]

        if not all_files:
            raise FileNotFoundError(
                f"No input files found under {self.data_root}. "
                f"Supported: .npz, .chgcar, .chgcar.gz, .chgcar.tar.gz, .vasp, .vasp.gz, .vasp.tar.gz"
            )

        self.files = self._make_split(all_files, split_ratio, split)
        if not self.files:
            raise RuntimeError(f"Split {split!r} is empty under {self.data_root}")

        # Build deterministic validation indices lazily.  Eagerly opening all
        # 14k validation records delayed every launch and left inherited HDF5
        # handles around before DataLoader workers were created.  The index RNG
        # is a pure function of split/seed/record, so lazy construction remains
        # exactly reproducible in every worker.  ``None`` is the all-points
        # sentinel for records that expose no more points than requested.
        self._fixed_query_indices: Optional[Dict[int, Optional[np.ndarray]]] = (
            {} if self.fixed_queries else None
        )

    @property
    def storage_backend(self) -> str:
        return (
            "packed_hdf5"
            if self.files and isinstance(self.files[0], PackedH5Record)
            else "individual_files"
        )

    def _discover_files(
        self,
        file_list_json: Optional[str | Path],
    ) -> List[InputRecord]:
        manifest_path = self.data_root / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("schema") == PACKED_H5_SCHEMA:
                if file_list_json is not None:
                    raise ValueError(
                        "--file-list-json is not supported with packed HDF5 data"
                    )
                records: List[InputRecord] = []
                for shard in payload["shards"]:
                    shard_path = self.data_root / str(shard["file"])
                    start = int(shard["start"])
                    count = int(shard["count"])
                    records.extend(
                        PackedH5Record(
                            shard_path=shard_path,
                            group_key=f"{index:08d}",
                            order=index,
                        )
                        for index in range(start, start + count)
                    )
                expected = int(payload["num_structures"])
                if len(records) != expected:
                    raise RuntimeError(
                        f"Packed manifest declares {expected} structures but "
                        f"indexes {len(records)}"
                    )
                return records

        if file_list_json is not None:
            payload = json.loads(Path(file_list_json).read_text(encoding="utf-8"))
            files = [Path(x) for x in payload]
            return sorted(set(files))

        patterns = [
            "*.npz",
            "*.chgcar",
            "*.chgcar.gz",
            "*.chgcar.tar.gz",
            "*.vasp",
            "*.vasp.gz",
            "*.vasp.tar.gz",
        ]
        # The MP cache is a flat directory with ~150k entries.  Avoid a
        # recursive filesystem walk (expensive on scratch metadata servers)
        # when the supported files are already present at the root.
        top_level: List[Path] = []
        for pattern in patterns:
            top_level.extend(self.data_root.glob(pattern))
        if top_level:
            return sorted(set(top_level))

        found: List[Path] = []
        for pattern in patterns:
            found.extend(self.data_root.rglob(pattern))
        return sorted(set(found))

    def _make_split(
        self,
        all_files: Sequence[InputRecord],
        split_ratio: Tuple[float, float, float],
        split: str,
    ) -> List[InputRecord]:
        ratios = np.asarray(split_ratio, dtype=np.float64)
        if np.any(ratios < 0):
            raise ValueError("split_ratio must be non-negative")
        if ratios.sum() <= 0:
            raise ValueError("split_ratio sum must be positive")

        ratios = ratios / ratios.sum()
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(all_files))
        n = len(all_files)

        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        idx_train = perm[:n_train]
        idx_val = perm[n_train:n_train + n_val]
        idx_test = perm[n_train + n_val:]

        split_map = {
            "train": idx_train,
            "val": idx_val,
            "valid": idx_val,
            "test": idx_test,
            "all": perm,
        }

        key = split.lower()
        if key not in split_map:
            raise ValueError(f"Unknown split: {split}")

        return [all_files[i] for i in split_map[key]]

    def __len__(self) -> int:
        return len(self.files)

    def load_density_values(self, idx: int) -> np.ndarray:
        """Load only the flattened density array for one input record.

        Dataset records are either ordinary filesystem paths or
        :class:`PackedH5Record` entries.  Training-density statistics should
        not need to know which representation is in use, nor materialize atom
        features and graphs from an HDF5 group just to read its target.
        """

        record = self.files[idx]
        if isinstance(record, PackedH5Record):
            group = self._packed_handle(record.shard_path)[record.group_key]
            return np.asarray(group["density"], dtype=np.float32).reshape(-1)
        if record.suffix == ".npz":
            with np.load(record, allow_pickle=False) as payload:
                return np.asarray(
                    payload["density"],
                    dtype=np.float32,
                ).reshape(-1)
        return np.asarray(
            self._load_structure(idx).density,
            dtype=np.float32,
        ).reshape(-1)

    def _cache_path(self, file_path: Path) -> Path:
        assert self.cache_dir is not None
        safe_name = file_path.name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_name}.npz"

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        split_offsets = {
            "train": 0,
            "val": 1,
            "valid": 1,
            "test": 2,
            "all": 3,
        }
        offset = split_offsets.get(self.split.lower(), 0)
        return np.random.default_rng(self.seed + 1000003 * (idx + 1) + 10000019 * offset)

    def _get_stochastic_query_rng(self) -> np.random.Generator:
        """One reproducibly seeded NumPy generator per DataLoader worker."""
        if self._stochastic_query_rng is None:
            worker = torch.utils.data.get_worker_info()
            worker_seed = (
                int(worker.seed)
                if worker is not None
                else int(self.seed + 10000019)
            )
            self._stochastic_query_rng = np.random.default_rng(worker_seed)
        return self._stochastic_query_rng

    def _build_atom_features_from_atomic_numbers(self, atomic_numbers: np.ndarray) -> np.ndarray:
        feats = []
        for z in atomic_numbers.astype(np.int64).tolist():
            if z not in self.z_to_feat:
                raise KeyError(f"Atomic number {z} not found in elem-des.csv")
            feats.append(self.z_to_feat[z])
        return np.asarray(feats, dtype=np.float32)

    def _resolve_atom_features(
        self,
        atomic_numbers: np.ndarray,
        loaded_atom_features: Optional[np.ndarray],
        schema_version: Optional[str],
    ) -> np.ndarray:
        if (
            loaded_atom_features is not None
            and loaded_atom_features.ndim == 2
            and loaded_atom_features.shape[0] == atomic_numbers.shape[0]
            and loaded_atom_features.shape[1] == self.feature_dim
            and schema_version == FEATURE_SCHEMA_VERSION
        ):
            return loaded_atom_features.astype(np.float32, copy=False)

        return self._build_atom_features_from_atomic_numbers(atomic_numbers)


    def _touch_memory_cache(self, idx: int, parsed: ParsedChgcar) -> None:
        if self.in_memory_cache_size <= 0:
            return
        self._memory_cache[idx] = parsed
        self._memory_cache.move_to_end(idx)
        while len(self._memory_cache) > self.in_memory_cache_size:
            self._memory_cache.popitem(last=False)

    def _get_memory_cache(self, idx: int) -> Optional[ParsedChgcar]:
        if self.in_memory_cache_size <= 0:
            return None
        parsed = self._memory_cache.get(idx)
        if parsed is not None:
            self._memory_cache.move_to_end(idx)
        return parsed

    def _ensure_atom_graph(self, parsed: ParsedChgcar) -> ParsedChgcar:
        need_graph = (
            parsed.atom_edge_src is None
            or parsed.atom_edge_dst is None
            or parsed.atom_edge_vec is None
            or parsed.atom_edge_dist is None
            or parsed.atom_graph_version != self.atom_graph_version
        )
        if not need_graph:
            return parsed

        src, dst, edge_vec, edge_dist = build_atom_graph_np(
            parsed.frac_coords.astype(np.float32, copy=False),
            parsed.lattice.astype(np.float32, copy=False),
            cutoff=self.atom_cutoff,
            exclude_self=True,
            max_neighbors=self.max_atom_neighbors,
        )
        parsed.atom_edge_src = src
        parsed.atom_edge_dst = dst
        parsed.atom_edge_vec = edge_vec
        parsed.atom_edge_dist = edge_dist
        parsed.atom_graph_version = self.atom_graph_version
        return parsed

    def close_storage_handles(self) -> None:
        handles = getattr(self, "_packed_handles", {})
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass
        handles.clear()
        getpid = getattr(os, "getpid", None)
        self._packed_handle_pid = getpid() if getpid is not None else -1

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_packed_handles"] = {}
        state["_packed_handle_pid"] = os.getpid()
        return state

    def __del__(self):
        # Module globals (including ``os`` and h5py internals) may already be
        # cleared during interpreter shutdown.  A destructor must never turn
        # successful training into a noisy "Exception ignored in ..." message.
        try:
            self.close_storage_handles()
        except Exception:
            pass

    def _packed_handle(self, shard_path: Path):
        current_pid = os.getpid()
        if current_pid != self._packed_handle_pid:
            self.close_storage_handles()
            self._packed_handle_pid = current_pid
        handle = self._packed_handles.get(shard_path)
        if handle is None:
            import h5py

            handle = h5py.File(
                shard_path,
                mode="r",
                libver="latest",
                swmr=True,
            )
            self._packed_handles[shard_path] = handle
        return handle

    def _load_packed_structure(
        self,
        record: PackedH5Record,
    ) -> ParsedChgcar:
        group = self._packed_handle(record.shard_path)[record.group_key]

        atomic_numbers = np.asarray(
            group["atomic_numbers"],
            dtype=np.int64,
        )
        schema_version = str(
            group.attrs.get("feature_schema_version", "")
        )
        loaded_atom_features = (
            np.asarray(group["atom_features"], dtype=np.float32)
            if "atom_features" in group
            else None
        )
        atom_features = self._resolve_atom_features(
            atomic_numbers,
            loaded_atom_features,
            schema_version,
        )

        def optional_array(key: str, dtype) -> Optional[np.ndarray]:
            if key not in group:
                return None
            value = np.asarray(group[key], dtype=dtype)
            return value if value.size else None

        return ParsedChgcar(
            name=str(group.attrs.get("name", record.group_key)),
            lattice=np.asarray(group["lattice"], dtype=np.float32),
            species=json.loads(str(group.attrs.get("species_json", "[]"))),
            counts=np.asarray(group["counts"], dtype=np.int64).tolist(),
            frac_coords=np.asarray(
                group["frac_coords"],
                dtype=np.float32,
            ),
            atomic_numbers=atomic_numbers,
            atom_features=atom_features,
            grid_shape=tuple(
                np.asarray(group["grid_shape"], dtype=np.int64).tolist()
            ),
            density=np.asarray(group["density"], dtype=np.float32),
            volume=float(np.asarray(group["volume"])),
            atom_cart=optional_array("atom_cart", np.float32),
            all_query_frac=optional_array(
                "all_query_frac",
                np.float32,
            ),
            all_query_cart=optional_array(
                "all_query_cart",
                np.float32,
            ),
            sample_probs=optional_array("sample_probs", np.float32),
            atom_edge_src=optional_array("atom_edge_src", np.int64),
            atom_edge_dst=optional_array("atom_edge_dst", np.int64),
            atom_edge_vec=optional_array("atom_edge_vec", np.float32),
            atom_edge_dist=optional_array("atom_edge_dist", np.float32),
            atom_graph_version=str(
                group.attrs.get("atom_graph_version", "")
            ),
        )

    def _load_preprocessed_structure(self, file_path: Path) -> ParsedChgcar:
        with np.load(file_path, allow_pickle=False) as payload:
            species = json.loads(str(payload["species_json"])) if "species_json" in payload else []
            counts = payload["counts"].astype(np.int64).tolist() if "counts" in payload else []
            atomic_numbers = payload["atomic_numbers"].astype(np.int64)
            schema_version = str(payload["feature_schema_version"]) if "feature_schema_version" in payload else None
            loaded_atom_features = payload["atom_features"].astype(np.float32) if "atom_features" in payload else None
            atom_features = self._resolve_atom_features(atomic_numbers, loaded_atom_features, schema_version)

            atom_cart = payload["atom_cart"].astype(np.float32) if "atom_cart" in payload else None
            all_query_frac = payload["all_query_frac"].astype(np.float32) if "all_query_frac" in payload else None
            all_query_cart = payload["all_query_cart"].astype(np.float32) if "all_query_cart" in payload else None
            sample_probs = payload["sample_probs"].astype(np.float32) if "sample_probs" in payload else None
            atom_edge_src = payload["atom_edge_src"].astype(np.int64) if "atom_edge_src" in payload else None
            atom_edge_dst = payload["atom_edge_dst"].astype(np.int64) if "atom_edge_dst" in payload else None
            atom_edge_vec = payload["atom_edge_vec"].astype(np.float32) if "atom_edge_vec" in payload else None
            atom_edge_dist = payload["atom_edge_dist"].astype(np.float32) if "atom_edge_dist" in payload else None
            atom_graph_version = (
                str(payload["atom_graph_version"])
                if "atom_graph_version" in payload
                else None
            )

            return ParsedChgcar(
                name=str(payload["name"]) if "name" in payload else file_path.name,
                lattice=payload["lattice"].astype(np.float32),
                species=species,
                counts=counts,
                frac_coords=payload["frac_coords"].astype(np.float32),
                atomic_numbers=atomic_numbers,
                atom_features=atom_features,
                grid_shape=tuple(payload["grid_shape"].tolist()),
                density=payload["density"].astype(np.float32),
                volume=float(payload["volume"]),
                atom_cart=atom_cart,
                all_query_frac=all_query_frac,
                all_query_cart=all_query_cart,
                sample_probs=sample_probs,
                atom_edge_src=atom_edge_src,
                atom_edge_dst=atom_edge_dst,
                atom_edge_vec=atom_edge_vec,
                atom_edge_dist=atom_edge_dist,
                atom_graph_version=atom_graph_version,
            )

    def _load_cached_structure(self, cache_path: Path, source_name: str) -> ParsedChgcar:
        with np.load(cache_path, allow_pickle=False) as payload:
            atomic_numbers = payload["atomic_numbers"].astype(np.int64)
            schema_version = str(payload["feature_schema_version"]) if "feature_schema_version" in payload else None
            loaded_atom_features = payload["atom_features"].astype(np.float32) if "atom_features" in payload else None
            atom_features = self._resolve_atom_features(atomic_numbers, loaded_atom_features, schema_version)

            atom_cart = payload["atom_cart"].astype(np.float32) if "atom_cart" in payload else None
            all_query_frac = payload["all_query_frac"].astype(np.float32) if "all_query_frac" in payload else None
            all_query_cart = payload["all_query_cart"].astype(np.float32) if "all_query_cart" in payload else None
            sample_probs = payload["sample_probs"].astype(np.float32) if "sample_probs" in payload else None
            atom_edge_src = payload["atom_edge_src"].astype(np.int64) if "atom_edge_src" in payload else None
            atom_edge_dst = payload["atom_edge_dst"].astype(np.int64) if "atom_edge_dst" in payload else None
            atom_edge_vec = payload["atom_edge_vec"].astype(np.float32) if "atom_edge_vec" in payload else None
            atom_edge_dist = payload["atom_edge_dist"].astype(np.float32) if "atom_edge_dist" in payload else None
            atom_graph_version = (
                str(payload["atom_graph_version"])
                if "atom_graph_version" in payload
                else None
            )

            return ParsedChgcar(
                name=str(payload["name"]) if "name" in payload else source_name,
                lattice=payload["lattice"].astype(np.float32),
                species=json.loads(str(payload["species_json"])),
                counts=payload["counts"].astype(np.int64).tolist(),
                frac_coords=payload["frac_coords"].astype(np.float32),
                atomic_numbers=atomic_numbers,
                atom_features=atom_features,
                grid_shape=tuple(payload["grid_shape"].tolist()),
                density=payload["density"].astype(np.float32),
                volume=float(payload["volume"]),
                atom_cart=atom_cart,
                all_query_frac=all_query_frac,
                all_query_cart=all_query_cart,
                sample_probs=sample_probs,
                atom_edge_src=atom_edge_src,
                atom_edge_dst=atom_edge_dst,
                atom_edge_vec=atom_edge_vec,
                atom_edge_dist=atom_edge_dist,
                atom_graph_version=atom_graph_version,
            )

    def _save_cache(self, cache_path: Path, parsed: ParsedChgcar) -> None:
        np.savez_compressed(
            cache_path,
            name=np.asarray(parsed.name),
            lattice=parsed.lattice.astype(np.float32),
            species_json=np.asarray(json.dumps(parsed.species)),
            counts=np.asarray(parsed.counts, dtype=np.int64),
            frac_coords=parsed.frac_coords.astype(np.float32),
            atomic_numbers=parsed.atomic_numbers.astype(np.int64),
            atom_features=parsed.atom_features.astype(np.float32),
            grid_shape=np.asarray(parsed.grid_shape, dtype=np.int64),
            density=parsed.density.astype(np.float32),
            volume=np.asarray(parsed.volume, dtype=np.float32),
            atom_cart=parsed.atom_cart.astype(np.float32) if parsed.atom_cart is not None else np.empty((0, 3), dtype=np.float32),
            all_query_frac=parsed.all_query_frac.astype(np.float32) if parsed.all_query_frac is not None else np.empty((0, 3), dtype=np.float32),
            all_query_cart=parsed.all_query_cart.astype(np.float32) if parsed.all_query_cart is not None else np.empty((0, 3), dtype=np.float32),
            sample_probs=parsed.sample_probs.astype(np.float32) if parsed.sample_probs is not None else np.empty((0,), dtype=np.float32),
            atom_edge_src=parsed.atom_edge_src.astype(np.int64) if parsed.atom_edge_src is not None else np.empty((0,), dtype=np.int64),
            atom_edge_dst=parsed.atom_edge_dst.astype(np.int64) if parsed.atom_edge_dst is not None else np.empty((0,), dtype=np.int64),
            atom_edge_vec=parsed.atom_edge_vec.astype(np.float32) if parsed.atom_edge_vec is not None else np.empty((0, 3), dtype=np.float32),
            atom_edge_dist=parsed.atom_edge_dist.astype(np.float32) if parsed.atom_edge_dist is not None else np.empty((0,), dtype=np.float32),
            atom_graph_version=np.asarray(parsed.atom_graph_version or ""),
            feature_schema_version=np.asarray(FEATURE_SCHEMA_VERSION),
        )

    def _load_structure(self, idx: int) -> ParsedChgcar:
        cached = self._get_memory_cache(idx)
        if cached is not None:
            return cached

        file_path = self.files[idx]

        if isinstance(file_path, PackedH5Record):
            parsed = self._load_packed_structure(file_path)
            parsed = self._ensure_atom_graph(parsed)
            self._touch_memory_cache(idx, parsed)
            return parsed

        if file_path.suffix == ".npz":
            parsed = self._load_preprocessed_structure(file_path)
            if parsed.atom_cart is not None and parsed.atom_cart.size == 0:
                parsed.atom_cart = None
            if parsed.all_query_frac is not None and parsed.all_query_frac.size == 0:
                parsed.all_query_frac = None
            if parsed.all_query_cart is not None and parsed.all_query_cart.size == 0:
                parsed.all_query_cart = None
            if parsed.sample_probs is not None and parsed.sample_probs.size == 0:
                parsed.sample_probs = None
            if parsed.atom_edge_src is not None and parsed.atom_edge_src.size == 0:
                parsed.atom_edge_src = None
            if parsed.atom_edge_dst is not None and parsed.atom_edge_dst.size == 0:
                parsed.atom_edge_dst = None
            if parsed.atom_edge_vec is not None and parsed.atom_edge_vec.size == 0:
                parsed.atom_edge_vec = None
            if parsed.atom_edge_dist is not None and parsed.atom_edge_dist.size == 0:
                parsed.atom_edge_dist = None
            parsed = self._ensure_atom_graph(parsed)
            self._touch_memory_cache(idx, parsed)
            return parsed

        if self.cache_dir is not None:
            cache_path = self._cache_path(file_path)
            if cache_path.exists():
                try:
                    parsed = self._load_cached_structure(cache_path, file_path.name)
                    if parsed.atom_cart is not None and parsed.atom_cart.size == 0:
                        parsed.atom_cart = None
                    if parsed.all_query_frac is not None and parsed.all_query_frac.size == 0:
                        parsed.all_query_frac = None
                    if parsed.all_query_cart is not None and parsed.all_query_cart.size == 0:
                        parsed.all_query_cart = None
                    if parsed.sample_probs is not None and parsed.sample_probs.size == 0:
                        parsed.sample_probs = None
                    if parsed.atom_edge_src is not None and parsed.atom_edge_src.size == 0:
                        parsed.atom_edge_src = None
                    if parsed.atom_edge_dst is not None and parsed.atom_edge_dst.size == 0:
                        parsed.atom_edge_dst = None
                    if parsed.atom_edge_vec is not None and parsed.atom_edge_vec.size == 0:
                        parsed.atom_edge_vec = None
                    if parsed.atom_edge_dist is not None and parsed.atom_edge_dist.size == 0:
                        parsed.atom_edge_dist = None
                    parsed = self._ensure_atom_graph(parsed)
                    self._touch_memory_cache(idx, parsed)
                    return parsed
                except Exception:
                    pass

        parsed = parse_chgcar_file(file_path, self.symbol_to_z, self.symbol_to_feat)
        parsed = self._ensure_atom_graph(parsed)

        if self.cache_dir is not None:
            cache_path = self._cache_path(file_path)
            self._save_cache(cache_path, parsed)

        self._touch_memory_cache(idx, parsed)
        return parsed

    def _build_fixed_query_indices(self, idx: int) -> Optional[np.ndarray]:
        rng = self._rng_for_index(idx)
        file_path = self.files[idx]
        # Validation/test query selection only needs density length and optional
        # sampling probabilities. Avoid building an exact periodic atom graph
        # for every structure during dataset construction.
        if isinstance(file_path, PackedH5Record):
            group = self._packed_handle(file_path.shard_path)[
                file_path.group_key
            ]
            n_total = int(group["density"].shape[0])
            sample_probs = (
                np.asarray(group["sample_probs"], dtype=np.float32)
                if "sample_probs" in group
                else None
            )
            if self.num_queries >= n_total:
                return None
            return self._sample_query_indices_from_values(
                n_total,
                sample_probs,
                rng=rng,
            )
        if file_path.suffix == ".npz":
            with np.load(file_path, allow_pickle=False) as payload:
                n_total = int(payload["density"].size)
                sample_probs = (
                    payload["sample_probs"].astype(np.float32)
                    if "sample_probs" in payload
                    else None
                )
            if self.num_queries >= n_total:
                return None
            return self._sample_query_indices_from_values(
                n_total,
                sample_probs,
                rng=rng,
            )
        parsed = self._load_structure(idx)
        if self.num_queries >= int(parsed.density.shape[0]):
            return None
        return self._sample_query_indices(parsed, rng=rng)

    def _sample_query_indices_from_values(
        self,
        n_total: int,
        sample_probs: Optional[np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        q = min(self.num_queries, int(n_total))

        if q >= n_total:
            return np.arange(n_total, dtype=np.int64)

        valid_probabilities = (
            self.use_sample_probs
            and sample_probs is not None
            and len(sample_probs) == n_total
        )
        if valid_probabilities:
            probs = np.asarray(sample_probs, dtype=np.float64)
            probs = probs / np.clip(probs.sum(), 1e-12, None)
        else:
            probs = None

        # deterministic branch for fixed val/test queries
        if rng is not None:
            sampled = rng.choice(
                n_total,
                size=q,
                replace=False,
                p=probs,
            )
            return sampled.astype(np.int64)

        # stochastic branch for train-time resampling
        sampled = self._get_stochastic_query_rng().choice(
            n_total,
            size=q,
            replace=False,
            p=probs,
        )
        return sampled.astype(np.int64)

    def _sample_query_indices(
        self,
        parsed: ParsedChgcar,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        return self._sample_query_indices_from_values(
            int(parsed.density.shape[0]),
            parsed.sample_probs,
            rng=rng,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        parsed = self._load_structure(idx)

        if self._fixed_query_indices is not None:
            if idx not in self._fixed_query_indices:
                self._fixed_query_indices[idx] = self._build_fixed_query_indices(idx)
            fixed_sampled = self._fixed_query_indices[idx]
            sampled: np.ndarray | slice = (
                slice(None) if fixed_sampled is None else fixed_sampled
            )
        else:
            # The 5,000-probe training path consumes the complete cached set.
            # Use a slice rather than allocating/fancy-indexing through an
            # identical arange array on every structure load.
            sampled = (
                slice(None)
                if self.num_queries >= int(parsed.density.shape[0])
                else self._sample_query_indices(parsed, rng=None)
            )

        if parsed.all_query_frac is not None:
            query_frac = parsed.all_query_frac[sampled]
        else:
            # ``slice(None)`` is the allocation-free fast path for stored
            # coordinates/densities, but the coordinate generator needs
            # explicit integer grid indices.
            grid_indices = (
                np.arange(parsed.density.shape[0], dtype=np.int64)
                if isinstance(sampled, slice)
                else sampled
            )
            query_frac = flat_indices_to_frac(grid_indices, parsed.grid_shape)

        if parsed.all_query_cart is not None:
            query_cart = parsed.all_query_cart[sampled]
        else:
            query_cart = query_frac @ parsed.lattice

        target = parsed.density[sampled].astype(np.float32)
        (
            probe_edge_atom,
            probe_edge_query,
            probe_edge_vec,
            probe_edge_dist,
        ) = build_probe_graph_np(
            parsed.frac_coords,
            parsed.lattice,
            query_frac,
            cutoff=self.query_cutoff,
            max_neighbors=self.max_probe_neighbors,
        )

        if parsed.atom_cart is not None:
            atom_cart = parsed.atom_cart.astype(np.float32)
        else:
            atom_cart = (parsed.frac_coords @ parsed.lattice).astype(np.float32)

        out = {
            "name": parsed.name,
            "lattice": torch.from_numpy(parsed.lattice.astype(np.float32)),
            "atom_frac": torch.from_numpy(parsed.frac_coords.astype(np.float32)),
            "atom_cart": torch.from_numpy(atom_cart),
            "atom_z": torch.from_numpy(parsed.atomic_numbers.astype(np.int64)),
            "atom_features": torch.from_numpy(parsed.atom_features.astype(np.float32)),
            "query_frac": torch.from_numpy(query_frac.astype(np.float32)),
            "query_cart": torch.from_numpy(query_cart.astype(np.float32)),
            "target": torch.from_numpy(target),
            "grid_shape": torch.tensor(parsed.grid_shape, dtype=torch.long),
            "volume": torch.tensor(parsed.volume, dtype=torch.float32),
            "probe_edge_atom": torch.from_numpy(probe_edge_atom),
            "probe_edge_query": torch.from_numpy(probe_edge_query),
            "probe_edge_vec": torch.from_numpy(probe_edge_vec),
            "probe_edge_dist": torch.from_numpy(probe_edge_dist),
        }
        if parsed.atom_edge_src is not None:
            out["atom_edge_src"] = torch.from_numpy(parsed.atom_edge_src.astype(np.int64, copy=False))
            out["atom_edge_dst"] = torch.from_numpy(parsed.atom_edge_dst.astype(np.int64, copy=False))
            out["atom_edge_vec"] = torch.from_numpy(parsed.atom_edge_vec.astype(np.float32, copy=False))
            out["atom_edge_dist"] = torch.from_numpy(parsed.atom_edge_dist.astype(np.float32, copy=False))
        return out
