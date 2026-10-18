from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

from .data import (
    FEATURE_SCHEMA_VERSION,
    PACKED_H5_SCHEMA,
    build_atom_graph_np,
    graph_backend_name,
)


def atom_graph_version(cutoff: float, max_neighbors: int) -> str:
    return (
        f"periodic_radius_v2_cutoff={float(cutoff):.8g}"
        f"_max_neighbors={int(max_neighbors)}"
    )


def _valid_shard(
    path: Path,
    start: int,
    count: int,
    graph_version: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, mode="r", swmr=True) as handle:
            return (
                bool(handle.attrs.get("complete", False))
                and str(handle.attrs.get("schema", "")) == PACKED_H5_SCHEMA
                and int(handle.attrs.get("start", -1)) == start
                and int(handle.attrs.get("count", -1)) == count
                and str(handle.attrs.get("atom_graph_version", ""))
                == graph_version
                and len(handle) == count
            )
    except (OSError, KeyError, ValueError):
        return False


def _copy_dataset(
    group: h5py.Group,
    payload,
    key: str,
    dtype,
    *,
    required: bool = False,
) -> None:
    if key not in payload:
        if required:
            raise KeyError(f"required array {key!r} is missing")
        return
    value = np.asarray(payload[key], dtype=dtype)
    if value.size:
        group.create_dataset(key, data=value, compression=None)


def _write_shard(
    task: Tuple[
        int,
        Sequence[str],
        str,
        float,
        int,
    ],
) -> Dict[str, object]:
    start, source_paths, output_path_text, cutoff, max_neighbors = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    output_path = Path(output_path_text)
    count = len(source_paths)
    graph_version = atom_graph_version(cutoff, max_neighbors)
    if _valid_shard(output_path, start, count, graph_version):
        return {
            "file": output_path.name,
            "start": start,
            "count": count,
            "reused": True,
        }

    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, mode="w", libver="latest") as handle:
        handle.attrs["schema"] = PACKED_H5_SCHEMA
        handle.attrs["start"] = start
        handle.attrs["count"] = count
        handle.attrs["atom_graph_version"] = graph_version
        handle.attrs["complete"] = False

        for local_index, source_path_text in enumerate(source_paths):
            global_index = start + local_index
            source_path = Path(source_path_text)
            with np.load(source_path, allow_pickle=False) as payload:
                lattice = np.asarray(payload["lattice"], dtype=np.float32)
                frac_coords = np.asarray(
                    payload["frac_coords"],
                    dtype=np.float32,
                )
                (
                    atom_edge_src,
                    atom_edge_dst,
                    atom_edge_vec,
                    atom_edge_dist,
                ) = build_atom_graph_np(
                    frac_coords,
                    lattice,
                    cutoff=cutoff,
                    exclude_self=True,
                    max_neighbors=max_neighbors,
                )

                group = handle.create_group(f"{global_index:08d}")
                group.attrs["name"] = str(
                    payload["name"]
                    if "name" in payload
                    else source_path.name
                )
                group.attrs["species_json"] = str(
                    payload["species_json"]
                    if "species_json" in payload
                    else "[]"
                )
                group.attrs["feature_schema_version"] = str(
                    payload["feature_schema_version"]
                    if "feature_schema_version" in payload
                    else FEATURE_SCHEMA_VERSION
                )
                group.attrs["atom_graph_version"] = graph_version
                group.attrs["source_file"] = source_path.name

                _copy_dataset(
                    group,
                    payload,
                    "lattice",
                    np.float32,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "counts",
                    np.int64,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "frac_coords",
                    np.float32,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "atomic_numbers",
                    np.int64,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "atom_features",
                    np.float32,
                )
                _copy_dataset(
                    group,
                    payload,
                    "grid_shape",
                    np.int64,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "density",
                    np.float32,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "volume",
                    np.float32,
                    required=True,
                )
                _copy_dataset(
                    group,
                    payload,
                    "atom_cart",
                    np.float32,
                )
                _copy_dataset(
                    group,
                    payload,
                    "all_query_frac",
                    np.float32,
                )
                _copy_dataset(
                    group,
                    payload,
                    "all_query_cart",
                    np.float32,
                )
                _copy_dataset(
                    group,
                    payload,
                    "sample_probs",
                    np.float32,
                )
                group.create_dataset(
                    "atom_edge_src",
                    data=atom_edge_src,
                    compression=None,
                )
                group.create_dataset(
                    "atom_edge_dst",
                    data=atom_edge_dst,
                    compression=None,
                )
                group.create_dataset(
                    "atom_edge_vec",
                    data=atom_edge_vec,
                    compression=None,
                )
                group.create_dataset(
                    "atom_edge_dist",
                    data=atom_edge_dist,
                    compression=None,
                )

        handle.attrs.modify("complete", True)
        handle.flush()
    os.replace(temporary, output_path)
    return {
        "file": output_path.name,
        "start": start,
        "count": count,
        "reused": False,
    }


def _manifest_is_complete(
    manifest_path: Path,
    source_root: Path,
    cutoff: float,
    max_neighbors: int,
    max_files: int | None,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != PACKED_H5_SCHEMA
            or payload.get("source_root") != str(source_root.resolve())
            or float(payload.get("atom_cutoff")) != float(cutoff)
            or int(payload.get("max_atom_neighbors")) != int(max_neighbors)
            or payload.get("max_files") != max_files
        ):
            return False
        graph_version = atom_graph_version(cutoff, max_neighbors)
        shards = payload["shards"]
        if (
            sum(int(shard["count"]) for shard in shards)
            != int(payload["num_structures"])
        ):
            return False
        return all(
            _valid_shard(
                manifest_path.parent / shard["file"],
                int(shard["start"]),
                int(shard["count"]),
                graph_version,
            )
            for shard in shards
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack individual density NPZ files into HDF5 shards "
            "and persist exact periodic atom graphs."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--atom-cutoff", type=float, default=4.0)
    parser.add_argument("--max-atom-neighbors", type=int, default=100)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    if _manifest_is_complete(
        manifest_path,
        input_root,
        args.atom_cutoff,
        args.max_atom_neighbors,
        args.max_files,
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            "Packed dataset ready "
            f"| structures={payload['num_structures']} "
            f"shards={len(payload['shards'])} "
            f"backend={graph_backend_name()} "
            f"root={output_root}",
            flush=True,
        )
        return

    # This cache is flat; the non-recursive fast path avoids a costly metadata
    # traversal across ~150k scratch files.  Keep recursive discovery as a
    # fallback so the packer remains useful for nested caches.
    source_files = sorted(input_root.glob("*.npz"))
    if not source_files:
        source_files = sorted(input_root.rglob("*.npz"))
    if args.max_files is not None:
        source_files = source_files[: args.max_files]
    if not source_files:
        raise FileNotFoundError(f"No NPZ files found below {input_root}")
    if args.shard_size <= 0 or args.num_workers <= 0:
        raise ValueError("--shard-size and --num-workers must be positive")

    tasks = []
    for start in range(0, len(source_files), args.shard_size):
        paths = source_files[start : start + args.shard_size]
        shard_index = start // args.shard_size
        output_path = output_root / f"shard-{shard_index:05d}.h5"
        tasks.append(
            (
                start,
                [str(path) for path in paths],
                str(output_path),
                float(args.atom_cutoff),
                int(args.max_atom_neighbors),
            )
        )

    print(
        "Packing Aiden data "
        f"| structures={len(source_files)} shards={len(tasks)} "
        f"workers={args.num_workers} backend={graph_backend_name()}",
        flush=True,
    )
    shards: List[Dict[str, object]] = []
    # Avoid forking a parent that has already imported PyTorch/torch-cluster.
    # Spawn costs a few seconds once, but is robust against inherited OpenMP
    # pools deadlocking during the long preprocessing stage.
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [executor.submit(_write_shard, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            shards.append(result)
            if completed == 1 or completed % 10 == 0 or completed == len(tasks):
                print(
                    f"Packed shards {completed}/{len(tasks)} "
                    f"| latest={result['file']} reused={result['reused']}",
                    flush=True,
                )

    shards.sort(key=lambda item: int(item["start"]))
    manifest = {
        "schema": PACKED_H5_SCHEMA,
        "source_root": str(input_root),
        "num_structures": len(source_files),
        "atom_cutoff": float(args.atom_cutoff),
        "max_atom_neighbors": int(args.max_atom_neighbors),
        "atom_graph_version": atom_graph_version(
            args.atom_cutoff,
            args.max_atom_neighbors,
        ),
        "graph_backend": graph_backend_name(),
        "max_files": args.max_files,
        "shard_size": int(args.shard_size),
        "shards": [
            {
                "file": item["file"],
                "start": int(item["start"]),
                "count": int(item["count"]),
            }
            for item in shards
        ],
    }
    temporary_manifest = manifest_path.with_suffix(".json.partial")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        "Packed dataset complete "
        f"| structures={len(source_files)} shards={len(shards)} "
        f"root={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
