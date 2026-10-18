"""Streaming, encoder-cached inference for the Aiden model.

The acceleration is exact in model semantics: the structure encoder and atom
GTO coefficients are evaluated once, then any number of probe blocks are
decoded from those cached tensors.  Query points therefore never enter the
equivariant encoder, which keeps the encoding cost independent of the density
grid resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import torch

from .model import Aiden


@dataclass
class PreparedStructure:
    """Query-independent tensors for one encoded periodic structure."""

    spherical_coefficients: torch.Tensor
    cartesian_coefficients: List[torch.Tensor]
    atom_z: torch.Tensor
    atom_frac: torch.Tensor
    lattice: torch.Tensor


def _periodic_shifts(lattice: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Return all lattice translations needed by the exact radius search."""
    inverse_lattice = torch.linalg.inv(lattice.detach())
    reciprocal_column_norm = torch.linalg.norm(inverse_lattice, dim=0)
    bounds = torch.ceil(1.0 + float(cutoff) * reciprocal_column_norm).to(torch.long)
    ranges = [
        torch.arange(-int(bound.item()), int(bound.item()) + 1, device=lattice.device)
        for bound in bounds
    ]
    mesh = torch.meshgrid(*ranges, indexing="ij")
    return torch.stack([value.reshape(-1) for value in mesh], dim=-1).to(lattice.dtype)


class GPUProbeGraphBuilder:
    """Construct an exact periodic atom-to-probe radius graph.

    It has the same cutoff and periodic-image semantics as
    ``dataset.build_probe_graph_np`` but never materializes a full-grid graph
    for the whole structure.  ``torch-cluster`` is used when available; a
    block-local pure-PyTorch search keeps the copied code runnable on CPU-only
    systems as well.
    """

    def __init__(
        self,
        atom_frac: torch.Tensor,
        lattice: torch.Tensor,
        cutoff: float,
        strict_order: bool = True,
    ) -> None:
        try:
            from torch_cluster import radius
        except (ImportError, OSError):
            radius = None

        self.radius = radius
        self.lattice = lattice
        self.cutoff = float(cutoff)
        self.strict_order = bool(strict_order)
        shifts = _periodic_shifts(lattice, self.cutoff)
        self.image_frac = (atom_frac.unsqueeze(0) + shifts.unsqueeze(1)).reshape(-1, 3)
        self.image_cart = self.image_frac @ lattice
        self.image_atom = (
            torch.arange(atom_frac.shape[0], device=atom_frac.device)
            .unsqueeze(0)
            .expand(shifts.shape[0], atom_frac.shape[0])
            .reshape(-1)
        )

    def build(
        self,
        query_frac: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return atom indices, local query indices, vectors, and distances."""
        query_cart = query_frac @ self.lattice
        if self.radius is None:
            # The outer inference loop already bounds query blocks, so this
            # dense temporary never scales with the full FFT grid.
            distance_matrix = torch.cdist(query_cart, self.image_cart)
            query_index, image_index = torch.where(distance_matrix < self.cutoff)
            distance = distance_matrix[query_index, image_index]
            edge_vector = query_cart.index_select(
                0, query_index
            ) - self.image_cart.index_select(0, image_index)
        else:
            query_index, image_index = self.radius(
                self.image_cart,
                query_cart,
                self.cutoff,
                max_num_neighbors=max(int(self.image_cart.shape[0]), 1),
            )
            edge_vector = query_cart.index_select(
                0, query_index
            ) - self.image_cart.index_select(0, image_index)
            distance = torch.linalg.norm(edge_vector, dim=-1)
            valid = distance < self.cutoff
            query_index = query_index[valid]
            image_index = image_index[valid]
            edge_vector = edge_vector[valid]
            distance = distance[valid]

        # dataset.py orders by probe and then distance before decoding.  It is
        # unnecessary mathematically but preserves the same scatter-add order
        # for the strict numerical-equivalence path.
        if self.strict_order and query_index.numel():
            sort_key = query_index.to(torch.float64) + distance.to(torch.float64) / max(
                self.cutoff * 2.0, 1.0e-12
            )
            order = torch.argsort(sort_key, stable=True)
            query_index = query_index[order]
            image_index = image_index[order]
            edge_vector = edge_vector[order]
            distance = distance[order]

        return self.image_atom.index_select(0, image_index), query_index, edge_vector, distance


def iter_regular_grid_queries(
    grid_shape: Sequence[int],
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[torch.Tensor]:
    """Yield VASP/CHGCAR flattened-grid fractional coordinates in blocks."""
    nx, ny, nz = (int(value) for value in grid_shape)
    total = nx * ny * nz
    for start in range(0, total, int(block_size)):
        stop = min(start + int(block_size), total)
        index = torch.arange(start, stop, device=device)
        ix = index.remainder(nx)
        iy = torch.div(index, nx, rounding_mode="floor").remainder(ny)
        iz = torch.div(index, nx * ny, rounding_mode="floor")
        yield torch.stack(
            [
                ix.to(dtype) / float(nx),
                iy.to(dtype) / float(ny),
                iz.to(dtype) / float(nz),
            ],
            dim=-1,
        )


class FastAidenInference:
    """Reuse Aiden's query-independent encoder across probe blocks."""

    def __init__(self, model: Aiden) -> None:
        if model.training:
            raise ValueError("FastAidenInference requires model.eval()")
        self.model = model
        self.decoder = model.decoder

    @torch.inference_mode()
    def prepare(self, structure: Dict[str, torch.Tensor]) -> PreparedStructure:
        """Run the expensive TACE/TECE encoder and coefficient head once."""
        atom_features = self.model.encoder.forward_batch(
            [structure["atom_features"]],
            [structure["atom_frac"]],
            [structure["lattice"]],
            edge_data_list=[structure],
        )
        cartesian_coefficients = self.decoder.coefficient_head(atom_features)
        spherical_coefficients = self.decoder.gto.cartesian_to_spherical.to_spherical(
            list(cartesian_coefficients)
        )
        return PreparedStructure(
            spherical_coefficients=spherical_coefficients,
            cartesian_coefficients=list(cartesian_coefficients),
            atom_z=structure["atom_z"].long(),
            atom_frac=structure["atom_frac"],
            lattice=structure["lattice"],
        )

    def _evaluate_factors(
        self,
        prepared: PreparedStructure,
        atom_index: torch.Tensor,
        edge_vector: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        """Same GTO evaluation as Aiden, with cached spherical coefficients."""
        if atom_index.numel() == 0:
            return edge_vector.new_zeros((0, 2, self.decoder.num_orbitals))
        chunks: List[torch.Tensor] = []
        edge_chunk_size = self.decoder.gto_edge_chunk_size
        for start in range(0, int(atom_index.numel()), edge_chunk_size):
            stop = min(start + edge_chunk_size, int(atom_index.numel()))
            chunks.append(
                self.decoder.gto._evaluate_spherical_chunk(
                    prepared.spherical_coefficients,
                    atom_index[start:stop],
                    edge_vector[start:stop],
                    distance[start:stop],
                    self.decoder.num_orbitals,
                )
            )
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def decode(
        self,
        prepared: PreparedStructure,
        query_frac: torch.Tensor,
        probe_edge_atom: torch.Tensor,
        probe_edge_query: torch.Tensor,
        probe_edge_vec: torch.Tensor,
        probe_edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one query block without recomputing the geometry encoder."""
        num_queries = int(query_frac.shape[0])
        valid = (
            (probe_edge_dist < self.decoder.orbital_cutoff)
            & (probe_edge_dist >= 0.0)
            & (probe_edge_query >= 0)
            & (probe_edge_query < num_queries)
        )
        atom_index = probe_edge_atom[valid].long()
        query_index = probe_edge_query[valid].long()
        edge_vector = probe_edge_vec[valid]
        distance = probe_edge_dist[valid]

        radial_s = self.decoder.gto.radial_values(distance)[:, 0]
        initial_density = self.decoder.initial_guess(
            prepared.atom_z,
            atom_index,
            query_index,
            radial_s,
            num_queries,
        )
        edge_factors = self._evaluate_factors(
            prepared,
            atom_index,
            edge_vector,
            distance,
        )
        left = edge_factors.new_zeros((num_queries, self.decoder.num_orbitals))
        right = edge_factors.new_zeros((num_queries, self.decoder.num_orbitals))
        left.index_add_(0, query_index, edge_factors[:, 0])
        right.index_add_(0, query_index, edge_factors[:, 1])
        if self.decoder.use_abs:
            left = torch.nn.functional.smooth_l1_loss(
                left * self.decoder.abs_scale,
                torch.zeros_like(left),
                reduction="none",
            ) / self.decoder.abs_scale
        learned_density = (left * right).sum(dim=-1)
        return (initial_density + learned_density) * self.model.density_scale.to(initial_density)

    @torch.inference_mode()
    def decode_reference(
        self,
        prepared: PreparedStructure,
        query_frac: torch.Tensor,
        probe_edge_atom: torch.Tensor,
        probe_edge_query: torch.Tensor,
        probe_edge_vec: torch.Tensor,
        probe_edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Run Aiden's unmodified GTO evaluator from the cached coefficients.

        This is a benchmark-only numerical reference. It differs from
        :meth:`decode` solely in recomputing the spherical coefficient transform
        inside ``gto.evaluate_factors``, exactly as the original decoder does.
        """
        num_queries = int(query_frac.shape[0])
        valid = (
            (probe_edge_dist < self.decoder.orbital_cutoff)
            & (probe_edge_dist >= 0.0)
            & (probe_edge_query >= 0)
            & (probe_edge_query < num_queries)
        )
        atom_index = probe_edge_atom[valid].long()
        query_index = probe_edge_query[valid].long()
        edge_vector = probe_edge_vec[valid]
        distance = probe_edge_dist[valid]

        radial_s = self.decoder.gto.radial_values(distance)[:, 0]
        initial_density = self.decoder.initial_guess(
            prepared.atom_z,
            atom_index,
            query_index,
            radial_s,
            num_queries,
        )
        edge_factors = self.decoder.gto.evaluate_factors(
            prepared.cartesian_coefficients,
            atom_index,
            edge_vector,
            distance,
            self.decoder.num_orbitals,
            self.decoder.gto_edge_chunk_size,
            self.decoder.checkpoint_gto_chunks,
        )
        left = edge_factors.new_zeros((num_queries, self.decoder.num_orbitals))
        right = edge_factors.new_zeros((num_queries, self.decoder.num_orbitals))
        left.index_add_(0, query_index, edge_factors[:, 0])
        right.index_add_(0, query_index, edge_factors[:, 1])
        if self.decoder.use_abs:
            left = torch.nn.functional.smooth_l1_loss(
                left * self.decoder.abs_scale,
                torch.zeros_like(left),
                reduction="none",
            ) / self.decoder.abs_scale
        learned_density = (left * right).sum(dim=-1)
        return (initial_density + learned_density) * self.model.density_scale.to(initial_density)

    @torch.inference_mode()
    def predict_prebuilt(self, prepared: PreparedStructure, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict stored probes whose graph was built by the normal dataset."""
        return self.decode(
            prepared,
            sample["query_frac"],
            sample["probe_edge_atom"],
            sample["probe_edge_query"],
            sample["probe_edge_vec"],
            sample["probe_edge_dist"],
        )

    @torch.inference_mode()
    def predict_prebuilt_reference(
        self,
        prepared: PreparedStructure,
        sample: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reference prediction using the original non-cached GTO evaluator."""
        return self.decode_reference(
            prepared,
            sample["query_frac"],
            sample["probe_edge_atom"],
            sample["probe_edge_query"],
            sample["probe_edge_vec"],
            sample["probe_edge_dist"],
        )

    @torch.inference_mode()
    def predict_prebuilt_chunked(
        self,
        prepared: PreparedStructure,
        sample: Dict[str, torch.Tensor],
        query_block_size: int,
    ) -> torch.Tensor:
        """Decode the normal dataset probes in blocks, retaining exact inputs."""
        predictions: List[torch.Tensor] = []
        num_queries = int(sample["query_frac"].shape[0])
        for start in range(0, num_queries, int(query_block_size)):
            stop = min(start + int(query_block_size), num_queries)
            edge_mask = (sample["probe_edge_query"] >= start) & (
                sample["probe_edge_query"] < stop
            )
            predictions.append(
                self.decode(
                    prepared,
                    sample["query_frac"][start:stop],
                    sample["probe_edge_atom"][edge_mask],
                    sample["probe_edge_query"][edge_mask] - start,
                    sample["probe_edge_vec"][edge_mask],
                    sample["probe_edge_dist"][edge_mask],
                )
            )
        return torch.cat(predictions, dim=0)

    @torch.inference_mode()
    def predict_full_grid(
        self,
        prepared: PreparedStructure,
        grid_shape: Sequence[int],
        query_block_size: int,
        strict_graph_order: bool = True,
    ) -> torch.Tensor:
        """Stream a full regular grid without creating a full probe graph."""
        builder = GPUProbeGraphBuilder(
            prepared.atom_frac,
            prepared.lattice,
            self.decoder.orbital_cutoff,
            strict_order=strict_graph_order,
        )
        predictions: List[torch.Tensor] = []
        for query_frac in iter_regular_grid_queries(
            grid_shape,
            query_block_size,
            prepared.atom_frac.device,
            prepared.atom_frac.dtype,
        ):
            atom_index, query_index, edge_vector, distance = builder.build(query_frac)
            predictions.append(
                self.decode(
                    prepared,
                    query_frac,
                    atom_index,
                    query_index,
                    edge_vector,
                    distance,
                )
            )
        return torch.cat(predictions, dim=0)


def slice_prebuilt_sample(
    sample: Dict[str, torch.Tensor],
    start: int,
    stop: int,
) -> Dict[str, torch.Tensor]:
    """Return a normal-model-compatible view of one stored-probe block."""
    edge_mask = (sample["probe_edge_query"] >= start) & (sample["probe_edge_query"] < stop)
    block = dict(sample)
    for key in ("query_frac", "query_cart", "target"):
        if key in block:
            block[key] = block[key][start:stop]
    block["probe_edge_atom"] = sample["probe_edge_atom"][edge_mask]
    block["probe_edge_query"] = sample["probe_edge_query"][edge_mask] - start
    block["probe_edge_vec"] = sample["probe_edge_vec"][edge_mask]
    block["probe_edge_dist"] = sample["probe_edge_dist"][edge_mask]
    return block


__all__ = [
    "FastAidenInference",
    "GPUProbeGraphBuilder",
    "PreparedStructure",
    "iter_regular_grid_queries",
    "slice_prebuilt_sample",
]
