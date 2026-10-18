"""Equivariant Cartesian atomic geometry encoder used by Aiden.

This module contains only periodic graph construction and the equivariant
backbone: geometry-informed embedding (GIE), the Cartesian atomic cluster
expansion (ACE) interaction, and the tensor edge cluster expansion (TECE)
interaction with radial rotary attention.  The continuous GTO density decoder
lives in ``aiden/model.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from . import cartesian as ct
from .tece_ops import TECEEdgeInteraction


# =========================
# Low-level graph / math utils
# =========================

_TORCH_CLUSTER_RADIUS = None
_TORCH_CLUSTER_IMPORT_ATTEMPTED = False


def _get_torch_cluster_radius():
    """Load the compiled radius operator only when a graph is first needed."""
    global _TORCH_CLUSTER_RADIUS, _TORCH_CLUSTER_IMPORT_ATTEMPTED
    if not _TORCH_CLUSTER_IMPORT_ATTEMPTED:
        _TORCH_CLUSTER_IMPORT_ATTEMPTED = True
        try:
            from torch_cluster import radius

            _TORCH_CLUSTER_RADIUS = radius
        except (ImportError, OSError):
            _TORCH_CLUSTER_RADIUS = None
    return _TORCH_CLUSTER_RADIUS

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out_shape = (dim_size,) + src.shape[1:]
    out = src.new_zeros(out_shape)
    out.index_add_(0, index, src)
    return out


def pbc_diff_frac(dst_frac: torch.Tensor, src_frac: torch.Tensor) -> torch.Tensor:
    diff = dst_frac - src_frac
    return diff - diff.round()


def _periodic_shifts(lattice: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Integer lattice shifts sufficient for an exact periodic radius graph."""
    inv_lattice = torch.linalg.inv(lattice.detach())
    reciprocal_column_norm = torch.linalg.norm(inv_lattice, dim=0)
    bounds = torch.ceil(
        1.0 + float(cutoff) * reciprocal_column_norm
    ).to(dtype=torch.long)
    ranges = [
        torch.arange(-int(bound.item()), int(bound.item()) + 1, device=lattice.device)
        for bound in bounds
    ]
    grid = torch.meshgrid(*ranges, indexing="ij")
    return torch.stack([x.reshape(-1) for x in grid], dim=-1).to(dtype=lattice.dtype)


def _limit_edges_per_destination(
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_vec: torch.Tensor,
    distance: torch.Tensor,
    num_destinations: int,
    max_neighbors: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_neighbors is None or max_neighbors <= 0:
        return src, dst, edge_vec, distance

    selected: List[torch.Tensor] = []
    for destination in range(num_destinations):
        edge_ids = torch.nonzero(dst == destination, as_tuple=False).flatten()
        if edge_ids.numel() > max_neighbors:
            edge_ids = edge_ids[
                torch.topk(
                    distance[edge_ids],
                    k=max_neighbors,
                    largest=False,
                    sorted=True,
                ).indices
            ]
        selected.append(edge_ids)
    if not selected:
        return src[:0], dst[:0], edge_vec[:0], distance[:0]
    keep = torch.cat(selected, dim=0)
    return src[keep], dst[keep], edge_vec[keep], distance[keep]


def make_edge_index(
    atom_frac: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    exclude_self: bool = True,
    max_neighbors: Optional[int] = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact atom-atom PBC graph including every periodic replica in the cutoff.

    ``exclude_self`` removes only the zero-distance central self edge.  Periodic
    self images are genuine neighbors and are deliberately retained.
    """
    compiled = periodic_radius_graph_sparse(
        center_frac=atom_frac,
        atom_frac=atom_frac,
        lattice=lattice,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        exclude_zero=exclude_self,
    )
    if compiled is not None:
        return compiled

    n = atom_frac.shape[0]
    requested = max_neighbors
    if requested is None or requested <= 0:
        requested = int(n * _periodic_shifts(lattice, cutoff).shape[0])
    # Ask for one extra image because the zero-distance central self image is
    # removed after top-k selection.
    search_neighbors = int(requested) + (1 if exclude_self else 0)
    src_dense, destination_to_source, distance_dense, valid_dense = (
        periodic_probe_neighbors(
            atom_frac,
            lattice,
            atom_frac,
            cutoff,
            search_neighbors,
        )
    )
    dst_dense = (
        torch.arange(n, device=atom_frac.device)
        .unsqueeze(1)
        .expand_as(src_dense)
    )
    mask = valid_dense
    if exclude_self:
        mask = mask & (distance_dense > 1.0e-8)

    src = src_dense[mask]
    dst = dst_dense[mask]
    # periodic_probe_neighbors returns destination/probe -> source-image.
    diff_cart = -destination_to_source[mask]
    distance = distance_dense[mask]
    return _limit_edges_per_destination(
        src,
        dst,
        diff_cart,
        distance,
        num_destinations=n,
        max_neighbors=max_neighbors,
    )


def periodic_probe_neighbors(
    atom_frac: torch.Tensor,
    lattice: torch.Tensor,
    query_frac: torch.Tensor,
    cutoff: float,
    max_neighbors: int,
    query_block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closest periodic atom images for each probe.

    Returns base atom indices, probe-to-image vectors, distances and a validity
    mask.  Unlike the former minimum-image top-k, the same base atom may occur
    more than once through distinct physically present periodic images.
    """
    shifts = _periodic_shifts(lattice, cutoff)
    n = atom_frac.shape[0]
    image_frac = (atom_frac.unsqueeze(0) + shifts.unsqueeze(1)).reshape(-1, 3)
    image_atom = (
        torch.arange(n, device=atom_frac.device)
        .unsqueeze(0)
        .expand(shifts.shape[0], n)
        .reshape(-1)
    )

    k = min(max(int(max_neighbors), 1), image_frac.shape[0])
    atom_chunks: List[torch.Tensor] = []
    vector_chunks: List[torch.Tensor] = []
    distance_chunks: List[torch.Tensor] = []
    valid_chunks: List[torch.Tensor] = []
    block = max(int(query_block_size), 1)
    for start in range(0, query_frac.shape[0], block):
        query = query_frac[start : start + block]
        edge_frac = image_frac.unsqueeze(0) - query.unsqueeze(1)
        edge_vec_all = edge_frac @ lattice
        distance_all = torch.linalg.norm(edge_vec_all, dim=-1)

        distance, image_index = torch.topk(
            distance_all,
            k=k,
            dim=1,
            largest=False,
            sorted=True,
        )
        atom_chunks.append(image_atom[image_index])
        vector_chunks.append(
            torch.gather(
                edge_vec_all,
                dim=1,
                index=image_index.unsqueeze(-1).expand(-1, -1, 3),
            )
        )
        distance_chunks.append(distance)
        valid_chunks.append(distance < float(cutoff))

    return (
        torch.cat(atom_chunks, dim=0),
        torch.cat(vector_chunks, dim=0),
        torch.cat(distance_chunks, dim=0),
        torch.cat(valid_chunks, dim=0),
    )


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


def periodic_radius_graph_sparse(
    center_frac: torch.Tensor,
    atom_frac: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    max_neighbors: Optional[int],
    exclude_zero: bool = False,
) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """Compiled periodic radius graph on the tensors' device.

    Returns base-atom index, center index, center-minus-image vector and
    distance.  The implementation deliberately mirrors ``dataset.py`` so
    cached training graphs and streaming full-grid graphs have the same
    selection, direction and truncation semantics.
    """
    radius = _get_torch_cluster_radius()
    if radius is None:
        return None

    shifts = _periodic_shifts(lattice, cutoff)
    num_atoms = int(atom_frac.shape[0])
    image_frac = (
        atom_frac.unsqueeze(0) + shifts.unsqueeze(1)
    ).reshape(-1, 3)
    image_atom = (
        torch.arange(num_atoms, device=atom_frac.device)
        .unsqueeze(0)
        .expand(shifts.shape[0], num_atoms)
        .reshape(-1)
    )
    center_cart = center_frac @ lattice
    image_cart = image_frac @ lattice

    requested = int(max_neighbors or 0)
    if requested > 0:
        search_neighbors = min(
            int(image_cart.shape[0]),
            max(
                5 * (requested + int(exclude_zero)),
                requested + 1,
            ),
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
        image_atom.index_select(0, image_index),
        center_index,
        edge_vector,
        distance,
    )


def cosine_cutoff(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    if cutoff <= 0:
        return torch.ones_like(dist)
    x = torch.clamp(dist / cutoff, min=0.0, max=1.0)
    return 0.5 * (torch.cos(math.pi * x) + 1.0) * (dist < cutoff).to(dist.dtype)


def poly_envelope(dist: torch.Tensor, cutoff: float, p: int = 6) -> torch.Tensor:
    """TACE C^2 polynomial envelope (Eqn 7)."""
    if cutoff <= 0:
        return torch.ones_like(dist)
    x = torch.clamp(dist / cutoff, min=0.0, max=1.0)
    env = (
        1.0
        - (p + 1) * (p + 2) / 2.0 * x ** p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2.0 * x ** (p + 2)
    )
    return env * (dist < cutoff).to(dist.dtype)


def bessel_features(dist: torch.Tensor, cutoff: float, num_rbf: int, envelope_p: int = 6) -> torch.Tensor:
    """
    Spherical-Bessel j0 radial basis with polynomial envelope (TACE Eqn 6-7):
        j0^n(r) = sqrt(2/c) * sin(n*pi*r/c) / r * f(r)
    dist: (E,) -> (E, num_rbf)
    """
    n = torch.arange(1, num_rbf + 1, device=dist.device, dtype=dist.dtype)
    r = dist.clamp_min(1e-6).unsqueeze(-1)
    base = math.sqrt(2.0 / max(cutoff, 1e-6)) * torch.sin(n * math.pi / cutoff * r) / r
    env = poly_envelope(dist, cutoff, envelope_p).unsqueeze(-1)
    return base * env


def gaussian_features(dist: torch.Tensor, cutoff: float, num_rbf: int) -> torch.Tensor:
    """Gaussian RBF fallback (kept for ablation)."""
    centers = torch.linspace(0.0, cutoff, num_rbf, device=dist.device, dtype=dist.dtype)
    width = (centers[1] - centers[0]).clamp_min(1e-6) if num_rbf > 1 else torch.tensor(cutoff)
    x = dist.unsqueeze(-1) - centers
    return torch.exp(-((x / width) ** 2)) * cosine_cutoff(dist, cutoff).unsqueeze(-1)


def radial_features(dist: torch.Tensor, cutoff: float, num_rbf: int, kind: str = "bessel") -> torch.Tensor:
    if kind == "bessel":
        return bessel_features(dist, cutoff, num_rbf)
    if kind == "gaussian":
        return gaussian_features(dist, cutoff, num_rbf)
    raise ValueError(f"unknown radial_type={kind!r}")


def edge_softmax_with_cutoff(
    logits: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    smooth_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DPA4 A3 envelope-gated attention: destination-wise softmax modulated by the cutoff envelope."""
    if logits.numel() == 0:
        return logits
    logits = logits.reshape(-1)
    smooth_weight = smooth_weight.reshape(-1).to(logits.dtype)
    maxima = logits.new_full((dim_size,), -torch.inf)
    maxima.scatter_reduce_(
        0,
        index,
        logits.detach(),
        reduce="amax",
        include_self=True,
    )
    exp_logits = torch.exp(
        logits - maxima.index_select(0, index)
    ) * smooth_weight.clamp_min(eps)
    denom = scatter_add(exp_logits, index, dim_size).index_select(0, index).clamp_min(eps)
    return exp_logits / denom


def weighted_softmax_with_cutoff(
    logits: torch.Tensor,
    smooth_weight: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    max_logits = logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits) * smooth_weight
    denom = exp_logits.sum(dim=dim, keepdim=True).clamp_min(eps)
    return exp_logits / denom


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    masked_logits = torch.where(
        mask,
        logits,
        torch.full_like(logits, torch.finfo(logits.dtype).min),
    )
    max_logits = masked_logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(masked_logits - max_logits) * mask.to(logits.dtype)
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True).clamp_min(eps)


def segment_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Numerically stable softmax over sparse edges sharing a destination."""
    if logits.numel() == 0:
        return logits
    maxima = logits.new_full((dim_size,), -torch.inf)
    maxima.scatter_reduce_(
        0,
        index,
        logits.detach(),
        reduce="amax",
        include_self=True,
    )
    weights = torch.exp(logits - maxima.index_select(0, index))
    denominator = scatter_add(weights, index, dim_size)
    return weights / denominator.index_select(0, index).clamp_min(eps)


# =========================
# Equivariant building blocks (Cartesian, channel-wise)
# =========================
# A multi-l feature is a List[torch.Tensor], element l has shape (N, C, 3,...,3) [l axes].

def zeros_like_feature(ref: torch.Tensor, channels: int, lmax: int) -> List[torch.Tensor]:
    return [ref.new_zeros((ref.shape[0], channels) + (3,) * l) for l in range(lmax + 1)]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CartLinear(nn.Module):
    """Equivariant channel-mixing linear: one Linear per weight l, shared over Cartesian components.
    Bias only on l=0 (a bias on l>0 would break equivariance)."""

    def __init__(self, lmax: int, c_in: int, c_out: int):
        super().__init__()
        self.lmax = lmax
        self.lins = nn.ModuleList([nn.Linear(c_in, c_out, bias=(l == 0)) for l in range(lmax + 1)])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for l, t in enumerate(feats):
            if l == 0:
                out.append(self.lins[0](t))
            else:
                x = t.movedim(1, -1)          # (N, *cart, C)
                x = self.lins[l](x)           # (N, *cart, C')
                out.append(x.movedim(-1, 1))  # (N, C', *cart)
        return out


class EquivariantRMSNorm(nn.Module):
    """DPA4-style equivariant RMSNorm: per weight l, normalize by the per-atom RMS of the
    channel Frobenius norms, then apply a learnable per-channel gain. Equivariant (per-atom
    per-channel scalar rescale)."""

    def __init__(self, lmax: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.lmax = lmax
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(lmax + 1, channels))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for l, t in enumerate(feats):
            # Per-channel mean-square computed DIRECTLY (no intermediate sqrt): squaring
            # frob_norm() back would reintroduce a sqrt whose gradient is 0/0 = NaN at a
            # zero feature (isolated atoms). Here the only sqrt is the outer one, taken
            # after clamp_min(eps), so its gradient is bounded by 1/(2*sqrt(eps)).
            if l == 0:
                msq = t.pow(2)                                       # (N, C)
            else:
                axes = tuple(range(t.dim() - l, t.dim()))
                msq = t.pow(2).sum(dim=axes)                         # (N, C) = ||.||_F^2
            rms = msq.mean(dim=1, keepdim=True).clamp_min(self.eps).sqrt()  # (N, 1)
            scale = (self.gain[l] / rms).reshape(t.shape[0], t.shape[1], *([1] * l))
            out.append(t * scale)
        return out


# =========================
# Shared edge cache (DPA4 Fig. 2b), Cartesian flavour
# =========================

@dataclass
class AtomEdgeCache:
    src: torch.Tensor
    dst: torch.Tensor
    edge_vec: torch.Tensor
    edge_dist: torch.Tensor
    harmonics: List[torch.Tensor]   # Cartesian harmonics Y_l(u_ij), l = 0..lmax
    rbf: torch.Tensor
    envelope: torch.Tensor
    num_nodes: int


def build_atom_edge_cache(
    atom_frac: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    lmax: int,
    num_rbf: int,
    radial_type: str,
    max_neighbors: Optional[int] = 100,
) -> AtomEdgeCache:
    n_nodes = int(atom_frac.shape[0])
    src, dst, edge_vec, edge_dist = make_edge_index(
        atom_frac,
        lattice,
        cutoff,
        exclude_self=True,
        max_neighbors=max_neighbors,
    )
    if src.numel() == 0:
        harmonics = [edge_vec.new_zeros((0,) + (3,) * l) for l in range(lmax + 1)]
        rbf = edge_vec.new_zeros((0, num_rbf))
        envelope = edge_vec.new_zeros((0,))
        return AtomEdgeCache(
            src,
            dst,
            edge_vec,
            edge_dist,
            harmonics,
            rbf,
            envelope,
            n_nodes,
        )

    uvec = edge_vec / edge_dist.clamp_min(1e-6).unsqueeze(-1)
    harmonics = [ct.cartesian_harmonic(uvec, l) for l in range(lmax + 1)]
    rbf = radial_features(edge_dist, cutoff, num_rbf, radial_type)
    envelope = poly_envelope(edge_dist, cutoff)
    return AtomEdgeCache(
        src,
        dst,
        edge_vec,
        edge_dist,
        harmonics,
        rbf,
        envelope,
        n_nodes,
    )


def build_atom_edge_cache_from_tensors(
    num_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_dist: torch.Tensor,
    cutoff: float,
    lmax: int,
    num_rbf: int,
    radial_type: str,
) -> AtomEdgeCache:
    """Materialize learnable edge attributes from a dataset-cached exact PBC graph."""
    mask = (
        (edge_dist < float(cutoff))
        & (edge_dist > 1.0e-8)
    )
    src = src[mask].long()
    dst = dst[mask].long()
    edge_vec = edge_vec[mask]
    edge_dist = edge_dist[mask]
    if src.numel() == 0:
        harmonics = [
            edge_vec.new_zeros((0,) + (3,) * l)
            for l in range(lmax + 1)
        ]
        rbf = edge_vec.new_zeros((0, num_rbf))
        envelope = edge_vec.new_zeros((0,))
        return AtomEdgeCache(
            src,
            dst,
            edge_vec,
            edge_dist,
            harmonics,
            rbf,
            envelope,
            int(num_nodes),
        )

    unit_vector = edge_vec / edge_dist.clamp_min(1.0e-6).unsqueeze(-1)
    harmonics = [
        ct.cartesian_harmonic(unit_vector, l)
        for l in range(lmax + 1)
    ]
    rbf = radial_features(
        edge_dist,
        cutoff,
        num_rbf,
        radial_type,
    )
    envelope = poly_envelope(edge_dist, cutoff)
    return AtomEdgeCache(
        src,
        dst,
        edge_vec,
        edge_dist,
        harmonics,
        rbf,
        envelope,
        int(num_nodes),
    )


def build_batched_atom_edge_cache(
    atom_frac_list: List[torch.Tensor],
    lattice_list: List[torch.Tensor],
    cutoff: float,
    lmax: int,
    num_rbf: int,
    radial_type: str,
    max_neighbors: Optional[int] = 100,
    edge_data_list: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> AtomEdgeCache:
    """Build per-cell periodic graphs, then concatenate them into one sparse graph.

    Lattices differ between structures, so neighbor discovery remains per cell.
    All learnable message-passing operations after this function run on the
    concatenated graph in one call.
    """
    if len(atom_frac_list) != len(lattice_list):
        raise ValueError("atom_frac_list and lattice_list must have equal length")
    if not atom_frac_list:
        raise ValueError("cannot build an atom graph for an empty batch")
    use_cached_edges = (
        edge_data_list is not None
        and len(edge_data_list) == len(atom_frac_list)
        and all(
            all(
                key in edge_data
                for key in (
                    "atom_edge_src",
                    "atom_edge_dst",
                    "atom_edge_vec",
                    "atom_edge_dist",
                )
            )
            for edge_data in edge_data_list
        )
    )

    caches: List[AtomEdgeCache] = []
    node_offset = 0
    src_parts: List[torch.Tensor] = []
    dst_parts: List[torch.Tensor] = []
    for structure_index, (atom_frac, lattice) in enumerate(
        zip(atom_frac_list, lattice_list)
    ):
        if use_cached_edges:
            edge_data = edge_data_list[structure_index]
            cache = build_atom_edge_cache_from_tensors(
                num_nodes=int(atom_frac.shape[0]),
                src=edge_data["atom_edge_src"],
                dst=edge_data["atom_edge_dst"],
                edge_vec=edge_data["atom_edge_vec"],
                edge_dist=edge_data["atom_edge_dist"],
                cutoff=cutoff,
                lmax=lmax,
                num_rbf=num_rbf,
                radial_type=radial_type,
            )
        else:
            cache = build_atom_edge_cache(
                atom_frac,
                lattice,
                cutoff,
                lmax,
                num_rbf,
                radial_type,
                max_neighbors=max_neighbors,
            )
        caches.append(cache)
        src_parts.append(cache.src + node_offset)
        dst_parts.append(cache.dst + node_offset)
        node_offset += cache.num_nodes

    ref = atom_frac_list[0]
    src = torch.cat(src_parts, dim=0)
    dst = torch.cat(dst_parts, dim=0)
    edge_vec = torch.cat([cache.edge_vec for cache in caches], dim=0)
    edge_dist = torch.cat([cache.edge_dist for cache in caches], dim=0)
    harmonics = [
        torch.cat([cache.harmonics[l] for cache in caches], dim=0)
        for l in range(lmax + 1)
    ]
    rbf = torch.cat([cache.rbf for cache in caches], dim=0)
    envelope = torch.cat([cache.envelope for cache in caches], dim=0)
    if node_offset <= 0:
        raise ValueError("every Cartesian encoder batch must contain at least one atom")
    if edge_vec.numel() == 0:
        edge_vec = ref.new_zeros((0, 3))

    return AtomEdgeCache(
        src=src,
        dst=dst,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        harmonics=harmonics,
        rbf=rbf,
        envelope=envelope,
        num_nodes=node_offset,
    )


def _channelize_harmonic(Y: torch.Tensor, channels: int, l: int) -> torch.Tensor:
    """(E, [3]*l) -> (E, C, [3]*l) by broadcasting over channels (view, no copy)."""
    if l == 0:
        return Y.unsqueeze(1).expand(Y.shape[0], channels)
    return Y.unsqueeze(1).expand(Y.shape[0], channels, *Y.shape[1:])


# =========================
# Geometry-Informed Embedding (DPA4 Sec. 4.2.2), Cartesian flavour
# =========================

class CartesianGIE(nn.Module):
    def __init__(self, atom_feat_dim: int, lmax: int, channels: int, num_rbf: int,
                 radial_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.num_rbf = num_rbf
        self.scalar = MLP(atom_feat_dim, [channels, channels], channels, dropout)
        self.film = MLP(atom_feat_dim + num_rbf, [radial_hidden], 2 * channels, dropout)
        if lmax >= 1:
            self.geom_radial = MLP(num_rbf, [radial_hidden], lmax * channels, dropout)
            self.species = MLP(atom_feat_dim, [channels], channels, dropout)

    def forward(self, atom_feat: torch.Tensor, cache: AtomEdgeCache) -> List[torch.Tensor]:
        n = atom_feat.shape[0]
        scalar = self.scalar(atom_feat)
        if cache.src.numel() > 0:
            inverse_sqrt_degree = scatter_add(
                cache.envelope.new_ones((cache.dst.shape[0],)),
                cache.dst,
                n,
            ).clamp_min(1.0).rsqrt()
            env = scatter_add(cache.rbf, cache.dst, n)
            env = env * inverse_sqrt_degree.unsqueeze(-1)
        else:
            inverse_sqrt_degree = atom_feat.new_ones((n,))
            env = atom_feat.new_zeros((n, self.num_rbf))
        gamma, beta = self.film(torch.cat([atom_feat, env], dim=-1)).chunk(2, dim=-1)
        scalar = scalar * (1.0 + torch.tanh(gamma)) + beta
        feats = [scalar]

        if self.lmax >= 1:
            if cache.src.numel() > 0:
                w = self.geom_radial(cache.rbf).reshape(-1, self.lmax, self.channels)  # (E, lmax, C)
                species_src = self.species(atom_feat)[cache.src]                        # (E, C)
                for l in range(1, self.lmax + 1):
                    wl = (w[:, l - 1, :] * species_src)                                 # (E, C)
                    Y = cache.harmonics[l]                                              # (E, [3]*l)
                    msg = Y.unsqueeze(1) * wl.reshape(wl.shape[0], self.channels, *([1] * l))
                    agg = scatter_add(msg, cache.dst, n)
                    agg = agg * inverse_sqrt_degree.reshape(
                        n,
                        1,
                        *([1] * l),
                    )
                    feats.append(ct.detrace(agg, l))
            else:
                for l in range(1, self.lmax + 1):
                    feats.append(atom_feat.new_zeros((n, self.channels) + (3,) * l))
        return feats


# =========================
# TACE interaction block (CG-free Cartesian ACE)
# =========================

class CartesianACEBlock(nn.Module):
    """One message-passing layer:
      (1) one-particle basis phi_ij = sum_paths R(r) * couple(h_j[l1], Y[l2] -> l3)   (TACE Eqn 11)
      (2) A-basis  = envelope-gated-attention aggregation over neighbors                (Eqn 12 + DPA4 A3)
      (3) B-basis  = self-contraction up to `correlation` (many body)                   (Eqn 13)
      (4) residual update                                                               (Eqn 18)
    """

    def __init__(self, lmax: int, channels: int, correlation: int, num_rbf: int,
                 radial_hidden: int, use_attention: bool, dropout: float = 0.0):
        super().__init__()
        if correlation < 1:
            raise ValueError("correlation must be >= 1")
        self.lmax = lmax
        self.channels = channels
        self.correlation = correlation
        self.use_attention = use_attention
        self.paths = ct.valid_paths(lmax)
        self.num_paths = len(self.paths)

        self.pre = CartLinear(lmax, channels, channels)
        self.radial = MLP(num_rbf, [radial_hidden, radial_hidden], self.num_paths * channels, dropout)
        self.self_connection = CartLinear(lmax, channels, channels)
        self.message_mix = CartLinear(lmax, channels, channels)
        if use_attention:
            self.attn_head = MLP(channels + num_rbf, [radial_hidden], 1, dropout)
        self.dest_gate = MLP(channels, [radial_hidden], channels, dropout)
        self.order_lin = nn.ModuleList([CartLinear(lmax, channels, channels) for _ in range(correlation)])
        self.out_lin = CartLinear(lmax, channels, channels)

    def _couple_features(self, A: List[torch.Tensor], B: List[torch.Tensor]) -> List[torch.Tensor]:
        """Per-channel Cartesian product of two multi-l features, summed over coupling paths."""
        out: List[Optional[torch.Tensor]] = [None] * (self.lmax + 1)
        for (l1, l2, l3) in self.paths:
            c = ct.couple(A[l1], B[l2], l1, l2, l3)
            out[l3] = c if out[l3] is None else out[l3] + c
        n = A[0].shape[0]
        return [out[l] if out[l] is not None else A[0].new_zeros((n, self.channels) + (3,) * l)
                for l in range(self.lmax + 1)]

    def forward(self, feats: List[torch.Tensor], cache: AtomEdgeCache) -> List[torch.Tensor]:
        n = cache.num_nodes
        src, dst = cache.src, cache.dst

        if src.numel() == 0:
            base = self._add(self.self_connection(feats), zeros_like_feature(feats[0], self.channels, self.lmax))
            return self._residual(feats, self._product_basis(base))

        # ----- (1) one-particle basis -----
        src_feats = [t[src] for t in self.pre(feats)]                     # gathered source, (E, C, *l)
        e = src.shape[0]
        radial_w = self.radial(cache.rbf).reshape(e, self.num_paths, self.channels)  # (E, P, C)

        phi: List[Optional[torch.Tensor]] = [None] * (self.lmax + 1)
        for p, (l1, l2, l3) in enumerate(self.paths):
            w = radial_w[:, p, :]                                         # (E, C)
            a = src_feats[l1] * w.reshape(e, self.channels, *([1] * l1))  # radial-weighted source
            Yc = _channelize_harmonic(cache.harmonics[l2], self.channels, l2)
            c = ct.couple(a, Yc, l1, l2, l3)
            phi[l3] = c if phi[l3] is None else phi[l3] + c
        phi = [phi[l] if phi[l] is not None else feats[0].new_zeros((e, self.channels) + (3,) * l)
               for l in range(self.lmax + 1)]

        # ----- (2) A-basis: envelope-gated attention aggregation (DPA4 A3) -----
        if self.use_attention:
            logit = self.attn_head(torch.cat([phi[0], cache.rbf], dim=-1)).squeeze(-1)  # (E,)
            alpha = edge_softmax_with_cutoff(logit, dst, n, cache.envelope)             # (E,)
            message = [scatter_add(phi[l] * alpha.reshape(e, 1, *([1] * l)), dst, n)
                       for l in range(self.lmax + 1)]
        else:
            inverse_sqrt_degree = scatter_add(
                cache.envelope.new_ones((e,)),
                dst,
                n,
            ).clamp_min(1.0).rsqrt()
            edge_scale = (
                cache.envelope
                * inverse_sqrt_degree.index_select(0, dst)
            )
            message = [
                scatter_add(
                    phi[l] * edge_scale.reshape(e, 1, *([1] * l)),
                    dst,
                    n,
                )
                for l in range(self.lmax + 1)
            ]

        # ----- self connection + destination output gate -----
        base = self._add(self.self_connection(feats), self.message_mix(message))
        gate = torch.sigmoid(self.dest_gate(base[0]))                                   # (N, C)
        base = [base[l] * (1.0 + gate).reshape(n, self.channels, *([1] * l)) for l in range(self.lmax + 1)]

        return self._residual(feats, self._product_basis(base))

    def _product_basis(self, base: List[torch.Tensor]) -> List[torch.Tensor]:
        out = self.order_lin[0](base)
        prod = base
        for o in range(1, self.correlation):
            prod = self._couple_features(prod, base)
            out = self._add(out, self.order_lin[o](prod))
        return self.out_lin(out)

    def _residual(self, feats: List[torch.Tensor], out: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._add(feats, out)

    @staticmethod
    def _add(a: List[torch.Tensor], b: List[torch.Tensor]) -> List[torch.Tensor]:
        return [x + y for x, y in zip(a, b)]


# =========================
# Encoder
# =========================

class AidenEncoder(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int,
        num_channels: int = 64,
        num_layers: int = 2,
        lmax: int = 3,
        correlation: int = 3,
        atom_cutoff: float = 4.0,
        max_atom_neighbors: int = 100,
        num_rbf: int = 8,
        radial_hidden: int = 128,
        radial_type: str = "bessel",
        dropout: float = 0.0,
        use_gie: bool = True,
        use_attention: bool = True,
        use_rmsnorm: bool = True,
        use_tece: bool = True,
        mmax: Optional[int] = None,
        edge_correlation: int = 2,
        num_heads: int = 4,
        edge_hidden: Optional[int] = None,
        use_ece: bool = True,
        use_rra: bool = True,
        use_radial_phase: bool = True,
        use_temperature: bool = True,
        so2_weight_type: str = "w1",
        wigner_type: str = "recursive",
        interaction_mode: str = "stacked",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if interaction_mode not in {
            "stacked",
            "alternating",
            "ace_only",
            "tece_only",
        }:
            raise ValueError(
                "interaction_mode must be one of "
                "{'stacked', 'alternating', 'ace_only', 'tece_only'}"
            )
        self.lmax = lmax
        self.channels = num_channels
        self.num_layers = int(num_layers)
        self.atom_cutoff = float(atom_cutoff)
        self.max_atom_neighbors = int(max_atom_neighbors)
        self.num_rbf = int(num_rbf)
        self.radial_type = radial_type
        self.use_gie = use_gie
        self.use_rmsnorm = use_rmsnorm
        self.use_tece = bool(use_tece)
        self.mmax = int(lmax if mmax is None else mmax)
        if not self.use_tece and interaction_mode in {"stacked", "alternating"}:
            interaction_mode = "ace_only"
        if not self.use_tece and interaction_mode == "tece_only":
            raise ValueError("interaction_mode='tece_only' requires use_tece=True")
        self.interaction_mode = str(interaction_mode)
        self._use_ace_layer = [
            self.interaction_mode in {"stacked", "ace_only"}
            or (self.interaction_mode == "alternating" and i % 2 == 0)
            for i in range(num_layers)
        ]
        self._use_tece_layer = [
            self.use_tece
            and (
                self.interaction_mode in {"stacked", "tece_only"}
                or (self.interaction_mode == "alternating" and i % 2 == 1)
            )
            for i in range(num_layers)
        ]

        if use_gie:
            self.gie = CartesianGIE(atom_feat_dim, lmax, num_channels, num_rbf, radial_hidden, dropout)
        else:
            self.embed = MLP(atom_feat_dim, [num_channels, num_channels], num_channels, dropout)

        self.blocks = nn.ModuleList([
            (
                CartesianACEBlock(
                    lmax,
                    num_channels,
                    correlation,
                    num_rbf,
                    radial_hidden,
                    use_attention,
                    dropout,
                )
                if self._use_ace_layer[i]
                else nn.Identity()
            )
            for i in range(num_layers)
        ])
        if any(self._use_tece_layer):
            self.tece_blocks = nn.ModuleList([
                (
                    TECEEdgeInteraction(
                        lmax=lmax,
                        mmax=self.mmax,
                        channels=num_channels,
                        num_rbf=num_rbf,
                        radial_hidden=radial_hidden,
                        num_heads=num_heads,
                        edge_hidden=edge_hidden,
                        edge_correlation=edge_correlation,
                        weight_type=so2_weight_type,
                        use_ece=use_ece,
                        use_rra=use_rra,
                        use_radial_phase=use_radial_phase,
                        use_temperature=use_temperature,
                        wigner_type=wigner_type,
                        dropout=dropout,
                    )
                    if self._use_tece_layer[i]
                    else nn.Identity()
                )
                for i in range(num_layers)
            ])
        if use_rmsnorm:
            self.norms = nn.ModuleList([EquivariantRMSNorm(lmax, num_channels) for _ in range(num_layers)])
            # The interaction stack is pre-normalized. Its final cubic ACE/TECE update
            # therefore also needs a terminal normalization before the probe decoder;
            # otherwise structure-dependent feature scales pass directly into density.
            self.final_norm = EquivariantRMSNorm(lmax, num_channels)

    def _forward_with_cache(
        self,
        atom_feat: torch.Tensor,
        cache: AtomEdgeCache,
    ) -> List[torch.Tensor]:
        if self.use_gie:
            feats = self.gie(atom_feat, cache)
        else:
            feats = [self.embed(atom_feat)] + [
                atom_feat.new_zeros((atom_feat.shape[0], self.channels) + (3,) * l) for l in range(1, self.lmax + 1)
            ]

        for i, block in enumerate(self.blocks):
            if self.use_rmsnorm:
                feats = self.norms[i](feats)
            if self._use_ace_layer[i]:
                feats = block(feats, cache)
            if self._use_tece_layer[i]:
                edge_update = self.tece_blocks[i](
                    feats,
                    cache.src,
                    cache.dst,
                    cache.edge_vec,
                    cache.rbf,
                    cache.envelope,
                )
                feats = [
                    (field + update) / math.sqrt(2.0)
                    for field, update in zip(feats, edge_update)
                ]
        if self.use_rmsnorm:
            feats = self.final_norm(feats)
        return feats

    def forward(
        self,
        atom_feat: torch.Tensor,
        atom_frac: torch.Tensor,
        lattice: torch.Tensor,
    ) -> List[torch.Tensor]:
        cache = build_atom_edge_cache(
            atom_frac,
            lattice,
            self.atom_cutoff,
            self.lmax,
            self.num_rbf,
            self.radial_type,
            max_neighbors=self.max_atom_neighbors,
        )
        return self._forward_with_cache(atom_feat, cache)

    def forward_batch(
        self,
        atom_feat_list: List[torch.Tensor],
        atom_frac_list: List[torch.Tensor],
        lattice_list: List[torch.Tensor],
        edge_data_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """Encode a list of periodic structures as one disconnected sparse graph."""
        if not atom_feat_list:
            raise ValueError("Cartesian encoder received an empty structure batch")
        atom_feat = torch.cat(atom_feat_list, dim=0)
        cache = build_batched_atom_edge_cache(
            atom_frac_list,
            lattice_list,
            self.atom_cutoff,
            self.lmax,
            self.num_rbf,
            self.radial_type,
            max_neighbors=self.max_atom_neighbors,
            edge_data_list=edge_data_list,
        )
        return self._forward_with_cache(atom_feat, cache)

__all__ = [
    "AtomEdgeCache",
    "CartLinear",
    "EquivariantRMSNorm",
    "MLP",
    "AidenEncoder",
    "build_atom_edge_cache",
    "build_atom_edge_cache_from_tensors",
    "build_batched_atom_edge_cache",
    "poly_envelope",
    "radial_features",
    "scatter_add",
]
