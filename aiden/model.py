"""Aiden charge-density model.

The geometry encoder is the equivariant Cartesian ACE/TECE backbone in
``aiden/encoder.py``.  The decoder predicts left/right coefficients of
continuous atom-centred Gaussian orbitals, and the environment-dependent
density is a rank-K product of the resulting functions.  Query points only
evaluate those functions; they never enter the encoder and carry no hidden
states.  The total density decomposes into a learnable element-dependent
one-centre contribution ``rho_init`` and the environment-dependent field
``rho_env``; the one-centre module can be pretrained first, matching the
training curriculum of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import cartesian as ct
from .encoder import CartLinear, AidenEncoder, scatter_add
from .tece_ops import SphericalCartesianTransform


def smooth_absolute(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Differentiable smooth-absolute approximation (BOA convention)."""
    scaled = value * float(scale)
    return F.smooth_l1_loss(
        scaled,
        torch.zeros_like(scaled),
        reduction="none",
    ) / float(scale)


class EvenTemperedCartesianGTO(nn.Module):
    """Element-independent, all-periodic-table Cartesian Gaussian basis.

    BOA obtains per-element orbital exponents from a quantum-chemistry basis.
    That is appropriate for its five-element QM9 data but not for the ECD
    crystal benchmark, which contains transition metals and heavy atoms.  We
    therefore use a fixed even-tempered basis in Angstrom units.  The function
    representation and low-rank basis products remain BOA-like, while basis
    availability no longer depends on PySCF supporting every element.
    """

    def __init__(
        self,
        lmax: int,
        num_radial: int,
        cutoff: float,
        min_exponent: float,
        max_exponent: float,
        use_radial_correction: bool = True,
        num_smearing: int = 50,
    ) -> None:
        super().__init__()
        if lmax < 0 or lmax > 4:
            raise ValueError("Cartesian GTO basis supports 0 <= lmax <= 4")
        if num_radial <= 0:
            raise ValueError("num_radial must be positive")
        if not 0.0 < min_exponent < max_exponent:
            raise ValueError("require 0 < min_exponent < max_exponent")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")

        self.lmax = int(lmax)
        self.num_radial = int(num_radial)
        self.cutoff = float(cutoff)
        self.use_radial_correction = bool(use_radial_correction)
        self.cartesian_to_spherical = SphericalCartesianTransform(self.lmax)

        exponent_1d = torch.logspace(
            math.log10(float(min_exponent)),
            math.log10(float(max_exponent)),
            steps=self.num_radial,
            dtype=torch.float32,
        )
        exponents = exponent_1d.unsqueeze(0).repeat(self.lmax + 1, 1)
        angular_orders = torch.arange(
            self.lmax + 1,
            dtype=torch.float32,
        ).unsqueeze(-1)
        power = angular_orders + 1.5
        log_norm = (
            power * torch.log(2.0 * exponents)
            + math.log(2.0)
            - torch.special.gammaln(power)
        ) / 2.0
        self.register_buffer("exponents", exponents)
        self.register_buffer("log_norm", log_norm)

        centers = torch.linspace(0.0, self.cutoff, num_smearing)
        spacing = self.cutoff / max(num_smearing - 1, 1)
        self.register_buffer("smearing_centers", centers)
        self.register_buffer(
            "smearing_precision",
            torch.tensor(1.0 / max(spacing * spacing, 1.0e-8)),
        )
        self.radial_correction = nn.Sequential(
            nn.Linear(num_smearing, num_smearing),
            nn.SiLU(),
            nn.Linear(num_smearing, (self.lmax + 1) * self.num_radial),
        )
        # The analytic GTO basis is the exact starting point.  Radial correction
        # is introduced continuously during full-model training.
        nn.init.zeros_(self.radial_correction[-1].weight)
        nn.init.zeros_(self.radial_correction[-1].bias)

    def radial_values(self, distance: torch.Tensor) -> torch.Tensor:
        """Return radial GTO values with shape ``(edges, lmax+1, n_radial)``."""
        radius = distance.clamp_min(1.0e-8).view(-1, 1, 1)
        order = torch.arange(
            self.lmax + 1,
            device=distance.device,
            dtype=distance.dtype,
        ).view(1, -1, 1)
        exponents = self.exponents.to(distance).unsqueeze(0)
        log_norm = self.log_norm.to(distance).unsqueeze(0)
        radial = torch.exp(
            log_norm
            - exponents * radius.square()
            + order * torch.log(radius)
        )

        if self.use_radial_correction:
            smearing = torch.exp(
                -self.smearing_precision.to(distance)
                * (
                    distance.unsqueeze(-1)
                    - self.smearing_centers.to(distance).unsqueeze(0)
                ).square()
            )
            correction = self.radial_correction(smearing).view(
                -1,
                self.lmax + 1,
                self.num_radial,
            )
            radial = radial * (1.0 + correction)
        return radial

    def _evaluate_spherical_chunk(
        self,
        spherical_coefficients: torch.Tensor,
        atom_index: torch.Tensor,
        edge_vector: torch.Tensor,
        distance: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        n_edges = int(atom_index.numel())
        if n_edges == 0:
            return edge_vector.new_zeros((0, 2, int(rank)))
        unit_vector = edge_vector / distance.clamp_min(1.0e-8).unsqueeze(-1)
        radial = self.radial_values(distance)
        factors = edge_vector.new_zeros((n_edges, 2, int(rank)))
        offset = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            atom_coefficients = spherical_coefficients[
                :, offset : offset + width
            ]
            offset += width
            edge_coefficients = atom_coefficients.index_select(0, atom_index)
            harmonic_cartesian = ct.cartesian_harmonic(unit_vector, l)
            harmonic_flat = harmonic_cartesian.reshape(n_edges, -1)
            basis = getattr(
                self.cartesian_to_spherical,
                f"basis_{l}",
            ).to(harmonic_flat)
            harmonic_spherical = torch.einsum(
                "pd,np->nd",
                basis,
                harmonic_flat,
            )
            angular = torch.einsum(
                "ndc,nd->nc",
                edge_coefficients,
                harmonic_spherical,
            )
            angular = angular * (1.0 / math.sqrt(3.0)) ** l
            angular = angular.view(
                n_edges,
                2,
                int(rank),
                self.num_radial,
            )
            factors = factors + (
                angular * radial[:, l].unsqueeze(1).unsqueeze(1)
            ).sum(dim=-1)
        return factors

    def evaluate_factors(
        self,
        coefficients: Sequence[torch.Tensor],
        atom_index: torch.Tensor,
        edge_vector: torch.Tensor,
        distance: torch.Tensor,
        rank: int,
        edge_chunk_size: int,
        checkpoint_chunks: bool,
    ) -> torch.Tensor:
        """Evaluate compact spherical GTO factors in memory-bounded chunks.

        Cartesian rank-``l`` tensors contain ``3**l`` stored components but only
        ``2*l+1`` independent components.  Converting once per atom before the
        atom--probe expansion cuts the l=3 edge representation from 27 to 7
        components.  Checkpointed chunks avoid retaining every expanded edge
        tensor until backward.
        """
        if len(coefficients) != self.lmax + 1:
            raise ValueError("coefficient angular orders do not match the GTO basis")
        expected_channels = 2 * int(rank) * self.num_radial
        for l, value in enumerate(coefficients):
            if int(value.shape[1]) != expected_channels:
                raise ValueError(
                    f"l={l} has {value.shape[1]} coefficient channels; "
                    f"expected {expected_channels}"
                )
        spherical = self.cartesian_to_spherical.to_spherical(list(coefficients))
        n_edges = int(atom_index.numel())
        if n_edges == 0:
            return edge_vector.new_zeros((0, 2, int(rank)))
        chunk_size = max(int(edge_chunk_size), 1)
        chunks: List[torch.Tensor] = []
        for start in range(0, n_edges, chunk_size):
            stop = min(start + chunk_size, n_edges)
            arguments = (
                spherical,
                atom_index[start:stop],
                edge_vector[start:stop],
                distance[start:stop],
            )
            if checkpoint_chunks and self.training and spherical.requires_grad:
                value = checkpoint(
                    lambda coeff, atoms, vectors, radii: self._evaluate_spherical_chunk(
                        coeff,
                        atoms,
                        vectors,
                        radii,
                        rank,
                    ),
                    *arguments,
                    use_reentrant=False,
                )
            else:
                value = self._evaluate_spherical_chunk(*arguments, rank)
            chunks.append(value)
        return torch.cat(chunks, dim=0)


class ElementInitialDensity(nn.Module):
    """Element-only one-centre initial density (the rho_init term)."""

    def __init__(
        self,
        max_atomic_number: int,
        num_radial: int,
        rank: int,
        abs_scale: float,
    ) -> None:
        super().__init__()
        self.num_radial = int(num_radial)
        self.rank = int(rank)
        self.abs_scale = float(abs_scale)
        self.coefficients = nn.Embedding(
            int(max_atomic_number) + 1,
            2 * self.rank * self.num_radial,
            padding_idx=0,
        )
        nn.init.normal_(self.coefficients.weight, mean=0.0, std=2.0e-2)
        with torch.no_grad():
            self.coefficients.weight[0].zero_()

    def forward(
        self,
        atom_z: torch.Tensor,
        atom_index: torch.Tensor,
        query_index: torch.Tensor,
        radial_s: torch.Tensor,
        num_queries: int,
    ) -> torch.Tensor:
        if atom_index.numel() == 0:
            return radial_s.new_zeros((num_queries,))
        coefficients = self.coefficients(atom_z.index_select(0, atom_index)).view(
            -1,
            2,
            self.rank,
            self.num_radial,
        )
        factors = (
            coefficients * radial_s.unsqueeze(1).unsqueeze(1)
        ).sum(dim=-1)
        edge_density = (
            smooth_absolute(factors[:, 0], self.abs_scale)
            * factors[:, 1]
        ).sum(dim=-1)
        return scatter_add(edge_density, query_index, num_queries)


class ContinuousDensityDecoder(nn.Module):
    """Continuous low-rank density decoder over atom-centred Gaussian orbitals."""

    def __init__(
        self,
        lmax: int,
        atom_channels: int,
        num_orbitals: int = 8,
        num_radial: int = 8,
        initial_guess_rank: int = 1,
        orbital_cutoff: float = 3.0,
        min_exponent: float = 0.15,
        max_exponent: float = 256.0,
        max_atomic_number: int = 118,
        use_radial_correction: bool = True,
        use_abs: bool = True,
        abs_scale: float = 1.0e3,
        coefficient_init_scale: float = 0.10,
        gto_edge_chunk_size: int = 32768,
        checkpoint_gto_chunks: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.num_orbitals = int(num_orbitals)
        self.num_radial = int(num_radial)
        self.orbital_cutoff = float(orbital_cutoff)
        self.use_abs = bool(use_abs)
        self.abs_scale = float(abs_scale)
        self.gto_edge_chunk_size = max(int(gto_edge_chunk_size), 1)
        self.checkpoint_gto_chunks = bool(checkpoint_gto_chunks)

        output_channels = 2 * self.num_orbitals * self.num_radial
        self.coefficient_head = CartLinear(
            self.lmax,
            int(atom_channels),
            output_channels,
        )
        for l, linear in enumerate(self.coefficient_head.lins):
            nn.init.normal_(
                linear.weight,
                mean=0.0,
                std=float(coefficient_init_scale) / math.sqrt(max(atom_channels, 1)),
            )
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

        self.gto = EvenTemperedCartesianGTO(
            lmax=self.lmax,
            num_radial=self.num_radial,
            cutoff=self.orbital_cutoff,
            min_exponent=min_exponent,
            max_exponent=max_exponent,
            use_radial_correction=use_radial_correction,
        )
        self.initial_guess = ElementInitialDensity(
            max_atomic_number=max_atomic_number,
            num_radial=self.num_radial,
            rank=initial_guess_rank,
            abs_scale=self.abs_scale,
        )

    @staticmethod
    def _packed_probe_graph(
        batch_list: Sequence[Dict[str, torch.Tensor]],
        cutoff: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
    ]:
        atom_indices: List[torch.Tensor] = []
        query_indices: List[torch.Tensor] = []
        vectors: List[torch.Tensor] = []
        distances: List[torch.Tensor] = []
        atom_z_parts: List[torch.Tensor] = []
        query_sizes: List[int] = []
        atom_offset = 0
        query_offset = 0

        required = (
            "probe_edge_atom",
            "probe_edge_query",
            "probe_edge_vec",
            "probe_edge_dist",
        )
        for batch in batch_list:
            missing = [key for key in required if key not in batch]
            if missing:
                raise KeyError(f"decoder probe graph is missing keys: {missing}")
            n_query = int(batch["query_frac"].shape[0])
            query_sizes.append(n_query)
            atom_z = batch["atom_z"].long()
            atom_z_parts.append(atom_z)

            distance = batch["probe_edge_dist"]
            valid = (
                (distance < float(cutoff))
                & (distance >= 0.0)
                & (batch["probe_edge_query"] >= 0)
                & (batch["probe_edge_query"] < n_query)
            )
            if valid.any():
                atom_indices.append(batch["probe_edge_atom"][valid].long() + atom_offset)
                query_indices.append(batch["probe_edge_query"][valid].long() + query_offset)
                vectors.append(batch["probe_edge_vec"][valid])
                distances.append(distance[valid])
            atom_offset += int(atom_z.numel())
            query_offset += n_query

        reference = batch_list[0]["query_frac"]
        if atom_indices:
            atom_index = torch.cat(atom_indices, dim=0)
            query_index = torch.cat(query_indices, dim=0)
            edge_vector = torch.cat(vectors, dim=0)
            distance = torch.cat(distances, dim=0)
        else:
            atom_index = torch.empty(0, dtype=torch.long, device=reference.device)
            query_index = torch.empty(0, dtype=torch.long, device=reference.device)
            edge_vector = reference.new_zeros((0, 3))
            distance = reference.new_zeros((0,))
        return (
            atom_index,
            query_index,
            edge_vector,
            distance,
            torch.cat(atom_z_parts, dim=0),
            query_sizes,
        )

    @staticmethod
    def _split_queries(value: torch.Tensor, query_sizes: Sequence[int]) -> List[torch.Tensor]:
        return list(torch.split(value, list(map(int, query_sizes)), dim=0))

    def forward_initial_guess(
        self,
        batch_list: Sequence[Dict[str, torch.Tensor]],
    ) -> List[torch.Tensor]:
        (
            atom_index,
            query_index,
            _,
            distance,
            atom_z,
            query_sizes,
        ) = self._packed_probe_graph(batch_list, self.orbital_cutoff)
        radial_s = self.gto.radial_values(distance)[:, 0]
        density = self.initial_guess(
            atom_z,
            atom_index,
            query_index,
            radial_s,
            sum(query_sizes),
        )
        return self._split_queries(density, query_sizes)

    def forward(
        self,
        atom_features: Sequence[torch.Tensor],
        batch_list: Sequence[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        coefficients = self.coefficient_head(list(atom_features))
        (
            atom_index,
            query_index,
            edge_vector,
            distance,
            atom_z,
            query_sizes,
        ) = self._packed_probe_graph(batch_list, self.orbital_cutoff)
        num_queries = sum(query_sizes)

        radial_s = self.gto.radial_values(distance)[:, 0]
        initial_density = self.initial_guess(
            atom_z,
            atom_index,
            query_index,
            radial_s,
            num_queries,
        )

        edge_factors = self.gto.evaluate_factors(
            coefficients,
            atom_index,
            edge_vector,
            distance,
            self.num_orbitals,
            self.gto_edge_chunk_size,
            self.checkpoint_gto_chunks,
        )
        left = scatter_add(edge_factors[:, 0], query_index, num_queries)
        right = scatter_add(edge_factors[:, 1], query_index, num_queries)
        if self.use_abs:
            left = smooth_absolute(left, self.abs_scale)
        learned_density = (left * right).sum(dim=-1)

        initial_parts = self._split_queries(initial_density, query_sizes)
        learned_parts = self._split_queries(learned_density, query_sizes)
        return list(zip(initial_parts, learned_parts))


@dataclass
class AidenOutput:
    pred: torch.Tensor
    pred_initial: torch.Tensor
    pred_learned: torch.Tensor


class Aiden(nn.Module):
    """Equivariant Cartesian ACE/TECE encoder with a continuous GTO decoder."""

    def __init__(
        self,
        atom_feat_dim: int,
        num_channels: int = 48,
        num_encoder_layers: int = 2,
        lmax: int = 3,
        correlation: int = 2,
        atom_cutoff: float = 4.0,
        max_atom_neighbors: int = 100,
        num_atom_rbf: int = 8,
        radial_hidden: int = 128,
        atom_radial_type: str = "bessel",
        dropout: float = 0.0,
        use_gie: bool = True,
        use_attention: bool = True,
        use_rmsnorm: bool = True,
        use_tece: bool = True,
        mmax: int | None = None,
        edge_correlation: int = 2,
        num_heads: int = 4,
        edge_hidden: int | None = None,
        use_ece: bool = True,
        use_rra: bool = True,
        use_radial_phase: bool = True,
        use_temperature: bool = True,
        so2_weight_type: str = "w1",
        wigner_type: str = "recursive",
        encoder_interaction_mode: str = "alternating",
        num_orbitals: int = 8,
        num_gto_radial: int = 8,
        initial_guess_rank: int = 1,
        orbital_cutoff: float = 3.0,
        gto_min_exponent: float = 0.15,
        gto_max_exponent: float = 256.0,
        max_atomic_number: int = 118,
        use_radial_correction: bool = True,
        use_abs: bool = True,
        abs_scale: float = 1.0e3,
        gto_edge_chunk_size: int = 32768,
        checkpoint_gto_chunks: bool = True,
        scale_factor: float = 0.1,
        target_std: float = 1.0,
        model_name: str = "Aiden",
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.lmax = int(lmax)
        self.encoder = AidenEncoder(
            atom_feat_dim=atom_feat_dim,
            num_channels=num_channels,
            num_layers=num_encoder_layers,
            lmax=lmax,
            correlation=correlation,
            atom_cutoff=atom_cutoff,
            max_atom_neighbors=max_atom_neighbors,
            num_rbf=num_atom_rbf,
            radial_hidden=radial_hidden,
            radial_type=atom_radial_type,
            dropout=dropout,
            use_gie=use_gie,
            use_attention=use_attention,
            use_rmsnorm=use_rmsnorm,
            use_tece=use_tece,
            mmax=mmax,
            edge_correlation=edge_correlation,
            num_heads=num_heads,
            edge_hidden=edge_hidden,
            use_ece=use_ece,
            use_rra=use_rra,
            use_radial_phase=use_radial_phase,
            use_temperature=use_temperature,
            so2_weight_type=so2_weight_type,
            wigner_type=wigner_type,
            interaction_mode=encoder_interaction_mode,
        )
        self.decoder = ContinuousDensityDecoder(
            lmax=lmax,
            atom_channels=num_channels,
            num_orbitals=num_orbitals,
            num_radial=num_gto_radial,
            initial_guess_rank=initial_guess_rank,
            orbital_cutoff=orbital_cutoff,
            min_exponent=gto_min_exponent,
            max_exponent=gto_max_exponent,
            max_atomic_number=max_atomic_number,
            use_radial_correction=use_radial_correction,
            use_abs=use_abs,
            abs_scale=abs_scale,
            gto_edge_chunk_size=gto_edge_chunk_size,
            checkpoint_gto_chunks=checkpoint_gto_chunks,
        )
        density_scale = max(float(target_std) * float(scale_factor), 1.0e-8)
        self.register_buffer(
            "density_scale",
            torch.tensor(density_scale, dtype=torch.float32),
        )

    def forward_initial_guess(
        self,
        batch_list: Sequence[Dict[str, torch.Tensor]],
    ) -> List[AidenOutput]:
        normalized = self.decoder.forward_initial_guess(batch_list)
        outputs: List[AidenOutput] = []
        for value in normalized:
            initial = value * self.density_scale.to(value)
            outputs.append(
                AidenOutput(
                    pred=initial,
                    pred_initial=initial,
                    pred_learned=torch.zeros_like(initial),
                )
            )
        return outputs

    def forward(
        self,
        batch_list: Sequence[Dict[str, torch.Tensor]],
    ) -> List[AidenOutput]:
        if not batch_list:
            return []
        atom_features = self.encoder.forward_batch(
            [batch["atom_features"] for batch in batch_list],
            [batch["atom_frac"] for batch in batch_list],
            [batch["lattice"] for batch in batch_list],
            edge_data_list=list(batch_list),
        )
        decoded = self.decoder(atom_features, batch_list)
        outputs: List[AidenOutput] = []
        for initial_normalized, learned_normalized in decoded:
            scale = self.density_scale.to(initial_normalized)
            initial = initial_normalized * scale
            learned = learned_normalized * scale
            outputs.append(
                AidenOutput(
                    pred=initial + learned,
                    pred_initial=initial,
                    pred_learned=learned,
                )
            )
        return outputs


__all__ = [
    "Aiden",
    "AidenOutput",
    "ContinuousDensityDecoder",
    "ElementInitialDensity",
    "EvenTemperedCartesianGTO",
    "smooth_absolute",
]
