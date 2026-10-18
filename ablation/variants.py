"""Structural AIDEN ablations built on the production Aiden model.

Only the pieces needed to define the eight paper variants live here.  The
production training/model files remain unchanged, so the existing L sweeps and
checkpoints keep their original architecture.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ablation.config import ABLATION_VARIANTS
from aiden.model import Aiden
from aiden.tece_ops.interaction import TECEEdgeInteraction, scatter_add
from aiden.tece_ops.linear import ScaledSigmoid


class CutoffNormalizedTECE(TECEEdgeInteraction):
    """TECE with ECE intact and learned radial rotary attention removed.

    Every head receives the same parameter-free cutoff-normalized weight

        a_e = f_c(r_e) / sum_{e' -> i} f_c(r_e').

    This is the zero-logit limit of the full envelope-weighted graph softmax.
    It preserves edge correlation, the cutoff, and neighbor normalization.
    """

    def forward(
        self,
        feats: list[torch.Tensor],
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_vector: torch.Tensor,
        radial_basis: torch.Tensor,
        envelope: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.use_rra:
            raise RuntimeError("CutoffNormalizedTECE must be built with use_rra=False")
        num_nodes = feats[0].shape[0]
        if src.numel() == 0:
            return [torch.zeros_like(field) for field in feats]

        spherical = self.transform.to_spherical(feats)
        wigner, wigner_inv = self.wigner.get_wigner(edge_vector)
        source = torch.bmm(wigner, spherical[src])
        target = torch.bmm(wigner, spherical[dst])
        local = torch.cat((source, target), dim=-1)

        weights = self.radial_weights(radial_basis).view(
            src.shape[0], self.num_unique_components, 2 * self.channels
        )
        weights = torch.index_select(weights, 1, self.expand_index)
        local = local * weights

        up = self.linear_up(local)
        gates = up[:, : self.num_gates]
        value = up[:, self.num_gates : self.num_gates + self.padded_components]
        gated = self.gate(value, gates)
        if self.use_ece:
            coefficients = ScaledSigmoid()(
                self.linear_coefficients(local).squeeze(-1)
            )
            product = self.ece(
                value,
                self.linear_glu(local),
                coefficients,
            )
            value = (value + gated + product) / math.sqrt(3.0)
        else:
            value = gated
        value = self.linear_down(value)

        cutoff_sum = scatter_add(envelope, dst, num_nodes)
        attention = envelope / cutoff_sum.index_select(0, dst).clamp_min(1.0e-12)
        value = value * attention.view(-1, 1, 1)

        global_message = torch.bmm(wigner_inv, value)
        global_message = self.output_channels(global_message)
        node_message = scatter_add(global_message, dst, num_nodes)
        return self.transform.to_cartesian(node_message)


class ZeroInitialDensity(nn.Module):
    """Parameter-free replacement for the one-center density branch."""

    def forward(
        self,
        atom_z: torch.Tensor,
        atom_index: torch.Tensor,
        query_index: torch.Tensor,
        radial_s: torch.Tensor,
        num_queries: int,
    ) -> torch.Tensor:
        del atom_z, atom_index, query_index
        return radial_s.new_zeros((int(num_queries),))


class AIDENAblation(Aiden):
    """Aiden with one explicitly selected structural ablation."""

    def __init__(self, ablation_variant: str = "full", **kwargs: object) -> None:
        variant = str(ablation_variant)
        if variant not in ABLATION_VARIANTS:
            raise ValueError(
                f"unknown ablation variant {variant!r}; choose from "
                f"{', '.join(ABLATION_VARIANTS)}"
            )
        if int(kwargs.get("num_encoder_layers", 2)) != 2:
            raise ValueError("the structural ablation requires exactly two interaction layers")

        model_kwargs = dict(kwargs)
        model_kwargs["encoder_interaction_mode"] = {
            "ace_ace": "ace_only",
            "tece_tece": "tece_only",
            # Build both operators first; unused modules are removed below.
            "tece_ace": "stacked",
        }.get(variant, "alternating")
        model_kwargs["use_gie"] = variant != "no_gie"
        model_kwargs["use_rra"] = variant != "no_rra"
        model_kwargs["use_radial_correction"] = variant != "fixed_gto"

        super().__init__(**model_kwargs)
        self.ablation_variant = variant

        if variant == "tece_ace":
            # Layer 0: TECE only. Layer 1: ACE only. Replacing the unused
            # modules with Identity also removes their parameters.
            self.encoder._use_ace_layer = [False, True]
            self.encoder._use_tece_layer = [True, False]
            self.encoder.blocks[0] = nn.Identity()
            self.encoder.tece_blocks[1] = nn.Identity()
            self.encoder.interaction_mode = "tece_then_ace"

        if variant == "no_rra":
            for layer, enabled in enumerate(self.encoder._use_tece_layer):
                if enabled:
                    block = self.encoder.tece_blocks[layer]
                    if not isinstance(block, TECEEdgeInteraction):
                        raise TypeError("enabled TECE layer is not a TECEEdgeInteraction")
                    # This subclass adds only a different forward definition.
                    # Reusing the initialized module preserves every shared
                    # weight and does not advance the RNG relative to Full.
                    block.__class__ = CutoffNormalizedTECE

        if variant == "no_rho_init":
            self.decoder.initial_guess = ZeroInitialDensity()

        if variant == "fixed_gto":
            # The production basis keeps the zero-initialized correction MLP
            # allocated when it is disabled. Remove it here so the reported
            # parameter count reflects the actual fixed-basis ablation.
            self.decoder.gto.radial_correction = nn.Identity()


__all__ = [
    "ABLATION_VARIANTS",
    "AIDENAblation",
    "CutoffNormalizedTECE",
    "ZeroInitialDensity",
]
