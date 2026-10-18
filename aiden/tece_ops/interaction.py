from __future__ import annotations

import math

import torch
from torch import nn

from .ece import SO2EdgeTensorProduct
from .linear import MLP, ScaledLinear, ScaledSigmoid
from .so2 import SO2Gate, UVSO2Linear, so2_expand_index
from .wigner import SphericalCartesianTransform, WignerD


def scatter_add(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    out = src.new_zeros((dim_size, *src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def graph_softmax_with_envelope(
    logits: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
    envelope: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Stable destination-wise softmax implementing TECE Eq. (38) once."""

    if logits.numel() == 0:
        return logits
    heads = logits.shape[1]
    expanded_index = index[:, None].expand(-1, heads)
    maxima = logits.new_full((num_nodes, heads), -torch.inf)
    maxima.scatter_reduce_(
        0, expanded_index, logits.detach(), reduce="amax", include_self=True
    )
    weights = torch.exp(logits - maxima.index_select(0, index))
    weights = weights * envelope.reshape(-1, 1)
    denom = logits.new_zeros((num_nodes, heads))
    denom.index_add_(0, index, weights)
    return weights / denom.index_select(0, index).clamp_min(eps)


class TECEEdgeInteraction(nn.Module):
    """Edge Cluster Expansion + Radial Rotary Attention for atom features."""

    def __init__(
        self,
        lmax: int,
        mmax: int,
        channels: int,
        num_rbf: int,
        radial_hidden: int,
        num_heads: int = 4,
        edge_hidden: int | None = None,
        edge_correlation: int = 2,
        weight_type: str = "w1",
        use_ece: bool = True,
        use_rra: bool = True,
        use_radial_phase: bool = True,
        use_temperature: bool = True,
        wigner_type: str = "recursive",
        dropout: float = 0.0,
    ):
        super().__init__()
        if edge_correlation != 2:
            raise ValueError(
                "This memory-efficient TECE density implementation supports "
                "edge_correlation=2, matching TECE-OAM-RRA."
            )
        if not 0 <= mmax <= lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
        edge_hidden = int(edge_hidden or channels)
        if edge_hidden % num_heads != 0:
            raise ValueError("edge_hidden must be divisible by num_heads")

        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.channels = int(channels)
        self.edge_hidden = edge_hidden
        self.num_heads = int(num_heads)
        self.channels_per_head = edge_hidden // num_heads
        self.use_ece = bool(use_ece)
        self.use_rra = bool(use_rra)
        self.use_radial_phase = bool(use_radial_phase)
        self.use_temperature = bool(use_temperature)

        self.transform = SphericalCartesianTransform(lmax)
        self.wigner = WignerD(mmax, lmax, wigner_type=wigner_type)
        num_unique, expand_index = so2_expand_index(mmax, lmax)
        self.num_unique_components = num_unique
        self.register_buffer("expand_index", expand_index, persistent=False)

        self.radial_weights = MLP(
            num_rbf,
            [radial_hidden, radial_hidden],
            num_unique * channels * 2,
            dropout,
        )

        padded_components = (lmax + 1) * (1 + 2 * mmax)
        self.padded_components = padded_components
        gate_m0 = False
        self.num_gates = mmax * (lmax + 1)
        components_out = [
            self.num_gates + lmax + 1,
            *([lmax + 1] * mmax),
        ]
        padded_per_m = [lmax + 1] * (mmax + 1)

        self.linear_up = UVSO2Linear(
            mmax,
            lmax,
            2 * channels,
            edge_hidden,
            num_components_out=components_out,
            weight_type=weight_type,
        )
        self.gate = SO2Gate(
            mmax,
            lmax,
            edge_hidden,
            scalar_act=ScaledSigmoid(),
            tensor_act=ScaledSigmoid(),
            gate_m0=gate_m0,
        )
        if use_ece:
            self.linear_glu = UVSO2Linear(
                mmax,
                lmax,
                2 * channels,
                edge_hidden,
                num_components_out=padded_per_m,
                weight_type=weight_type,
            )
            self.ece = SO2EdgeTensorProduct(
                mmax, lmax, edge_hidden, m1m2=">="
            )
            self.linear_coefficients = UVSO2Linear(
                0,
                lmax,
                2 * channels,
                1,
                num_components_out=[self.ece.weight_numel],
                weight_type=weight_type,
            )
        self.linear_down = UVSO2Linear(
            mmax,
            lmax,
            edge_hidden,
            edge_hidden,
            num_components_in=padded_per_m,
            weight_type=weight_type,
        )
        self.output_channels = (
            ScaledLinear(edge_hidden, channels, bias=False)
            if edge_hidden != channels
            else nn.Identity()
        )

        if use_rra:
            self.query_projection = UVSO2Linear(
                mmax, lmax, channels, edge_hidden, weight_type=weight_type
            )
            self.key_projection = UVSO2Linear(
                mmax, lmax, channels, edge_hidden, weight_type=weight_type
            )
            self.radial_attention = ScaledLinear(
                num_rbf,
                num_heads * (2 if use_radial_phase else 1),
                bias=True,
            )
            nn.init.zeros_(self.radial_attention.weight)
            nn.init.zeros_(self.radial_attention.bias)
            compact_components = (lmax + 1) + sum(
                2 * (lmax + 1 - m) for m in range(1, mmax + 1)
            )
            self.attention_scale = 1.0 / math.sqrt(
                compact_components * self.channels_per_head
            )
            if use_temperature:
                self.temperature_min = 0.25
                self.temperature_max = 4.0
                initial = math.log(
                    (1.0 - self.temperature_min)
                    / (self.temperature_max - 1.0)
                )
                self.temperature_logit = nn.Parameter(
                    torch.full((num_heads,), initial)
                )

    def _attention_logits(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        radial_basis: torch.Tensor,
    ) -> torch.Tensor:
        batch = query.shape[0]
        heads = self.num_heads
        channels = self.channels_per_head
        radial = self.radial_attention(radial_basis)
        bias = radial[:, :heads]
        phase = (
            math.pi * torch.tanh(radial[:, heads:])
            if self.use_radial_phase
            else None
        )

        n = self.lmax + 1
        q0 = query[:, :n].view(batch, n, heads, channels)
        k0 = key[:, :n].view(batch, n, heads, channels)
        score = (q0 * k0).sum(dim=(1, 3))
        offset = n
        for m in range(1, self.mmax + 1):
            n = self.lmax + 1 - m
            q = query[:, offset : offset + 2 * n].view(
                batch, 2, n, heads, channels
            )
            k = key[:, offset : offset + 2 * n].view(
                batch, 2, n, heads, channels
            )
            offset += 2 * n
            if phase is None:
                score = score + (
                    q[:, 0] * k[:, 0] + q[:, 1] * k[:, 1]
                ).sum(dim=(1, 3))
            else:
                angle = (m * phase).view(batch, 1, heads, 1)
                cos, sin = torch.cos(angle), torch.sin(angle)
                key_real = cos * k[:, 0] - sin * k[:, 1]
                key_imag = sin * k[:, 0] + cos * k[:, 1]
                score = score + (
                    q[:, 0] * key_real + q[:, 1] * key_imag
                ).sum(dim=(1, 3))

        score = score * self.attention_scale
        if self.use_temperature:
            temperature = self.temperature_min + (
                self.temperature_max - self.temperature_min
            ) * torch.sigmoid(self.temperature_logit)
            score = score * temperature
        return score + bias

    def forward(
        self,
        feats: list[torch.Tensor],
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_vector: torch.Tensor,
        radial_basis: torch.Tensor,
        envelope: torch.Tensor,
    ) -> list[torch.Tensor]:
        num_nodes = feats[0].shape[0]
        if src.numel() == 0:
            return [torch.zeros_like(field) for field in feats]

        spherical = self.transform.to_spherical(feats)
        wigner, wigner_inv = self.wigner.get_wigner(edge_vector)
        source = torch.bmm(wigner, spherical[src])
        target = torch.bmm(wigner, spherical[dst])
        local = torch.cat((source, target), dim=-1)

        if self.use_rra:
            key = self.key_projection(source)
            query = self.query_projection(target)
            logits = self._attention_logits(query, key, radial_basis)

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

        if self.use_rra:
            attention = graph_softmax_with_envelope(
                logits, dst, num_nodes, envelope
            )
            value = value.view(
                value.shape[0],
                value.shape[1],
                self.num_heads,
                self.channels_per_head,
            )
            value = (
                value * attention[:, None, :, None]
            ).reshape(value.shape[0], value.shape[1], self.edge_hidden)
        else:
            # Per-destination normalization keeps a disconnected structure's
            # output independent of the other structures in the same batch.
            degree = scatter_add(
                envelope.new_ones((dst.shape[0],)),
                dst,
                num_nodes,
            ).clamp_min(1.0)
            edge_scale = envelope * degree.index_select(0, dst).rsqrt()
            value = value * edge_scale.view(-1, 1, 1)

        global_message = torch.bmm(wigner_inv, value)
        global_message = self.output_channels(global_message)
        node_message = scatter_add(global_message, dst, num_nodes)
        return self.transform.to_cartesian(node_message)
