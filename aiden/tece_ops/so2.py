from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .linear import ScaledLinear


def satisfy(left: int, right: int, restriction: Optional[str] = None) -> bool:
    if restriction is None:
        return True
    operations = {
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
        "==": left == right,
        "!=": left != right,
    }
    if restriction not in operations:
        raise ValueError(f"Unknown restriction: {restriction}")
    return operations[restriction]


def so2_expand_index(mmax: int, lmax: int, start: int = 0) -> tuple[int, torch.Tensor]:
    indices = []
    offset = 0
    for m in range(start, mmax + 1):
        index = torch.arange(lmax + 1 - m) + offset
        indices.append(index)
        if m > 0:
            indices.append(index)
        offset += len(index)
    if not indices:
        return 0, torch.empty(0, dtype=torch.long)
    return offset, torch.cat(indices).long()


def num_so2_components(lmax: int, mmax: int) -> int:
    return (lmax + 1) + sum(
        2 * (lmax + 1 - m) for m in range(1, mmax + 1)
    )


class UVSO2MLinear(nn.Module):
    """Channel/degree mixing within one complex SO(2) order."""

    def __init__(
        self,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: int,
        num_components_out: int,
        weight_type: str = "w1",
    ):
        super().__init__()
        self.num_channel_in = int(num_channel_in)
        self.num_channel_out = int(num_channel_out)
        self.num_components_in = int(num_components_in)
        self.num_components_out = int(num_components_out)
        self.weight_type = weight_type
        if weight_type not in {"w1", "w1_w1", "w1_w2"}:
            raise ValueError(f"Unsupported SO(2) weight type: {weight_type}")

        in_dim = self.num_components_in * self.num_channel_in
        out_dim = self.num_components_out * self.num_channel_out
        self.out_dim = out_dim
        self.fc = ScaledLinear(
            in_dim,
            out_dim * (2 if weight_type == "w1_w2" else 1),
            bias=False,
        )
        if weight_type in {"w1_w1", "w1_w2"}:
            self.fc.weight.data.mul_(1.0 / math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.weight_type == "w1_w2":
            projected = self.fc(x)
            w1_x = projected[..., : self.out_dim]
            w2_x = projected[..., self.out_dim :]
            real = w1_x[:, 0:1] - w2_x[:, 1:2]
            imag = w1_x[:, 1:2] + w2_x[:, 0:1]
            return real, imag
        if self.weight_type == "w1_w1":
            real_in, imag_in = x[:, 0:1], x[:, 1:2]
            return self.fc(real_in - imag_in), self.fc(imag_in + real_in)
        projected = self.fc(x)
        return projected[:, 0:1], projected[:, 1:2]


class UVSO2Linear(nn.Module):
    """The uvSO2Linear operator used by TECE for local edge features."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: Optional[list[int]] = None,
        num_components_out: Optional[list[int]] = None,
        weight_type: str = "w1",
    ):
        super().__init__()
        self.mmax = int(mmax)
        self.lmax = int(lmax)
        self.num_channel_in = int(num_channel_in)
        self.num_channel_out = int(num_channel_out)
        self.num_components_in = (
            list(num_components_in)
            if num_components_in is not None
            else [lmax + 1 - m for m in range(mmax + 1)]
        )
        self.num_components_out = (
            list(num_components_out)
            if num_components_out is not None
            else [lmax + 1 - m for m in range(mmax + 1)]
        )
        if len(self.num_components_in) != mmax + 1:
            raise ValueError("num_components_in must have mmax + 1 entries")
        if len(self.num_components_out) != mmax + 1:
            raise ValueError("num_components_out must have mmax + 1 entries")

        self.m0_linear = ScaledLinear(
            self.num_components_in[0] * num_channel_in,
            self.num_components_out[0] * num_channel_out,
            bias=True,
        )
        self.m_linears = nn.ModuleList(
            [
                UVSO2MLinear(
                    num_channel_in,
                    num_channel_out,
                    self.num_components_in[m],
                    self.num_components_out[m],
                    weight_type=weight_type,
                )
                for m in range(1, mmax + 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        outputs = []

        width0 = self.num_components_in[0]
        m0 = x[:, :width0].reshape(batch, width0 * self.num_channel_in)
        m0 = self.m0_linear(m0).view(
            batch, self.num_components_out[0], self.num_channel_out
        )
        outputs.append(m0)

        offset = width0
        for m in range(1, self.mmax + 1):
            width = self.num_components_in[m]
            block = x[:, offset : offset + 2 * width].reshape(
                batch, 2, width * self.num_channel_in
            )
            offset += 2 * width
            real, imag = self.m_linears[m - 1](block)
            outputs.append(
                real.reshape(
                    batch, self.num_components_out[m], self.num_channel_out
                )
            )
            outputs.append(
                imag.reshape(
                    batch, self.num_components_out[m], self.num_channel_out
                )
            )
        return torch.cat(outputs, dim=1)


class SO2Gate(nn.Module):
    """SO(2) gate for the padded ECE layout."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        scalar_act: nn.Module,
        tensor_act: nn.Module,
        gate_m0: bool = False,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.num_channel = int(num_channel)
        self.gate_m0 = bool(gate_m0)
        self.num_m0_components = lmax + 1
        gate_start = 0 if gate_m0 else 1

        expanded = []
        offset = 0
        for m in range(gate_start, mmax + 1):
            index = torch.arange(lmax + 1) + offset
            expanded.append(index)
            if m > 0:
                expanded.append(index)
            offset += lmax + 1
        self.num_gates = offset
        self.register_buffer(
            "expand_index",
            torch.cat(expanded).long() if expanded else torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self.scalar_act = scalar_act
        self.tensor_act = tensor_act

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        gates = self.tensor_act(gates).view(
            batch, self.num_gates, self.num_channel
        )
        gates = torch.index_select(gates, 1, self.expand_index)
        if self.gate_m0:
            return gates * x
        return torch.cat(
            (
                self.scalar_act(x[:, : self.num_m0_components]),
                gates * x[:, self.num_m0_components :],
            ),
            dim=1,
        )
