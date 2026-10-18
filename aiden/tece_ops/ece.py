from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .so2 import satisfy


class SO2EdgeTensorProduct(nn.Module):
    """Pairwise complex product basis used by the published TECE (ECE order 2).

    Features use the padded layout
    ``m=0 real; m=1 real,imag; ...`` with ``lmax+1`` degree slots for every
    order.  Learnable edge-dependent coefficients are real, preserving O(2)
    reflection behavior as recommended in the TECE paper.
    """

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        m1m2: Optional[str] = ">=",
    ):
        super().__init__()
        self.mmax = int(mmax)
        self.lmax = int(lmax)
        self.num_channels = int(num_channels)
        self.m1m2 = m1m2
        self.instructions: list[list[tuple[int, int, str]]] = []

        weight_numel = 0
        for m_out in range(mmax + 1):
            paths = self._enumerate_paths(m_out)
            self.instructions.append(paths)
            weight_numel += num_channels * (lmax + 1) * len(paths)
        self.weight_numel = weight_numel

        scales = []
        n = lmax + 1
        for m_out, paths in enumerate(self.instructions):
            scale = 1.0 / math.sqrt(max(len(paths), 1))
            scales.append(torch.full((n if m_out == 0 else 2 * n,), scale))
        self.register_buffer("output_scales", torch.cat(scales), persistent=False)

    def _enumerate_paths(self, m_out: int) -> list[tuple[int, int, str]]:
        paths = []
        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                if not satisfy(m1, m2, self.m1m2):
                    continue
                if m1 + m2 == m_out:
                    paths.append((m1, m2, "sum"))
                elif abs(m1 - m2) == m_out:
                    paths.append((m1, m2, "diff"))
        return paths

    def _split(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch = x.shape[0]
        n = self.lmax + 1
        blocks = [x[:, :n]]
        offset = n
        for _m in range(1, self.mmax + 1):
            blocks.append(
                x[:, offset : offset + 2 * n].view(
                    batch, 2, n, self.num_channels
                )
            )
            offset += 2 * n
        return blocks

    @staticmethod
    def _complex_mul(
        x: torch.Tensor, y: torch.Tensor, mode: str
    ) -> torch.Tensor:
        a, b = x[:, 0], x[:, 1]
        c, d = y[:, 0], y[:, 1]
        if mode == "sum":
            real, imag = a * c - b * d, a * d + b * c
        else:
            real, imag = a * c + b * d, b * c - a * d
        return torch.stack((real, imag), dim=1)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        xs, ys = self._split(x), self._split(y)
        batch = x.shape[0]
        n = self.lmax + 1
        channels = self.num_channels
        outputs = []
        offset = 0

        scalar = x.new_zeros(batch, n, channels)
        width = n * channels
        for m1, m2, mode in self.instructions[0]:
            w = weights[:, offset : offset + width].view(batch, n, channels)
            offset += width
            if m1 == 0 and m2 == 0:
                scalar = scalar + xs[0] * ys[0] * w
            elif m1 > 0 and m2 > 0:
                scalar = scalar + self._complex_mul(xs[m1], ys[m2], mode)[:, 0] * w
        outputs.append(scalar)

        for m_out in range(1, self.mmax + 1):
            real = x.new_zeros(batch, n, channels)
            imag = x.new_zeros(batch, n, channels)
            for m1, m2, mode in self.instructions[m_out]:
                w = weights[:, offset : offset + width].view(
                    batch, 1, n, channels
                )
                offset += width
                if m1 == 0:
                    z = xs[0].unsqueeze(1) * ys[m2]
                elif m2 == 0:
                    z = xs[m1] * ys[0].unsqueeze(1)
                elif m1 < m2 and mode == "diff":
                    z = self._complex_mul(ys[m2], xs[m1], mode)
                else:
                    z = self._complex_mul(xs[m1], ys[m2], mode)
                real = real + z[:, 0] * w[:, 0]
                imag = imag + z[:, 1] * w[:, 0]
            outputs.extend((real, imag))

        out = torch.cat(outputs, dim=1)
        return out * self.output_scales.view(1, -1, 1).to(out)
