from __future__ import annotations

import math

import torch
from ..e3nn import o3
from torch import nn


def ictd_path_matrix(
    rank: int,
    degree: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Orthonormal spherical-to-Cartesian path matrix from TACE's ICTD."""

    current_rank = 0
    current_degree = 0
    current = o3.wigner_3j(0, 0, 0, dtype=dtype, device=device)
    answer: list[torch.Tensor] = []

    def visit(
        rank_now: int,
        degree_now: int,
        path_matrix: torch.Tensor,
    ) -> None:
        if answer:
            return
        if rank_now == rank:
            if degree_now == degree:
                matrix = path_matrix.reshape(-1, path_matrix.shape[-1])
                matrix = matrix / (matrix.square().sum(0)[0].sqrt())
                answer.append(matrix)
            return
        for degree_next in (degree_now + 1, degree_now, degree_now - 1):
            if degree_now == 0 and degree_next != 1:
                continue
            if degree_next < 0:
                continue
            cg = o3.wigner_3j(
                1, degree_now, degree_next, dtype=dtype, device=device
            )
            updated = torch.einsum("abc,dce->dabe", path_matrix, cg)
            updated = updated.reshape(cg.shape[0], -1, cg.shape[-1])
            visit(rank_now + 1, degree_next, updated)

    visit(current_rank, current_degree, current)
    if not answer:
        raise RuntimeError(f"No ICTD path for rank={rank}, degree={degree}")
    return answer[0]


def _norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps * eps)


def _quaternion_normalize(q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return q / _norm(q, eps)


def _smooth_step_cinf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    eps = torch.finfo(x.dtype).eps
    left = torch.exp(-1.0 / torch.clamp(x, min=eps))
    right = torch.exp(-1.0 / torch.clamp(1.0 - x, min=eps))
    interior = left / (left + right)
    return torch.where(
        x <= 0.0,
        torch.zeros_like(x),
        torch.where(x >= 1.0, torch.ones_like(x), interior),
    )


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack(
                (
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    2 * (x * y + w * z),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - w * x),
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x * x + y * y),
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )


def init_edge_rotation(edge_vector: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Smooth quaternion frame aligning an edge with the local y axis."""

    unit = edge_vector / _norm(edge_vector, eps)
    x, y, z = unit.unbind(dim=-1)
    q_pos = _quaternion_normalize(
        torch.stack((1 + y, -z, torch.zeros_like(x), x), dim=-1), eps
    )
    q_neg = _quaternion_normalize(
        torch.stack((-z, 1 - y, x, torch.zeros_like(x)), dim=-1), eps
    )
    weight = _smooth_step_cinf(0.5 * (y + 1.0))
    dot = torch.sum(q_neg * q_pos, dim=-1, keepdim=True)
    q_pos = torch.where(dot < 0, -q_pos, q_pos)
    q = _quaternion_normalize(
        (1 - weight.unsqueeze(-1)) * q_neg + weight.unsqueeze(-1) * q_pos,
        eps,
    )
    return _quaternion_to_rotation_matrix(q)


class CoefficientMapping(nn.Module):
    def __init__(self, lmax: int, mmax: int):
        super().__init__()
        m_complex, m_abs, degrees = [], [], []
        for degree in range(lmax + 1):
            order = torch.arange(
                -min(mmax, degree), min(mmax, degree) + 1
            ).long()
            m_complex.append(order)
            m_abs.append(order.abs())
            degrees.append(torch.full_like(order, degree))
        m_complex_t = torch.cat(m_complex)
        m_abs_t = torch.cat(m_abs)
        degrees_t = torch.cat(degrees)
        size = len(degrees_t)
        to_m = torch.zeros(size, size)
        offset = 0
        for m in range(mmax + 1):
            real = torch.where(m_complex_t == m)[0]
            imag = (
                torch.where(m_complex_t == -m)[0]
                if m > 0
                else torch.empty(0, dtype=torch.long)
            )
            to_m[offset : offset + len(real), real] = torch.eye(len(real))
            offset += len(real)
            if len(imag):
                to_m[offset : offset + len(imag), imag] = torch.eye(len(imag))
                offset += len(imag)
        self.register_buffer("to_m", to_m, persistent=False)
        self.register_buffer("degrees", degrees_t, persistent=False)
        self.register_buffer("m_abs", m_abs_t, persistent=False)


class WignerD(nn.Module):
    """Direct/recursive Wigner-D construction from the TECE implementation."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        wigner_type: str = "recursive",
    ):
        super().__init__()
        if not 0 <= mmax <= lmax:
            raise ValueError("WignerD requires 0 <= mmax <= lmax")
        if wigner_type not in {"recursive", "direct"}:
            raise ValueError("wigner_type must be 'recursive' or 'direct'")
        self.mmax = int(mmax)
        self.lmax = int(lmax)
        self.wigner_type = wigner_type

        for degree in range(2, lmax + 1):
            self.register_buffer(
                f"cg_{degree}",
                o3.wigner_3j(1, degree - 1, degree),
                persistent=False,
            )
        for degree in range(lmax + 1):
            self.register_buffer(
                f"cart_basis_{degree}",
                ictd_path_matrix(degree, degree),
                persistent=False,
            )

        full_mapping = CoefficientMapping(lmax, lmax)
        kept = torch.where(full_mapping.m_abs <= mmax)[0]
        compact_mapping = CoefficientMapping(lmax, mmax)
        selector = torch.zeros(
            compact_mapping.to_m.shape[0], (lmax + 1) ** 2
        )
        selector[:, kept] = compact_mapping.to_m
        self.register_buffer("selector", selector, persistent=False)

        inverse_rescale = torch.ones((lmax + 1) ** 2, (lmax + 1) ** 2)
        for degree in range(lmax + 1):
            if degree <= mmax:
                continue
            start = degree**2
            width = 2 * degree + 1
            inverse_rescale[
                start : start + width, start : start + width
            ] = math.sqrt(width / (2 * mmax + 1))
        inverse_rescale = inverse_rescale[:, kept]
        inverse_rescale = torch.einsum(
            "ia,ba->ib", inverse_rescale, compact_mapping.to_m
        )
        self.register_buffer(
            "inverse_rescale",
            inverse_rescale.unsqueeze(0),
            persistent=False,
        )

    def _direct(self, rotation: torch.Tensor) -> torch.Tensor:
        batch = rotation.shape[0]
        blocks = []
        for degree in range(self.lmax + 1):
            basis = getattr(self, f"cart_basis_{degree}")
            width = 2 * degree + 1
            cart = basis.unsqueeze(0).expand(batch, -1, -1).reshape(
                batch, *([3] * degree), width
            )
            rotated = cart
            for axis in range(1, degree + 1):
                rotated = rotated.movedim(axis, -1)
                rotated = torch.einsum(
                    "bij,b...j->b...i", rotation, rotated
                ).movedim(-1, axis)
            rotated = rotated.reshape(batch, 3**degree, width)
            blocks.append(torch.einsum("pi,bpj->bij", basis, rotated))
        return self._pack(blocks)

    @staticmethod
    def _pack(blocks: list[torch.Tensor]) -> torch.Tensor:
        batch = blocks[0].shape[0]
        size = sum(block.shape[-1] for block in blocks)
        out = blocks[0].new_zeros(batch, size, size)
        offset = 0
        for block in blocks:
            width = block.shape[-1]
            out[:, offset : offset + width, offset : offset + width] = block
            offset += width
        return out

    def _recursive(self, rotation: torch.Tensor) -> torch.Tensor:
        batch = rotation.shape[0]
        blocks = [rotation.new_ones(batch, 1, 1)]
        if self.lmax >= 1:
            blocks.append(rotation)
        for degree in range(2, self.lmax + 1):
            cg = getattr(self, f"cg_{degree}").to(rotation)
            left = torch.einsum("abm,eac->ebmc", cg, blocks[1])
            left = torch.einsum("ebmc,ebd->emcd", left, blocks[-1])
            block = torch.einsum("emcd,cdn->emn", left, cg)
            blocks.append(block * (2 * degree + 1))
        return self._pack(blocks)

    def get_wigner(
        self, edge_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = init_edge_rotation(edge_vector)
        full = (
            self._direct(rotation)
            if self.wigner_type == "direct"
            else self._recursive(rotation)
        )
        local = torch.einsum("mi,nij->nmj", self.selector.to(full), full)
        inverse = local.transpose(1, 2).contiguous()
        inverse = inverse * self.inverse_rescale.to(inverse)
        return local, inverse


class SphericalCartesianTransform(nn.Module):
    """Convert the model's ICT list to/from l-major real spherical features."""

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = int(lmax)
        for degree in range(lmax + 1):
            self.register_buffer(
                f"basis_{degree}",
                ictd_path_matrix(degree, degree),
                persistent=False,
            )

    @property
    def num_components(self) -> int:
        return (self.lmax + 1) ** 2

    def to_spherical(self, feats: list[torch.Tensor]) -> torch.Tensor:
        fields = []
        for degree, tensor in enumerate(feats):
            basis = getattr(self, f"basis_{degree}").to(tensor)
            flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
            fields.append(torch.einsum("pd,ncp->ndc", basis, flat))
        return torch.cat(fields, dim=1)

    def to_cartesian(self, spherical: torch.Tensor) -> list[torch.Tensor]:
        fields = []
        offset = 0
        for degree in range(self.lmax + 1):
            width = 2 * degree + 1
            field = spherical[:, offset : offset + width]
            offset += width
            basis = getattr(self, f"basis_{degree}").to(field)
            flat = torch.einsum("pd,ndc->ncp", basis, field)
            fields.append(
                flat.reshape(
                    field.shape[0], field.shape[2], *([3] * degree)
                )
            )
        return fields
