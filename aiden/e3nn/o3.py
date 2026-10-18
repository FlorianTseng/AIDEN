"""Small, API-compatible implementation of ``e3nn.o3.wigner_3j``.

The real-basis convention and SU(2) Clebsch--Gordan construction follow the
e3nn project implementation.  AIDEN only requests integer orders up to four,
so the rest of e3nn is intentionally not vendored.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from math import factorial

import torch


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"wigner_3j only supports float32/float64, got {dtype}")


def _change_basis_real_to_complex(
    degree: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    q = torch.zeros(
        (2 * degree + 1, 2 * degree + 1),
        dtype=torch.complex128,
        device=device,
    )
    root_two = 2.0**0.5
    for m in range(-degree, 0):
        q[degree + m, degree + abs(m)] = 1.0 / root_two
        q[degree + m, degree - abs(m)] = -1j / root_two
    q[degree, degree] = 1.0
    for m in range(1, degree + 1):
        q[degree + m, degree + abs(m)] = (-1) ** m / root_two
        q[degree + m, degree - abs(m)] = 1j * (-1) ** m / root_two
    q = (-1j) ** degree * q
    return q.to(dtype=_complex_dtype(dtype), device=device).contiguous()


def _factorial(value: float) -> int:
    if value != round(value):
        raise ValueError(f"factorial argument must be integral, got {value}")
    return factorial(round(value))


def _su2_coefficient(
    idx1: tuple[float, float],
    idx2: tuple[float, float],
    idx3: tuple[float, float],
) -> float:
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3
    if m3 != m1 + m2:
        return 0.0

    vmin = int(max(-j1 + j2 + m3, -j1 + m1, 0))
    vmax = int(min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3))
    prefactor = (
        (2.0 * j3 + 1.0)
        * Fraction(
            _factorial(j3 + j1 - j2)
            * _factorial(j3 - j1 + j2)
            * _factorial(j1 + j2 - j3)
            * _factorial(j3 + m3)
            * _factorial(j3 - m3),
            _factorial(j1 + j2 + j3 + 1)
            * _factorial(j1 - m1)
            * _factorial(j1 + m1)
            * _factorial(j2 - m2)
            * _factorial(j2 + m2),
        )
    ) ** 0.5

    total = Fraction(0)
    for value in range(vmin, vmax + 1):
        total += (-1) ** int(value + j2 + m2) * Fraction(
            _factorial(j2 + j3 + m1 - value)
            * _factorial(j1 - m1 + value),
            _factorial(value)
            * _factorial(j3 - j1 + j2 - value)
            * _factorial(j3 + m3 - value)
            * _factorial(value + j1 - j2 - m3),
        )
    return float(prefactor * total)


def _su2_clebsch_gordan(j1: int, j2: int, j3: int) -> torch.Tensor:
    matrix = torch.zeros(
        (2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1), dtype=torch.float64
    )
    if 2 * j3 not in range(2 * abs(j1 - j2), 2 * (j1 + j2) + 1, 2):
        return matrix
    for m1_twice in range(-2 * j1, 2 * j1 + 1, 2):
        m1 = m1_twice / 2
        for m2_twice in range(-2 * j2, 2 * j2 + 1, 2):
            m2 = m2_twice / 2
            if abs(m1 + m2) <= j3:
                matrix[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)] = (
                    _su2_coefficient((j1, m1), (j2, m2), (j3, m1 + m2))
                )
    return matrix


@functools.lru_cache(maxsize=None)
def _so3_clebsch_gordan(l1: int, l2: int, l3: int) -> torch.Tensor:
    q1 = _change_basis_real_to_complex(l1)
    q2 = _change_basis_real_to_complex(l2)
    q3 = _change_basis_real_to_complex(l3)
    coefficient = _su2_clebsch_gordan(l1, l2, l3).to(torch.complex128)
    coefficient = torch.einsum(
        "ij,kl,mn,ikn->jlm", q1, q2, torch.conj(q3.T), coefficient
    )
    if not torch.all(torch.abs(torch.imag(coefficient)) < 1.0e-5):
        raise RuntimeError("Clebsch--Gordan transform unexpectedly became complex")
    real = torch.real(coefficient)
    return (real / torch.linalg.norm(real)).contiguous()


def wigner_3j(
    l1: int,
    l2: int,
    l3: int,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return normalized real-basis Wigner 3-j coefficients."""
    if not all(isinstance(value, int) for value in (l1, l2, l3)):
        raise TypeError("angular orders must be integers")
    if not abs(l2 - l3) <= l1 <= l2 + l3:
        raise ValueError(f"invalid angular coupling ({l1}, {l2}, {l3})")
    if dtype is None:
        dtype = torch.get_default_dtype()
    return _so3_clebsch_gordan(l1, l2, l3).to(
        dtype=dtype, device=device, copy=True
    )


__all__ = ["wigner_3j"]
