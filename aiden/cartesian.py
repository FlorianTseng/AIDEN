# cartesian.py
# =====================================================================================
# Generic, Clebsch-Gordan-FREE irreducible Cartesian tensor (ICT) engine.
#
# This is the genuine TACE backbone primitive set (Xu, Xie, Hu, "TACE", arXiv:2509.14961):
#   * features are irreducible Cartesian tensors = fully symmetric, traceless rank-l
#     tensors in R^3 (the "highest weight" components, TACE Sec. 2.3.6 / Eqn 10),
#   * angular coupling is the Cartesian tensor product / contraction (TACE Eqn 4)
#     followed by projection back to the highest weight via symmetrize + detrace,
#   * NO spherical harmonics, NO Clebsch-Gordan coefficients, NO e3nn.
#
# A weight-l feature is stored as a torch.Tensor of shape
#       (*batch, C, 3, 3, ..., 3)          # l trailing Cartesian axes
# where C is the channel axis (immediately before the l Cartesian axes) and *batch are
# leading axes (atoms / edges / probes). l = 0 -> (*batch, C); l = 1 -> (*batch, C, 3); ...
#
# Everything here is written generically in l. The only l-specific piece is `detrace`,
# which is given in closed form for l <= 4. Extending beyond l = 4 requires the
# general Applequist 1989 detracer series (TACE ref [36]). The l! permutation
# symmetrization is the practical cap on lmax, so this implementation stops at 4.
# =====================================================================================
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import List, Tuple

import torch

# letters used to build einsum subscripts for Cartesian indices (avoid 'e','c').
_CART_LETTERS = "fghijklmnopqrstuvwxyzab"


def double_factorial(n: int) -> float:
    """(2l-1)!! style double factorial with the convention (-1)!! = 1."""
    if n <= 0:
        return 1.0
    out = 1.0
    while n > 0:
        out *= n
        n -= 2
    return out


def harmonic_norm_constant(l: int) -> float:
    """TACE Eqn 10 prefactor (2l-1)!! / l! ."""
    return double_factorial(2 * l - 1) / math.factorial(l)


@lru_cache(maxsize=None)
def _permutations(l: int) -> Tuple[Tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(l)))


def symmetrize(t: torch.Tensor, l: int) -> torch.Tensor:
    """Average a tensor over all l! permutations of its trailing l Cartesian axes."""
    if l <= 1:
        return t
    ndim = t.dim()
    cart = list(range(ndim - l, ndim))
    prefix = list(range(ndim - l))
    perms = _permutations(l)
    acc = t.new_zeros(t.shape)
    for p in perms:
        order = prefix + [cart[i] for i in p]
        acc = acc + t.permute(*order)
    return acc / len(perms)


def _trace_last_two(t: torch.Tensor) -> torch.Tensor:
    """Contract the last two Cartesian axes (sum over the diagonal)."""
    return t.diagonal(dim1=-2, dim2=-1).sum(-1)


def detrace(t: torch.Tensor, l: int) -> torch.Tensor:
    """
    Project a rank-l tensor onto its fully-symmetric traceless (= weight-l) part.

    Closed form for l <= 4. The result T satisfies
    T = symmetric and every pairwise contraction Sum_j T[..., j, j, ...] = 0.
    """
    if l <= 1:
        return t
    s = symmetrize(t, l)

    if l == 2:
        # T_ij = S_ij - (1/3) d_ij tr(S)
        eye = torch.eye(3, dtype=s.dtype, device=s.device)
        tr = _trace_last_two(s)  # (*batch, C)
        return s - (1.0 / 3.0) * tr[..., None, None] * eye

    if l == 3:
        # T_ijk = S_ijk - (1/5)(d_ij v_k + d_ik v_j + d_jk v_i),  v_k = S_kll
        eye = torch.eye(3, dtype=s.dtype, device=s.device)
        v = s.diagonal(dim1=-2, dim2=-1).sum(-1)  # (*batch, C, 3) ; v_i = S_i j j
        term = (
            torch.einsum("ij,...k->...ijk", eye, v)
            + torch.einsum("ik,...j->...ijk", eye, v)
            + torch.einsum("jk,...i->...ijk", eye, v)
        )
        return s - 0.2 * term

    if l == 4:
        # In three dimensions, for symmetric S_ijkl,
        #
        # STF(S) = S - (1/7) sym_6(delta_ij A_kl)
        #            + (1/35) sym_3(delta_ij delta_kl) B,
        # A_ij = S_ijkk and B = A_iijj.  The coefficients make every
        # pairwise trace vanish while retaining full permutation symmetry.
        eye = torch.eye(3, dtype=s.dtype, device=s.device)
        pair_trace = s.diagonal(dim1=-2, dim2=-1).sum(-1)
        double_trace = pair_trace.diagonal(dim1=-2, dim2=-1).sum(-1)
        single_delta = (
            torch.einsum("ij,...kl->...ijkl", eye, pair_trace)
            + torch.einsum("ik,...jl->...ijkl", eye, pair_trace)
            + torch.einsum("il,...jk->...ijkl", eye, pair_trace)
            + torch.einsum("jk,...il->...ijkl", eye, pair_trace)
            + torch.einsum("jl,...ik->...ijkl", eye, pair_trace)
            + torch.einsum("kl,...ij->...ijkl", eye, pair_trace)
        )
        double_delta = double_trace[..., None, None, None, None] * (
            torch.einsum("ij,kl->ijkl", eye, eye)
            + torch.einsum("ik,jl->ijkl", eye, eye)
            + torch.einsum("il,jk->ijkl", eye, eye)
        )
        return s - single_delta / 7.0 + double_delta / 35.0

    raise NotImplementedError(
        f"detrace is implemented in closed form for l <= 4 (got l={l}). "
        "Extend with the Applequist 1989 detracer series (TACE ref [36]) for higher l."
    )


def cartesian_harmonic(unit_vec: torch.Tensor, l: int) -> torch.Tensor:
    """
    Cartesian analogue of the spherical harmonic Y_l (TACE Eqn 10):
        Y_l(u) = ((2l-1)!!/l!) * detrace( u (x) u (x) ... (x) u )   [l factors]
    unit_vec: (..., 3), assumed (approximately) unit length.
    returns:  (..., 3, 3, ..., 3)  with l trailing Cartesian axes (l=0 -> (...,)).
    """
    if l == 0:
        return unit_vec.new_ones(unit_vec.shape[:-1])
    out = _outer_power(unit_vec, l)
    return harmonic_norm_constant(l) * detrace(out, l)


def _outer_power(v: torch.Tensor, l: int) -> torch.Tensor:
    """v (x) v (x) ... (x) v  with l factors.  v: (..., 3) -> (..., 3,...,3)."""
    out = v
    for _ in range(l - 1):
        out = out.unsqueeze(-1) * v.reshape(v.shape[:-1] + (1,) * (out.dim() - v.dim() + 1) + (3,))
    return out


def couple(a: torch.Tensor, b: torch.Tensor, l1: int, l2: int, l3: int) -> torch.Tensor:
    """
    CG-free Cartesian tensor product / contraction (TACE Eqn 4) projected to weight l3.

    a: (*batch, C, [3]*l1)   b: (*batch, C, [3]*l2)   ->  (*batch, C, [3]*l3)
    Contracts k = (l1+l2-l3)/2 index pairs, then symmetrize + detrace to weight l3.
    Both inputs carry the SAME leading (*batch, C) layout.
    """
    k2 = l1 + l2 - l3
    if k2 < 0 or k2 % 2 != 0:
        raise ValueError(f"invalid Cartesian path l1={l1}, l2={l2}, l3={l3}")
    k = k2 // 2
    if k > min(l1, l2):
        raise ValueError(f"too many contractions for path l1={l1}, l2={l2}, l3={l3}")

    a_idx = _CART_LETTERS[:l1]
    b_idx = list(_CART_LETTERS[l1:l1 + l2])
    # contract the last k axes of a with the first k axes of b
    for t in range(k):
        b_idx[t] = a_idx[l1 - k + t]
    free_a = a_idx[: l1 - k]
    free_b = "".join(b_idx[k:])
    out_idx = free_a + free_b
    sub = f"ec{a_idx},ec{''.join(b_idx)}->ec{out_idx}"

    # Flatten all leading dims except channel into a single "e" axis for einsum.
    a2, lead = _as_ec(a, l1)
    b2, _ = _as_ec(b, l2)
    out = torch.einsum(sub, a2, b2)
    scale = (1.0 / math.sqrt(3.0)) ** k
    out = out * scale
    out = out.reshape(lead + (out.shape[1],) + out.shape[2:])  # restore (*batch, C, [3]*l3)
    return detrace(symmetrize(out, l3), l3)


def _as_ec(t: torch.Tensor, l: int) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Reshape (*batch, C, [3]*l) -> (E, C, [3]*l) collapsing batch dims to a single E."""
    cart = t.shape[t.dim() - l:] if l > 0 else ()
    c = t.shape[t.dim() - l - 1]
    lead = t.shape[: t.dim() - l - 1]
    e = 1
    for s in lead:
        e *= s
    return t.reshape((e, c) + tuple(cart)), lead


def frob_norm(t: torch.Tensor, l: int, keepdim_cart: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """Rotation-invariant per-channel Frobenius norm of a weight-l tensor.

    Returns (*batch, C). For l == 0 this is |value|.

    The `eps` under the square root is what keeps the *gradient* finite when the
    feature is exactly zero: d/dt sqrt(sum t^2) = t / sqrt(sum t^2) is 0/0 = NaN at
    t = 0 without it. Zero weight-l (l>=1) features occur in practice for isolated
    atoms (no neighbors inside the cutoff), so a bare sqrt poisons training with NaN
    gradients even though the forward value is a clean 0.
    """
    if l == 0:
        return t.abs()
    axes = tuple(range(t.dim() - l, t.dim()))
    return (t.pow(2).sum(dim=axes).clamp_min(0) + eps).sqrt()


@lru_cache(maxsize=None)
def valid_paths(lmax: int) -> Tuple[Tuple[int, int, int], ...]:
    """All (l1, l2, l3) with 0<=li<=lmax and integer k=(l1+l2-l3)/2 in [0, min(l1,l2)]."""
    paths: List[Tuple[int, int, int]] = []
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l1 - l2), min(lmax, l1 + l2) + 1):
                if (l1 + l2 - l3) % 2 == 0:
                    paths.append((l1, l2, l3))
    return tuple(paths)


def rotate_tensor(t: torch.Tensor, R: torch.Tensor, l: int) -> torch.Tensor:
    """Apply a rotation R to every Cartesian axis of a weight-l tensor (for tests)."""
    if l == 0:
        return t
    out = t
    ndim = out.dim()
    for axis in range(ndim - l, ndim):
        out = torch.tensordot(out, R, dims=([axis], [1]))
        # tensordot moves the contracted axis to the end; move it back
        out = out.movedim(-1, axis)
    return out


# =====================================================================================
# Stand-alone self-test of the algebra (run `python cartesian.py`).
# =====================================================================================
def _random_rotation(dtype=torch.float64) -> torch.Tensor:
    a = torch.randn(3, 3, dtype=dtype)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # fix signs
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _selftest() -> None:
    torch.manual_seed(0)
    dt = torch.float64

    # 1) harmonics are traceless + correctly normalized + equivariant
    u = torch.randn(7, 3, dtype=dt)
    u = u / u.norm(dim=-1, keepdim=True)
    R = _random_rotation(dt)
    for l in range(0, 5):
        Y = cartesian_harmonic(u, l)
        if l >= 2:
            tr = Y.diagonal(dim1=-2, dim2=-1).sum(-1) if l == 2 else _trace_last_two(symmetrize(Y, l))
            assert tr.abs().max() < 1e-9, f"harmonic l={l} not traceless: {tr.abs().max()}"
        # equivariance: Y(Ru) == rotate(Y(u))
        Yr = cartesian_harmonic((u @ R.T), l)
        Yrot = rotate_tensor(Y, R, l)
        assert (Yr - Yrot).abs().max() < 1e-9, f"harmonic l={l} not equivariant"

    # 2) couple is equivariant and produces traceless weight-l3 outputs
    E, C = 5, 4
    for (l1, l2, l3) in valid_paths(4):
        a = torch.randn((E, C) + (3,) * l1, dtype=dt)
        b = torch.randn((E, C) + (3,) * l2, dtype=dt)
        a = detrace(symmetrize(a, l1), l1)
        b = detrace(symmetrize(b, l2), l2)
        out = couple(a, b, l1, l2, l3)
        # equivariance
        ar = rotate_tensor(a, R, l1)
        br = rotate_tensor(b, R, l2)
        out_r = couple(ar, br, l1, l2, l3)
        out_rot = rotate_tensor(out, R, l3)
        err = (out_r - out_rot).abs().max().item()
        assert err < 1e-8, f"couple path {(l1, l2, l3)} not equivariant: {err}"
        # tracelessness of output
        if l3 >= 2:
            s = symmetrize(out, l3)
            tr = s.diagonal(dim1=-2, dim2=-1).sum(-1)
            assert tr.abs().max() < 1e-8, f"couple {(l1,l2,l3)} output not traceless"

    # 3) couple bilinearity in a
    l1, l2, l3 = 1, 2, 1
    a1 = detrace(symmetrize(torch.randn((E, C, 3), dtype=dt), 1), 1)
    a2 = detrace(symmetrize(torch.randn((E, C, 3), dtype=dt), 1), 1)
    b = detrace(symmetrize(torch.randn((E, C, 3, 3), dtype=dt), 2), 2)
    lhs = couple(a1 + 2.5 * a2, b, l1, l2, l3)
    rhs = couple(a1, b, l1, l2, l3) + 2.5 * couple(a2, b, l1, l2, l3)
    assert (lhs - rhs).abs().max() < 1e-9, "couple not bilinear"

    # 4) frob_norm invariance
    x = detrace(symmetrize(torch.randn((E, C, 3, 3), dtype=dt), 2), 2)
    n0 = frob_norm(x, 2)
    n1 = frob_norm(rotate_tensor(x, R, 2), 2)
    assert (n0 - n1).abs().max() < 1e-9, "frob_norm not invariant"

    print("cartesian.py self-test passed:")
    print(
        f"  paths(lmax=2) = {len(valid_paths(2))}, "
        f"paths(lmax=3) = {len(valid_paths(3))}, "
        f"paths(lmax=4) = {len(valid_paths(4))}"
    )
    print(
        "  harmonic norm consts: "
        + ", ".join(
            f"l{l}={harmonic_norm_constant(l):.3f}"
            for l in range(5)
        )
    )


if __name__ == "__main__":
    _selftest()
