"""AIDEN: Atomic-Interaction Density Equivariant Network.

E(3)-equivariant charge-density learning with irreducible Cartesian tensor
encoding and a continuous GTO density decoder.
"""

from .model import (
    Aiden,
    AidenOutput,
    ContinuousDensityDecoder,
    ElementInitialDensity,
    EvenTemperedCartesianGTO,
    smooth_absolute,
)

__all__ = [
    "Aiden",
    "AidenOutput",
    "ContinuousDensityDecoder",
    "ElementInitialDensity",
    "EvenTemperedCartesianGTO",
    "smooth_absolute",
]
