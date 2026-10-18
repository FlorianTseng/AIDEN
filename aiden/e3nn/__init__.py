"""Minimal vendored e3nn compatibility layer required by AIDEN inference.

Only :func:`e3nn.o3.wigner_3j` is needed by the copied model source.  Keeping
this small implementation beside the model makes CPU-only inference portable
without changing the checkpoint's numerical conventions.
"""

from . import o3

__all__ = ["o3"]
