# Third-party notices

`aiden/e3nn/o3.py` implements only the real-basis `wigner_3j` API needed by the
TECE operators. Its basis convention and Clebsch--Gordan construction follow
e3nn's MIT-licensed `e3nn/o3/_wigner.py`; the coefficient routine in e3nn
credits the BSD-licensed QuTiP implementation.

`aiden/tece_ops` follows the tensor edge cluster expansion construction of
Xu, Xie, and Hu, "Edge cluster expansion with radial rotary attention for
interatomic potentials" (arXiv:2607.10664). The equivariant Cartesian tensor
engine in `aiden/cartesian.py` follows the irreducible Cartesian tensor
construction of Xu, Xie, and Hu, "Spectral/Spatial Tensor Atomic Cluster
Expansion with Universal Embeddings in Cartesian Space"
(arXiv:2509.14961).

The equivariance visualization scripts depend on ASE (LGPL-2.1-or-later),
installed separately as a pip dependency and not redistributed here.
