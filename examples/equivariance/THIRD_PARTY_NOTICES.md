# Third-party notices

The scripts in this directory import the Aiden model from the `aiden` package
at the repository root.

`aiden/e3nn/o3.py` (repository root) implements only the real-basis
`wigner_3j` API needed by that code. Its basis convention and
Clebsch--Gordan construction follow e3nn's MIT-licensed `e3nn/o3/_wigner.py`;
the coefficient routine in e3nn credits the BSD-licensed QuTiP implementation.

The visualization scripts import ASE, an external dependency licensed under
LGPL-2.1-or-later (see https://gitlab.com/ase/ase); it is not redistributed
with this repository.
