# Aiden equivariance check and density decomposition

This directory is self-contained with respect to case inputs and model data:

- `GaAs_mp-2534_symmetrized.cif`: default zinc-blende GaAs input structure
  (paper Fig. 6)
- `NaCl.cif`, `H2O.cif`: cases for the `rho_init` / `rho_env` decomposition
- `run_equivariance.py`: CIF expansion, three group transforms, inference, and
  comparison; the model comes from the `aiden` package at the repository root
  and the checkpoint defaults to `checkpoints/aiden-pbe.pt`
- `visualize_equivariance_ase.py`: ASE-based 1x4 charge-density isosurface
  figure
- `visualize_density_components_ase.py`: NaCl / H2O decomposition figures

Run the default `32 x 32 x 32` GaAs test:

```bash
./run.sh
```

Use a different grid or device:

```bash
./run.sh --grid 48 48 48 --device cuda --query-block-size 65536
```

Outputs are written under `results-gaas/`: expanded/transformed CIF and POSCAR
files, predicted NPZ and CHGCAR densities, `summary.json`, `REPORT.md`, and a
comparison figure. NPZ densities are flat with x fastest; reconstruct them
using `density.reshape(grid_shape, order="F")`. The shipped `results-gaas/`
contains the paper's 32 x 32 x 32 run.

Create the ASE-based 1x4 charge-density isosurface figure:

```bash
./plot.sh
```

The code-level `SHOW_WORLD_AXES = False` switch in
`visualize_equivariance_ase.py` gives the compact style. To render a second
version with the same large lab-frame axes behind all four crystals:

```bash
./plot.sh --world-axes \
  --output results-gaas/gaas_density_equivariance_1x4_world_axes.png
```

Plotting requires an external ASE installation (`pip install ase`); both PNG
and PDF are written for each invocation.

For the NaCl 2x2 decomposition into the model's initial and
environment-dependent density heads:

```bash
./plot_components.sh --cif NaCl.cif --threads 4
```

This writes an annotation-free `figsize=(2, 2)` figure in the order: atomic
structure, structure + `rho_init`, structure + `rho_env`, and structure + total
density. The component arrays and an exact sum check are saved alongside it in
`results-nacl-components/` (NaCl) or `results-h2o-components/` (H2O, the
shipped run).

The translation is chosen in integer grid steps, so its reference is an exact
periodic array roll. Proper rotation and inversion transform the full Cartesian
lattice and atom positions together; their scalar-density reference therefore
uses the same fractional-grid indices.

Standard CIF cell metrics cannot encode global Cartesian orientation or cell
handedness. The generated CIFs include an `_eqproof_lattice_vector_*` extension
that this script round-trips exactly, and standard POSCAR companions preserve
the same oriented lattice for other tools.
