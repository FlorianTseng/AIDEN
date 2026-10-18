# AIDEN GaAs_mp-2534_symmetrized equivariance test

Result: **E(3)** under the definitions used below.

| transform | determinant | normalized L1 | max/peak | correlation | pass |
|---|---:|---:|---:|---:|:---:|
| translation | 1 | 7.722226e-06 | 3.427457e-05 | 1.000000000 | yes |
| proper_rotation | 1 | 7.273499e-05 | 4.903811e-05 | 0.999999999 | yes |
| inversion | -1 | 9.512631e-07 | 9.408138e-08 | 1.000000000 | yes |

Practical pass threshold: normalized L1 <= 1.000e-04 (0.01% for this run); raw errors are reported so stricter thresholds can be applied.

Definitions: E(3) = translations + O(3); SE(3) = translations + SO(3); O(3) includes proper rotations and inversion/reflections; SO(3) includes only proper rotations.

The translated reference is an exact periodic grid roll. The proper-rotation and inversion tests transform the complete Cartesian lattice and atoms together, so their expected scalar density has the same fractional-grid indices.

CIF note: standard cell lengths/angles do not retain an absolute Cartesian orientation or handedness. Generated CIFs therefore include a documented `_eqproof_lattice_vector_*` loop; the accompanying POSCAR files store the same exact oriented cells in a standard format.

This is a numerical test on this checkpoint and GaAs_mp-2534_symmetrized, not a formal proof for every structure.

ASE 1x4 isosurface figures are provided in both the TBG-like style
(`gaas_density_equivariance_1x4.png/.pdf`) and with a shared world Cartesian
frame (`gaas_density_equivariance_1x4_world_axes.png/.pdf`). The common
isosurface levels are 0.0584910, 0.239610, 0.981566, and 4.02100 e/Angstrom^3.
