# AlMg density-order and local-extrema audit

Reference: the fresh matched PBE-SCF `CHGCAR` from job 23334532. The original
supplied AlMg file is used only for structure and the `216 × 216 × 216` grid.

## Result

The model arrays use the correct VASP grid order. Charge3Net's stored
`rho[x,y,z]` C-order array is flattened in Fortran order before comparison and
before writing CHGCAR, so x varies fastest in the VASP file.

| Charge3Net layout tested | NMAE |
|---|---:|
| correct `rho[x,y,z].ravel(order="F")` | 4.327420% |
| incorrect raw C-order flattening | 71.114638% |
| other axis permutations | 68.3377%–76.4192% |

The large maximum errors are localized at Mg nuclei and are peak-height errors,
not displaced-density errors.

| density | min (e/Å³) | max (e/Å³) | negative-grid fraction | integrated electrons |
|---|---:|---:|---:|---:|
| fresh PBE-SCF | -0.010986 | 15.321525 | 0.024192% | 594.000001 |
| Aiden original | -0.015775 | 18.749634 | 0.022743% | 594.753555 |
| Charge3Net | -0.011304 | 18.403503 | 0.024936% | 593.121205 |

For Aiden, the maximum absolute error is 3.428109 e/Å³ at a point 0.00135 Å
from Mg atom 108. For Charge3Net it is 3.217815 e/Å³ at a point 0.05793 Å from
Mg atom 77. The top ten error points for both methods are all within 0.062 Å of
Mg atoms. At 99% of grid points, the absolute error is at most 0.211174 e/Å³
for Aiden and 0.195668 e/Å³ for Charge3Net.

Conclusion: both predictions are valid inputs for the controlled frozen-density
VASP comparison. Their Mg-core peak-height bias is a real model error that the
energy/force/band/DOS stage should retain rather than normalize away.
