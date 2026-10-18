# OOD charge-density and fixed-density property benchmark

The fresh matched PBE-SCF calculation is the sole reference for density, energy, force, bands, and DOS; supplied CHGCAR/HDF5 files provide structure and grid only. All density NMAEs are `sum(|prediction-reference|)/sum(|reference|)`. Aiden downstream calculations use the original full inference. All Aiden, Charge3Net, and SAD candidate densities are evaluated with identical `ICHARG=11` and k-point settings. The reported forces are frozen-density, Harris-corrected approximate forces rather than self-consistent ground-state forces. Band errors are reported both raw and after one rigid energy shift; DOS primary errors align each spectrum to its own Fermi level.

Selected samples: water, almg, asi_011. Exactly one a-Si frame is analyzed; these are not 108-frame ensemble averages.

Timing scope: model timings are GPU inference. SAD timing is the complete fixed-density VASP call including energy/force diagonalization, not isolated density construction.

## water

| method | density NMAE (%) | inference s mean | |ΔE| meV/atom mean | force MAE eV/Å mean | band MAE eV mean | DOS NMAE mean |
|---|---:|---:|---:|---:|---:|---:|
| aiden_original | 1.87636 | 70.6592 | 3.44361 | 0.114674 | 0.0406192 | 0.380214 |
| aiden_fast | 1.87641 | 23.6989 |  |  |  |  |
| charge3net | 1.78798 | 1286.43 | 6.44272 | 0.125378 | 0.0339446 | 0.319672 |
| sad | 12.3527 | 1433 | 485.445 | 1.11744 |  |  |

## almg

| method | density NMAE (%) | inference s mean | |ΔE| meV/atom mean | force MAE eV/Å mean | band MAE eV mean | DOS NMAE mean |
|---|---:|---:|---:|---:|---:|---:|
| aiden_original | 4.30036 | 29.3137 | 262.809 | 0.103979 | 0.0202955 | 0.0526395 |
| aiden_fast | 4.30039 | 13.3956 |  |  |  |  |
| charge3net | 4.32742 | 936.429 | 276.07 | 0.0357955 | 0.00729015 | 0.0228371 |
| sad | 4.80572 | 1344 | 49.6434 | 0.157755 |  |  |

## asi

| method | density NMAE (%) | inference s mean | |ΔE| meV/atom mean | force MAE eV/Å mean | band MAE eV mean | DOS NMAE mean |
|---|---:|---:|---:|---:|---:|---:|
| aiden_original | 1.32156 | 8.65291 | 3.21298 | 0.0408691 | 0.00698434 | 0.0202465 |
| aiden_fast | 1.32153 | 4.34148 |  |  |  |  |
| charge3net | 1.6391 | 328.873 | 4.77398 | 0.0408964 | 0.00803204 | 0.024403 |
| sad | 9.99954 | 307 | 82.1065 | 0.127783 |  |  |

Failures recorded: 0. Pending comparisons: 0.

Machine-readable tables: `density_and_timing.csv`, `energy_force_metrics.csv`, `band_metrics.csv`, `dos_metrics.csv`, and `summary.json`.
