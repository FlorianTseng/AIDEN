# Twisted bilayer graphene benchmark

Non-self-consistent-field (NSCF) evaluation of Aiden densities on three
unrelaxed commensurate twisted bilayer graphene cells (paper Sec. 5.3 and
Appendix C.4), compared against ChargE3Net and DFT.

| twist angle | atoms | FFT grid | grid points | SCF mesh | points/segment | bands |
|---:|---:|---|---:|---|---:|---:|
| 21.79 deg | 28 | 96 x 96 x 300 | 2,764,800 | 9 x 9 x 1 | 60 | 80 |
| 13.17 deg | 76 | 160 x 160 x 300 | 7,680,000 | 6 x 6 x 1 | 6 | 168 |
| 9.43 deg | 148 | 224 x 224 x 300 | 15,052,800 | 4 x 4 x 1 | 6 | 312 |

All cells have a 20 A out-of-plane lattice parameter. `structures/` contains
the 9.43 deg POSCAR extracted from the reference CHGCAR; the two smaller cells
follow the same commensurate construction.

## Contents

```
results/                    CSV tables, figures, and plotting scripts
  energy_summary.csv        total energies and per-atom errors per method/cell
  forces_per_atom.csv       per-atom forces and differences (eV/A)
  bands_long.csv            long-format eigenvalues along G-M-K-G
  bands_wide.csv            one row per cell/k-point/band, methods in columns
  band_kpoints.csv          path/k-point metadata
  property_metrics.csv      density/charge-conservation + energy metrics
  band_metrics.csv          raw and rigid-shift-aligned band errors
  overview_summary.csv      combined per-method summary
  inference_timing.csv      structure/graph, model-load, full-grid, export times
  timings_detailed.csv      VASP stage wall times
  tbg-band.*, tbg-summary.* figures of the band and summary comparisons
  view-res/                 additional per-figure exports
metrics-sample03.json       inference metrics of the 9.43 deg prediction job
```

## Method

The converged PBE-SCF densities, energies, and forces are the references. Each
model predicts the complete density grid without charge renormalization; the
model smooth density is supplemented with the one-center PAW data of the
matched reference. Fixed-density energy/force and band calculations use
`ICHARG = 11` with a 520 eV cutoff, the carbon PAW-PBE potential (08Apr2002),
Gaussian smearing 0.05 eV, non-spin-polarized fixed-position calculations with
symmetry disabled, `PREC = Accurate`, `LREAL = .FALSE.`, `LASPH = .TRUE.`,
`ADDGRID = .TRUE.`, and `LMAXMIX = 2` for bands. Electronic convergence is
1e-7 eV (energy/force) and 1e-8 eV (bands). The band path is
G-M-K-G in fractional coordinates.

## Headline results

Full-grid inference times (Aiden vs ChargE3Net official per-sample full-grid
inference): 13.27 vs 128.81 s, 28.93 vs 314.06 s, 41.43 vs 634.14 s, i.e.
9.7-15.3x speedups. Aligned band MAEs within E_Fermi +/- 5 eV: 8.42 / 7.41 /
11.47 meV for Aiden vs 27.35 / 10.30 / 12.93 meV for ChargE3Net. Per-atom
energy errors are 0.753 / 0.747 / 0.741 meV/atom for Aiden, about 28 % of the
ChargE3Net errors. See `results/README.txt` for the column documentation and
the paper's Appendix C.4 for the full tables.
