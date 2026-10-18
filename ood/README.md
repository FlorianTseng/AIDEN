# Out-of-distribution benchmark: liquid water, AlMg, and amorphous silicon

Zero-shot evaluation of the PBE-pretrained Aiden checkpoint
(`checkpoints/aiden-pbe.pt`) on three out-of-distribution structures,
following the protocol of the paper (Sec. 5.3 and Appendix C.5):

| system | structure | atoms | FFT grid |
|---|---|---:|---|
| water | 64 H2O molecules | 192 | 216 x 216 x 216 |
| almg | Al54Mg54 substitutional alloy | 108 | 216 x 216 x 216 |
| a-Si | amorphous-silicon frame 11 | 64 | 168 x 168 x 168 |

The supplied CHGCAR/HDF5 files only provide structure, cell, and grid; every
density, energy, and force reference is a freshly converged PBE-SCF
calculation. Structures are in `structures/` (`water.POSCAR`,
`almg.POSCAR`, `asi_011.POSCAR` extracted from the reference inputs).

## Contents

```
structures/     water / almg / a-Si POSCARs (plus a water CIF)
vasp/           VASP inputs and small outputs, per system and per job:
                reference/   PBE-SCF reference + ICHARG=12 SAD density jobs
                aiden-energy-force/          ICHARG=11 frozen-density job
                charge3net-energy-force/     ICHARG=11 frozen-density job
                (CHGCAR/DOSCAR/POTCAR are excluded for size/licensing)
predictions/    full-grid densities used by the property jobs
                aiden/full_grid_prediction.npz       (density_pred, CHGCAR order)
                charge3net/full_grid_prediction.npy  (official MP checkpoint)
analysis/       aggregated metric tables, atom-rich slices, band/DOS metrics,
                NSCF sensitivity study, plots, and REPORT.md
ood-bar-summary.*, ood-density-summary.*   summary figures (plot-ood-sum.py)
```

## Reproducing

1. Full-grid inference from the input CHGCAR (the reference SCF CHGCAR or any
   structure-matched CHGCAR with the same FFT grid):

   ```bash
   python infer_full_grid.py \
       --system water \
       --input water.CHGCAR \
       --checkpoint checkpoints/aiden-pbe.pt \
       --elem-csv data/elem-des.csv \
       --output-dir ood/predictions/water/aiden \
       --expected-atoms 192 --expected-grid-shape 216 216 216
   ```

2. Write the predicted smooth density into a CHGCAR and append the one-center
   PAW augmentation data of the matched PBE-SCF reference CHGCAR (the model
   predicts the smooth valence density; VASP expects the augmented density).

3. Evaluate energies and forces with the density frozen
   (`ICHARG=11`); the exact INCAR/KPOINTS used for every job are under
   `vasp/`. Common settings: PAW-PBE potentials, ENCUT = 520 eV,
   PREC = Accurate, EDIFF = 1e-8 eV, 2 x 2 x 2 Monkhorst-Pack mesh, Gaussian
   smearing 0.05 eV, ISPIN = 1, ISYM = 0, fixed positions. SAD baselines use
   `ICHARG=12`. ChargE3Net predictions come from the official Materials
   Project checkpoint with its full-grid inference.

4. Densities are compared without charge renormalization; the NMAE is
   `sum|rho_cand - rho_DFT| / sum|rho_DFT|` over the complete grid.

## Headline results (paper Table 2)

| system | method | density NMAE [%] | inference [s] | \|dE\| [meV/atom] | force MAE [eV/A] |
|---|---|---:|---:|---:|---:|
| water | Aiden | 1.8764 | 23.70 | 3.444 | 0.11467 |
| water | ChargE3Net | 1.7880 | 1286.43 | 6.443 | 0.12538 |
| water | SAD | 12.3527 | 1433 (dagger) | 485.445 | 1.11744 |
| almg | Aiden | 0.9542 | 10.17 | 3.484 | 0.04977 |
| almg | ChargE3Net | 1.1928 | 486.82 | 0.938 | 0.02451 |
| almg | SAD | 10.2581 | 1382 (dagger) | 53.269 | 0.14645 |
| a-Si | Aiden | 1.3215 | 4.34 | 3.213 | 0.04087 |
| a-Si | ChargE3Net | 1.6391 | 328.87 | 4.774 | 0.04090 |
| a-Si | SAD | 9.9995 | 307 (dagger) | 82.107 | 0.12778 |

(dagger) SAD times are complete fixed-density VASP wall times including
diagonalization. Model timings are CUDA-synchronized GPU inference after
warm-up, excluding environment loading. See `analysis/REPORT.md` for the full
tables, the band/DOS metrics, and atom-rich cross-sections.
