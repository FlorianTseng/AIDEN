# AIDEN

**A**tomic-**I**nteraction **D**ensity **E**quivariant **N**etwork for E(3)-equivariant
real-space charge-density learning.

**Paper**: [Neural-Network Solutions to Real-Space Charge Density and Generalization](https://arxiv.org/abs/2609.14906) (arXiv:2609.14906)

AIDEN predicts the ground-state charge density of periodic crystals and finite
molecules as a *continuous* scalar field that can be evaluated at arbitrary
real-space positions. An equivariant Cartesian encoder — a geometry-informed
embedding (GIE), a Cartesian atomic cluster expansion (ACE), and a tensor edge
cluster expansion (TECE) with radial rotary attention (RRA) — produces
irreducible Cartesian tensor features per atom; a continuous low-rank decoder
then contracts these tensors with atom-centered Gaussian-type orbitals at the
requested query positions. Query points never carry hidden states, so the
expensive geometric encoding runs once per structure and is reused for every
density query, from 5,000 training probes up to ten-million-point full FFT
grids.

The predicted density decomposes into a learnable element-dependent one-center
contribution `rho_init` and an environment-dependent low-rank field `rho_env`,
which separates the dominant atomic density structure from the redistribution
induced by the local chemical environment.

## Results overview

Benchmark NMAE of the total density (paper Table 1; ChargE3Net / AIDEN PBE
columns are the L = 3 / L = 4 pair, the AIDEN HSE pair is from-scratch /
fine-tuned):

| dataset | eqDeepDFT | ChargE3Net | NeuralSCF | SCDP | ELECTRA | BOA | AIDEN |
|---|---:|---:|---:|---:|---:|---:|---:|
| ECD PBE (140,646 crystals) | 0.799 | 0.685 / 0.523 | – | – | – | – | **0.621 / 0.454** |
| ECD HSE06 (7,147 crystals) | – | 1.534 | – | – | – | – | **1.431 / 1.301** |
| QM9 (133,885 molecules) | 0.284 | 0.196 | 0.197 | 0.178 | 0.177 | **0.134** | 0.163 |

Zero-shot out-of-distribution transfer (paper Table 2; liquid water, Al54Mg54,
amorphous Si against freshly converged PBE-SCF references): AIDEN reaches
0.95–1.88 % density NMAE with fixed-density energies within 3.5 meV/atom of
SCF, and full-grid inference is 48–76x faster than the evaluated ChargE3Net
implementation and roughly 60–136x faster than the corresponding fixed-density
VASP calculations. On twisted bilayer graphene (21.79 / 13.17 / 9.43 degrees),
AIDEN densities give aligned band MAEs of 8.42 / 7.41 / 11.47 meV within
E_F +/- 5 eV at 9.7–15.3x lower inference cost than ChargE3Net.

## Repository layout

```
aiden/                    model and data package
  model.py                Aiden model: continuous GTO decoder, rho_init + rho_env
  encoder.py              equivariant Cartesian encoder (GIE + Cartesian ACE + TECE)
  cartesian.py            irreducible Cartesian tensor engine (CG-free)
  tece_ops/               tensor edge cluster expansion operators
  e3nn/                   minimal vendored real-basis wigner_3j (see THIRD_PARTY_NOTICES.md)
  data.py                 CHGCAR dataset, periodic graph builders, packed-H5 reader
  fast_infer.py           encoder-cached streaming full-grid inference
  prepare_data.py         pack downloaded ECD CHGCAR archives into training shards
  prepare_qm9_data.py     pack QM9 charge-density HDF5 into isolation-cell shards
  download_ecd_pbe.py     resumable ECDBench archive downloader
train.py                  training entry point
evaluate.py               held-out evaluation (MAE + per-structure NMAE)
predict.py                single-structure prediction
infer_full_grid.py        full-FFT-grid inference for one CHGCAR structure
smoke_test.py             fail-fast data/model/equivariance smoke test
configs/                  reference training configurations of the paper runs
checkpoints/aiden-pbe.pt  PBE-pretrained L=4 checkpoint used for zero-shot transfer
runs/                     training-run registry: configs, full logs, metric tables,
                          and the plotting scripts behind Fig. 2 / 8 / 9
ablation/                 structural ablations of the paper's Table 3 (8 variants)
benchmarks/               inference-speed and parameter-count comparisons vs baselines
examples/equivariance/    E(3)-equivariance check + density-decomposition cases
ood/                      out-of-distribution benchmark (water, AlMg, amorphous Si)
tbg/                      twisted-bilayer-graphene benchmark (bands, DOS, forces)
data/elem-des.csv         element descriptors (133-dim input features)
tests/                    unit tests for the downloader
```

Model weights of every paper run are provided through the repository's
[releases](https://github.com/FlorianTseng/AIDEN/releases); training data are
not included (see below).

## Installation

Python >= 3.10 with PyTorch >= 2.0 is required.

```bash
pip install -r requirements.txt
```

`torch-cluster` is optional; when present it accelerates periodic neighbor
searches. Without it the package falls back to an exact vectorized NumPy
implementation (select with the `AIDEN_GRAPH_BACKEND` environment variable:
`auto`, `torch_cluster`, or `numpy`).

## Quick start

Run the GaAs equivariance check on CPU (no datasets required; downloads
nothing):

```bash
cd examples/equivariance
./run.sh                      # 32 x 32 x 32 grid, writes results-gaas/
./run.sh --grid 16 16 16 --device cpu
```

Predict a complete FFT-grid density for an existing CHGCAR with the released
PBE checkpoint:

```bash
python infer_full_grid.py \
    --system my-system \
    --input path/to/CHGCAR \
    --checkpoint checkpoints/aiden-pbe.pt \
    --elem-csv data/elem-des.csv \
    --output-dir results/my-system \
    --query-block-size 65536 \
    --device cuda
```

The output `full_grid_prediction.npz` stores `density_pred` flat in VASP
CHGCAR order (x fastest). Reconstruct it with

```python
import numpy as np
payload = np.load("results/my-system/full_grid_prediction.npz")
rho = payload["density_pred"].reshape(payload["grid_shape"], order="F")
```

## Data preparation

Training data are **not** included in this repository. The crystal benchmarks
come from the [ECD dataset](https://openreview.net/forum?id=SBCMNc3Mq3)
(PBE: 140,646 VASP PAW-PBE charge densities over 94 elements; HSE06: 7,147
recalculated samples), and the molecular benchmark from the QM9 charge-density
dataset of DeepDFT.

```bash
# 1) download ECD PBE archives (resumable; ~500 GB total)
python -m aiden.download_ecd_pbe --output data/ecd-pbe

# 2) pack CHGCARs into sharded HDF5 with cached periodic atom graphs
python -m aiden.prepare_data --input-root data/ecd-pbe --output-root data/aiden-mp

# 3) QM9: convert the DeepDFT HDF5 files into non-periodic isolation cells
python -m aiden.prepare_qm9_data --input-root data/qm9-h5 --output-root data/aiden-qm9
```

The HSE06 runs use the recalculated ECD HSE06 charge densities, packed the
same way into `data/aiden-hse`. `data/elem-des.csv` defines the
133-dimensional element input (118-dim one-hot + 15 elemental descriptors,
min-max normalized).

## Training

```bash
python train.py \
    --data-root data/aiden-mp \
    --elem-csv data/elem-des.csv \
    --save-dir runs/aiden-pbe-l4 \
    --lmax 4 --mmax 4 \
    --max-steps 2000000
```

Training follows two stages, both handled by `train.py`: the element-indexed
one-center coefficients are optimized alone for 1,000 steps with the radial
correction disabled, after which the complete model is optimized jointly with
the grid-averaged MAE objective (5,000 uniformly sampled grid points per
structure, batch size 12). The encoder and decoder use learning rates
2e-4 / 1e-3 with Adam, cosine annealing, gradient-norm clipping at 0.5, and
EMA weights (decay 0.995) for evaluation. The paper models were trained on a
single NVIDIA A100-40GB.

`configs/` records the exact configurations of the paper runs:

| config | dataset | note |
|---|---|---|
| `pbe-l0.json` ... `pbe-l4.json` | ECD PBE | angular-order sweep, L = 0..4 |
| `qm9-l0.json` ... `qm9-l4.json` | QM9 | angular-order sweep, L = 0..4 |
| `hse-l3-scratch.json` | ECD HSE06 | trained from scratch |
| `hse-l3-finetune.json` | ECD HSE06 | fine-tuned from the PBE L=3 checkpoint |
| `hse-l3-decoder.json` | ECD HSE06 | decoder-only transfer from PBE |
| `hse-zero-shot-*.json/csv` | ECD HSE06 | zero-shot evaluation of the PBE model |

Principal model configuration (paper Appendix C.6): `L = M = 3`
(`lmax = mmax = 3`; the released OOD checkpoint uses `L = M = 4`), 48 channels,
ACE -> TECE interactions, correlation orders (2, 2), 4 attention heads,
`N_B = 8` Bessel radial functions, `r_at = 4.0` A, 100 neighbors, environment
rank `K = 8`, one-center rank `K_0 = 1`, `N_G = 8` Gaussian radial functions,
`r_orb = 3.0` A. The L = 4 model has 4.46 M parameters.

## Evaluation

```bash
python evaluate.py \
    --checkpoint runs/aiden-pbe-l4/best_nmae.pt \
    --data-root data/aiden-mp \
    --elem-csv data/elem-des.csv \
    --split test
```

Reports the density MAE and the per-structure NMAE
`sum|rho_pred - rho_ref| / sum|rho_ref|` used in the paper.

## Reproducing the paper experiments

### Benchmarks (Table 1, Fig. 2, Fig. 8, Fig. 9)

Train with the reference configs above; `runs/` archives every run's config
and full training log plus the evaluation tables and plotting scripts.
Headline validation NMAE of the paper: PBE 0.621 % / 0.454 % (L = 3 / L = 4),
HSE 1.431 % / 1.301 % (scratch / fine-tune), QM9 0.163 %.

### Structural ablations (Table 3)

See `ablation/`. Eight L = 4 variants (interaction ordering, GIE, RRA,
`rho_init`, fixed GTO basis) trained for 100k steps on a fixed
2,000-structure PBE subset with a shared one-center initialization. Against
the full model's 1.132 % validation NMAE: removing `rho_init` costs +35.7 %,
a fixed GTO basis +7.4 %, removing RRA +2.6 %, removing GIE +1.3 %, and
reordering TECE -> ACE +1.3 %.

### Inference speed and parameter counts (Fig. 2a/2b)

See `benchmarks/`. `benchmarks/inference-speed/` benchmarks the
encoder-cached streaming decoder against the conventional per-block path
(numerics verified to 1e-6 NMAE); `benchmarks/parameter-count/` compares
learnable-parameter counts of AIDEN, ChargE3Net, and BOA at L = 0..4
(AIDEN grows ~2.2x per angular order, reaching 4.46 M at L = 4).

### E(3)-equivariance check (Fig. 7)

See `examples/equivariance/`. The default case expands zinc-blende GaAs
(mp-2534) on a 32 x 32 x 32 grid and compares the predicted density under
translation, proper rotation, and inversion; L1 deviations stay at
1e-6–1e-5 of the total cell charge (7.72e-6 / 7.27e-5 / 9.51e-7). The same
directory also decomposes the predicted density of NaCl and H2O into
`rho_init` and `rho_env`.

### Out-of-distribution systems (Table 2, Fig. 3)

See `ood/`. Zero-shot inference of the PBE-pretrained checkpoint on liquid
water (192 atoms), Al54Mg54 (108 atoms), and amorphous silicon (64 atoms),
evaluated against freshly converged PBE-SCF references: density NMAE,
full-grid inference time, fixed-density energies and forces (VASP
`ICHARG=11`), band/DOS metrics, and atom-rich cross-sections. VASP
inputs/outputs and the full-grid prediction archives are included.

### Twisted bilayer graphene (Fig. 4, Fig. 10, Tables 5-9)

See `tbg/`. Twist angles 21.79 / 13.17 / 9.43 degrees (28 / 76 / 148 atoms).
The directory contains the eigenvalue, energy, force, timing, and DOS tables
behind the NSCF band-structure and charge-conservation comparison against
ChargE3Net and DFT, plus the plotting scripts.

## Released checkpoint

`checkpoints/aiden-pbe.pt` is the PBE-pretrained L = 4 model
(4.46 M parameters, validation NMAE 0.454 %) used for all zero-shot OOD and
TBG evaluations. It stores EMA weights under `ema_model`, which the inference
entry points select by default. Checkpoints of the remaining paper runs
(PBE/QM9 L = 0..4, HSE variants) are attached to the repository releases.

## Citation

If you use AIDEN in your work, please cite:

```bibtex
@misc{zeng2026neuralnetworksolutionsrealspacecharge,
      title={Neural-Network Solutions to Real-Space Charge Density and Generalization},
      author={Yuxuan Zeng and Tong Xie and Taoyuze Lv and Zhicheng Zhong},
      year={2026},
      eprint={2609.14906},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2609.14906}
}
```

## Third-party components

`aiden/e3nn` contains a minimal real-basis `wigner_3j` implementation whose
convention follows e3nn (MIT); see `THIRD_PARTY_NOTICES.md`. The equivariance
visualization scripts depend on ASE (LGPL-2.1-or-later), installed as a pip
dependency. Baseline comparisons (ChargE3Net, BOA) use the official
third-party repositories, cloned read-only where needed.
