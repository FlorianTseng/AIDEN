# AIDEN

**A**tomic-**I**nteraction **D**ensity **E**quivariant **N**etwork for E(3)-equivariant
real-space charge-density learning.

**Paper**: [Neural-Network Solutions to Real-Space Charge Density and Generalization](https://arxiv.org/abs/2609.14906) (arXiv:2609.14906)

AIDEN predicts the ground-state charge density of periodic crystals and finite
molecules as a continuous scalar field, evaluated at arbitrary real-space
positions. On the paper's benchmarks it reaches 0.45 % density NMAE on ECD PBE
(L = 4), 1.30 % on ECD HSE06 after fine-tuning, and 0.16 % on QM9, and it
transfers zero-shot to out-of-distribution systems (water, AlMg, amorphous Si,
twisted bilayer graphene) with full-grid inference tens of times faster than
the evaluated baselines. Architecture and experiment details are in the paper.

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

Run the GaAs equivariance check on CPU (no datasets required):

```bash
cd examples/equivariance
./run.sh                      # 32 x 32 x 32 grid, writes results-gaas/
./run.sh --grid 16 16 16 --device cpu
```

Fail-fast check of data pipeline, model forward/backward, and equivariance:

```bash
python smoke_test.py
```

## Using the pretrained checkpoints

`checkpoints/aiden-pbe.pt` (in this repository) is the PBE-pretrained L = 4
model — 4.46 M parameters, 0.454 % validation NMAE — used for all zero-shot
OOD and TBG evaluations in the paper. Checkpoints of every other paper run
(PBE/QM9 at L = 0..4, HSE variants) are downloadable from the
[releases](https://github.com/FlorianTseng/AIDEN/releases); each stores the
full training state with EMA weights under `ema_model`, which the inference
entry points select by default.

### Predict the charge density of a new structure

Full FFT-grid prediction for a structure supplied as a CHGCAR (the file only
provides structure, cell, and grid — no reference density is needed for the
prediction itself):

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
CHGCAR order (x fastest); reconstruct the 3-D array with

```python
import numpy as np
payload = np.load("results/my-system/full_grid_prediction.npz")
rho = payload["density_pred"].reshape(payload["grid_shape"], order="F")
```

If you only have a structure file (POSCAR/cif), first wrap it in a CHGCAR
with the FFT grid you want — for example with ASE:

```python
import io
import numpy as np
from ase.io import read, write

atoms = read("structure.cif")
grid = (120, 120, 120)                     # your chosen FFT grid

buf = io.StringIO()
write(buf, atoms, format="vasp", direct=True, vasp5=True)
lines = buf.getvalue().rstrip("\n").splitlines()
lines.append("  ".join(str(n) for n in grid))
lines.extend(f" {0.0:.11E}" for _ in range(int(np.prod(grid))))
lines.append("augmented occupancies   1 1")
open("CHGCAR", "w").write("\n".join(lines) + "\n")
```

Use the same checkpoint directly in Python (`predict.py` shows the full
recipe):

```python
import torch
from aiden.model import Aiden

ckpt = torch.load("checkpoints/aiden-pbe.pt", map_location="cpu")
model = Aiden(**ckpt["model_args"])
model.load_state_dict(ckpt["ema_model"])   # EMA weights; ckpt["model"] for raw
model.eval()
```

### Fine-tune on new data

Pack your training densities with `aiden.prepare_data` (see below), then start
from a pretrained checkpoint with `--resume-mode model_only` — this loads the
model weights and resets the optimizer, scheduler, and step counter:

```bash
python train.py \
    --data-root data/aiden-mp \
    --elem-csv data/elem-des.csv \
    --save-dir runs/aiden-finetune \
    --lmax 4 --mmax 4 \
    --initial-guess-steps 0 \
    --resume checkpoints/aiden-pbe.pt \
    --resume-mode model_only \
    --encoder-lr 2e-5 --decoder-lr 1e-4 \
    --max-steps 100000
```

Two ready-made transfer recipes from the paper (ECD HSE06) are recorded in
`configs/`: `hse-l3-finetune.json` fine-tunes all weights from
`checkpoints/aiden-pbe-l3.pt`, and `hse-l3-decoder.json` adds
`--freeze-encoder` to optimize the decoder only. Omit `--resume` to train
from scratch. To continue an interrupted run exactly where it stopped, use
`--resume runs/<run>/last.pt` with the default `--resume-mode full`.

## Data preparation

Training data are **not** included in this repository. The crystal benchmarks
come from the [ECD dataset](https://openreview.net/forum?id=SBCMNc3Mq3)
(PBE: 140,646 VASP PAW-PBE charge densities; HSE06: 7,147 recalculated
samples), and the molecular benchmark from the QM9 charge-density dataset of
DeepDFT.

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
133-dimensional element input (118-dim one-hot + 15 elemental descriptors).

## Training and evaluation

```bash
python train.py \
    --data-root data/aiden-mp \
    --elem-csv data/elem-des.csv \
    --save-dir runs/aiden-pbe-l4 \
    --lmax 4 --mmax 4 \
    --max-steps 2000000
```

Training first optimizes the element one-center coefficients alone for 1,000
steps, then the full model jointly with the grid-averaged MAE objective
(5,000 sampled grid points per structure, batch size 12, Adam with cosine
annealing and EMA evaluation weights; single A100-40GB in the paper).
`configs/` records the exact configurations of every paper run
(`pbe/qm9-l{0..4}`, `hse-l3-{scratch,finetune,decoder}`, and the HSE zero-shot
evaluation).

```bash
python evaluate.py \
    --checkpoint runs/aiden-pbe-l4/best_nmae.pt \
    --data-root data/aiden-mp \
    --elem-csv data/elem-des.csv \
    --split test
```

Reports the density MAE and the per-structure NMAE
`sum|rho_pred - rho_ref| / sum|rho_ref|` used in the paper.

## Repository layout

```
aiden/                    model and data package (model, encoder, TECE operators,
                          CHGCAR dataset/packing, encoder-cached fast inference)
train.py                  training entry point (incl. fine-tuning via --resume)
evaluate.py               held-out evaluation (MAE + per-structure NMAE)
predict.py                single-structure prediction
infer_full_grid.py        full-FFT-grid inference for one CHGCAR structure
smoke_test.py             fail-fast data/model/equivariance smoke test
configs/                  reference training configurations of the paper runs
checkpoints/aiden-pbe.pt  released PBE L=4 checkpoint
runs/                     training-run registry: configs, logs, metric tables,
                          and plotting scripts behind the paper figures
ablation/                 structural ablations (paper Table 3)
benchmarks/               inference-speed and parameter-count comparisons
examples/equivariance/    E(3)-equivariance check + density-decomposition cases
ood/                      out-of-distribution benchmark (water, AlMg, a-Si)
tbg/                      twisted-bilayer-graphene benchmark
tests/                    unit tests for the downloader
```

## Reproducing the paper experiments

Each experiment section of the paper maps to one directory: train with the
`configs/` references (benchmarks, Table 1 and Figs. 2/8/9 via `runs/`),
`ablation/` (Table 3), `benchmarks/` (inference speed and parameter counts),
`examples/equivariance/` (equivariance check, Fig. 7), `ood/` (zero-shot OOD,
Table 2 and Fig. 3), and `tbg/` (twisted bilayer graphene, Figs. 4/10 and
Tables 5-9). Every directory has its own README with the protocol and the
headline numbers.

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
