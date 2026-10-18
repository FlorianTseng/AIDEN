# AIDEN vs ChargE3Net vs BOA parameter count (L = 0..4)

`model-param-compare.csv` compares learnable-parameter counts of the three
models at maximum angular order L = 0..4, with every other hyperparameter
kept at each model's current defaults. Source repositories are only read,
never modified; all scripts and outputs live in this directory.

## What was counted

`total_params = sum over model.named_parameters()` — all registered
parameters, including the few `requires_grad=False` constants BOA stores as
Parameters (per-element L2 norm scales; ~0.7 k of ~102 M).
`trainable_params` excludes those; the two columns coincide for AIDEN and
ChargE3Net. Registered buffers (e3nn output masks, GTO exponent tables,
overlap-matrix inverses, EMA shadow copies) are excluded everywhere.

Per-model scope, kept apples-to-apples — each count is the complete model a
training run would optimize:

- **AIDEN**: `Aiden` module (Cartesian ACE/TECE encoder + GTO decoder,
  including the GTO radial-correction MLP and the element one-centre table).
- **ChargE3Net**: `E3DensityModel` (atom representation + probe message model).
- **BOA**: BOA net (BoaBlock stack + node/edge embeddings + initial-guess
  module) **plus** the per-element `GTOs` radial-correction MLPs that
  `ChgLightningModule.construct_orbitals` owns — mirroring AIDEN's convention.

## Configuration provenance

| | AIDEN | ChargE3Net | BOA |
|---|---|---|---|
| code | repository `aiden/` package | `third_party/charge3net/src/charge3net/models/e3.py` | `third_party/boa/boa/model` |
| defaults source | constructor defaults, identical to `configs/pbe-l{0..4}.json` | `configs/charge3net/model/e3_density.yaml` (what every published `*_e3_final` run resolves to) | `configs/model/net/boa.yaml` (4 blocks, hidden 32, K=8) + `configs/data/qm9_vasp_small.yaml` basis |
| L meaning | `lmax = mmax = L` | `lmax = L` | max shell angular momentum of the GTO basis: def2-qzvppd uncontracted, filtered to shells l ≤ L |
| fixed hyperparameters | atom_feat_dim=133, 48 ch, 2 layers, alternating ACE/TECE, corr (2,2), 4 heads, 8 Bessel, r_at=4.0 Å, K=8, K₀=1, N_G=8, r_orb=3.0 Å | 3 interactions, num_neighbors=20, mul=500, cutoff=4.0, gaussian basis, num_basis=20 | Z=[1,6,7,8,9] (QM9), orb_cutoff=3.0, radial correction on, direct GS prediction, initial-guess channels=1 |

Notes:

- ChargE3Net's code constructor defaults to `num_basis=10`, but every config
  and released checkpoint uses `num_basis=20`; the config value is used.
- def2-qzvppd reaches l=4 on C/N/O/F (g shells) and l=3 on H, so **BOA's L=4
  row is exactly its default configuration** and L=0..3 are the same basis
  with p/d/f/g shells progressively removed — a nested same-family sweep.

## Third-party checkouts

The two baseline repositories are not redistributed here. Clone them
read-only before running the counters:

```bash
mkdir -p third_party
git clone https://github.com/AIforGreatGood/charge3net.git third_party/charge3net
git clone https://github.com/sciai-lab/boa.git third_party/boa
```

Their Python dependencies (e3nn, pyscf, torch_geometric, ...) should be
installed in the active environment. If an isolated dependency bundle is
placed under `parameter-count/libs` (pip `--target`), the counters prefer it.

## Validation against real artifacts

- **AIDEN**: the released `checkpoints/aiden-pbe.pt` strict-loads into a fresh
  `Aiden(lmax=4, mmax=4)`; 4,458,747 learnable parameters match exactly.
- **ChargE3Net L=4**: official `third_party/charge3net/models/charge3net_mp.pt`
  strict-loads into `E3DensityModel(lmax=4, **e3_density.yaml defaults)`;
  1,909,389 learnable parameters match exactly.
- **BOA L=4**: the filtered-to-l≤4 basis reproduces
  `BasisInfo.from_atomic_numbers_with_even_tempered_basis(def2-qzvppd,
  uncontracted)` in per-element irreps, so the default model is counted.
  BOA releases no checkpoint in the repo.

## Headline numbers

| L | AIDEN | ChargE3Net | BOA |
|---|---|---|---|
| 0 | 161,979 | 3,910,680 | 101,943,491 |
| 1 | 387,699 | 2,508,341 | 101,949,101 |
| 2 | 975,291 | 2,011,470 | 101,950,835 |
| 3 | **2,187,507** (paper config) | 1,895,939 | 101,951,498 |
| 4 | 4,458,747 | **1,909,389** (official default) | **101,951,750** (default) |

Three different L-scaling behaviours, all structural:

- **AIDEN** grows ~2.2× per angular order — Cartesian coupling paths between
  weight-l features multiply with L (the TECE edge interaction dominates:
  1.85 M of 2.19 M at L=3).
- **ChargE3Net** *shrinks* with L — its fixed total multiplicity budget
  (`get_irreps(mul=500, lmax)`) is split over `6·(L+1)` parity-labeled
  irreps, so each irrep (and its tensor-product weights) gets smaller.
- **BOA** is almost L-independent (~102 M, spread across the sweep only
  8,259): its capacity sits in angular-order-agnostic channel-mixing MLPs
  (edge update ~51.6 M, per-element L2 nonlinearities ~42 M, message-pass
  attention ~8.4 M); L enters only through the per-element e3nn linears over
  basis irreps (tens of thousands of parameters).

At the models' respective operating points: AIDEN paper config L=3 → 2.19 M,
ChargE3Net default L=4 → 1.91 M, BOA default → 101.95 M (~50× AIDEN).

## Reproduce

```bash
cd benchmarks/parameter-count
python3 run_compare.py     # rebuilds model-param-compare.csv (+ details JSON)
```

`count_aiden.py`, `count_charge3net.py` and `count_boa.py` run standalone
(each prints JSON). Counters run as separate subprocesses because AIDEN
vendors a minimal `aiden.e3nn` subpackage that must not be confused with the
full `e3nn` the other two models import.
