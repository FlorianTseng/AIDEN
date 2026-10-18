# Training-run registry

This directory archives the paper's training runs: the exact configuration
and the complete training log of every model, plus the evaluation tables and
plotting scripts behind the paper's loss-curve, violin, and parity figures
(Fig. 2, Fig. 8, Fig. 9). Model weights are not stored in git; every
checkpoint is downloadable from the repository's
[releases](https://github.com/FlorianTseng/AIDEN/releases) (see the table
below).

Each `aiden-*` directory holds the run's `config.json` (canonicalized to the
repository's relative data paths) and `train.log.gz` (the full per-100-step
log; the parsers in the plotting scripts read it transparently).

## Runs

| run | dataset | L | best val NMAE | paper reference |
|---|---|---:|---:|---|
| `aiden-pbe-l0` … `aiden-pbe-l4` | ECD PBE | 0–4 | 1.269 % / 1.019 % / 0.766 % / 0.621 % / 0.454 % | Table 1 PBE row (L = 3 / 4), Fig. 8a |
| `aiden-qm9-l0` … `aiden-qm9-l4` | QM9 | 0–4 | 1.180 % / 0.446 % / 0.217 % / 0.176 % / 0.163 % | Table 1 QM9 row, Fig. 8b |
| `aiden-hse-l3-finetune` | ECD HSE06 | 3 | 1.301 % | Table 1 HSE row, Fig. 2c |
| `aiden-hse-l3-decoder` | ECD HSE06 | 3 | decoder-only transfer | Fig. 2c |
| `aiden-hse-l3-scratch` | ECD HSE06 | 3 | 1.431 % | Table 1 HSE row, Fig. 2c |

The matching launch configurations (with canonical data paths) live in
`configs/` as `pbe-l{0..4}.json`, `qm9-l{0..4}.json`, and
`hse-l3-{finetune,decoder,scratch}.json`; the zero-shot evaluation of the PBE
model on HSE06 is archived as `configs/hse-zero-shot-*`.

## Evaluation tables and figures

| file | content |
|---|---|
| `mp_train_val_lmax_nmae.csv` / `qm9_train_val_lmax_nmae.csv` | per-structure train/val NMAE of every L, all splits |
| `mp_train_val_lmax_charge_5000.csv` (+ `.metadata.json`) | per-structure true/predicted total charge from the stored 5,000 probes (parity, Fig. 9) |
| `param_efficiency_comparison.csv` | parameter count and inference throughput vs L (Fig. 2a/2b) |
| `per_sample_inference_times.csv` | per-structure full-grid inference timing vs L |
| `mp-loss-curve.*` | PBE loss curves, L = 0..4 (Fig. 8a) |
| `pbe-hse-loss.*` | PBE vs HSE training/transfer curves |
| `new-pbe-qm-loss.*`, `new-mp-qm-distribution-hse.*`, `mp-qm-loss-distribution-pbe.*` | error violin/distribution panels of Fig. 8 |
| `mp-qm-charge-parity.*` | total-charge parity plot (Fig. 9) |

## Plotting

```bash
python plot-loss-curve-mp.py          # Fig. 8a-style loss curves
python new-plot-pbe-hse-qm9.py        # combined PBE/HSE/QM9 panels
```

`plot-pbe-hse-qm9.py` holds the data-loading and plotting primitives
(`new-plot-pbe-hse-qm9.py` builds on it); both read `train.log` or
`train.log.gz` from the run directories and the CSV tables above.
`run_qm9_reports.py` regenerates the QM9 CSVs from the released QM9
checkpoints and the prepared `data/aiden-qm9` shards (cluster-side wrapper).
