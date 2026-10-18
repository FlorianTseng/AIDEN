# AIDEN L=4 rapid structural ablation

This directory contains the eight PBE ablations used to isolate AIDEN's
structural design choices. All runs start from the complete packed PBE manifest
at `data/aiden-mp` and reproduce the existing seed-42 80/10/10 split.
They then draw the same deterministic random subset of exactly 2,000 structures
from the original training split. Validation uses the same 256 evenly spaced
members of the original validation split, and final evaluation uses 2,000
evenly spaced members of the original test split.

Every run uses 5,000 stored density probes per structure, two interaction
layers, `lmax=mmax=4`, and 100,000 main-training steps with a cosine schedule
whose `T_max` is also 100,000. This is a one-seed rapid structural screen, not a
random-seed uncertainty study.

| ID | Variant | Definition |
|---:|---|---|
| 0 | `full` | GIE -> ACE -> TECE, full decoder |
| 1 | `no_gie` | Scalar element seed; higher-order initial fields are zero |
| 2 | `ace_ace` | Two ACE interaction layers |
| 3 | `tece_tece` | Two TECE interaction layers |
| 4 | `tece_ace` | TECE followed by ACE |
| 5 | `no_rra` | ECE retained; shared cutoff-normalized aggregation replaces RRA |
| 6 | `no_rho_init` | The one-center initial-density branch is absent |
| 7 | `fixed_gto` | The analytic even-tempered GTO basis has no radial correction |

`no_rra` uses

```text
a_e = f_c(r_e) / sum_(e' -> i) f_c(r_e')
```

for every head.  Thus it removes learned query/key scores and radial rotary
phase while retaining ECE, cutoff weighting, and neighbor normalization.

`no_rho_init` uses a parameter-free zero module and automatically skips the
initial-density pretraining stage. It is therefore a removal of the
branch, not merely a removal of its pretraining.

The other seven variants load one shared one-center checkpoint. That checkpoint
is trained for 1,000 steps on the same fixed 2,000-structure training subset, so
differences in independent one-center pretraining do not confound the encoder
comparison.

## Submit the shared initialization and eight independent jobs

Run from the project root:

```bash
init_job=$(sbatch --parsable ablation/job_prepare_initial_guess.sh)
init_job=${init_job%%;*}
sbatch --dependency=afterok:${init_job} ablation/job_full.sh
sbatch --dependency=afterok:${init_job} ablation/job_no_gie.sh
sbatch --dependency=afterok:${init_job} ablation/job_ace_ace.sh
sbatch --dependency=afterok:${init_job} ablation/job_tece_tece.sh
sbatch --dependency=afterok:${init_job} ablation/job_tece_ace.sh
sbatch --dependency=afterok:${init_job} ablation/job_no_rra.sh
sbatch --dependency=afterok:${init_job} ablation/job_no_rho_init.sh
sbatch --dependency=afterok:${init_job} ablation/job_fixed_gto.sh
```

The preparation job takes only a few minutes. The eight formal experiments are
independent one-GPU jobs on `a100x4`; no Slurm array is used. Once the dependency
is satisfied, Slurm can keep up to the available four GPUs occupied. Each job
first runs a real-data, 5,000-probe forward/backward and checkpoint-
reconstruction smoke test. A valid `last.pt` (or `best_nmae.pt` fallback) is
resumed automatically. An invalid existing checkpoint causes that job to stop
instead of overwriting it.

The old L=2 outputs are untouched. New checkpoints are written under
`ablation/runs_l4_2k_100k/<variant>/`, and test results under
`ablation/results_l4_2k_100k/<variant>/`. Once all eight runs complete, collect
the table with:

```bash
python ablation/summarize.py
```

This writes `ablation/ablation_results_l4_2k_100k.csv` and prints a Markdown
table.
