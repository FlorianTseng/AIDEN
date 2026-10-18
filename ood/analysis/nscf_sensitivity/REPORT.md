# Exploratory NSCF sensitivity audit

No SCF or new VASP calculation. Existing path and uniform-mesh NSCF eigenvalues only.
Windows: ±1, ±3, ±5 eV about the same reference SCF Fermi level. All results retained in metrics.csv.
Path points have equal weights (including repeated endpoints); mesh uses EIGENVAL k-point weights.
Alignment subtracts one weighted mean error per method and window; the same shift is used for both subsets.
This is post-hoc exploratory analysis, not a replacement for the original benchmark.

| sample | sampling | window eV | Aiden raw meV | C3 raw meV | Aiden aligned meV | C3 aligned meV |
|---|---|---:|---:|---:|---:|---:|
| water | bands | 1 | 64.702 | 129.646 | 73.186 | 42.819 |
| water | bands | 3 | 32.075 | 69.385 | 31.791 | 30.689 |
| water | bands | 5 | 40.371 | 58.282 | 40.619 | 33.945 |
| water | dos | 1 | 64.352 | 129.362 | 73.175 | 43.179 |
| water | dos | 3 | 32.023 | 69.250 | 31.833 | 30.916 |
| water | dos | 5 | 39.861 | 57.718 | 40.077 | 33.107 |
| almg | bands | 1 | 22.618 | 10.053 | 17.221 | 6.048 |
| almg | bands | 3 | 26.885 | 10.750 | 19.181 | 6.963 |
| almg | bands | 5 | 29.101 | 11.345 | 20.295 | 7.290 |
| almg | dos | 1 | 24.275 | 9.468 | 15.592 | 5.678 |
| almg | dos | 3 | 25.959 | 10.035 | 17.129 | 6.301 |
| almg | dos | 5 | 28.055 | 10.811 | 18.051 | 6.567 |
| asi_011 | bands | 1 | 7.193 | 9.865 | 7.018 | 8.429 |
| asi_011 | bands | 3 | 7.480 | 12.589 | 7.021 | 7.809 |
| asi_011 | bands | 5 | 7.311 | 13.656 | 6.984 | 8.032 |
| asi_011 | dos | 1 | 6.383 | 9.722 | 6.230 | 7.851 |
| asi_011 | dos | 3 | 6.944 | 12.130 | 6.360 | 7.543 |
| asi_011 | dos | 5 | 6.946 | 13.251 | 6.492 | 7.626 |
