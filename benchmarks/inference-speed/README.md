# Encoder-cached inference benchmark

`aiden/fast_infer.py` is the streaming, encoder-cached inference path: it
computes the atom encoder and the atom GTO coefficients once per structure,
then decodes probe blocks from that cached state. The conventional
full-grid strategy recomputes the entire encoder for every query block. The
fast path also constructs each full-grid atom-probe radius graph on the GPU,
block by block, so a full-grid probe graph is never materialized or
transferred. The GTO equations, checkpoint state, float32 precision, cutoff,
and periodic images are unchanged.

## Benchmark on the stored 5,000 probes

```bash
sbatch benchmarks/inference-speed/job_benchmark_5000.sh
# or directly:
python benchmarks/inference-speed/benchmark_5000.py \
    --checkpoint checkpoints/aiden-pbe-l3.pt \
    --elem-csv data/elem-des.csv
```

This writes `benchmarks/inference-speed/benchmark_5000.json` (the shipped copy
is the paper run). It reports microseconds per point for four paths: original
one-shot, conventional original chunked, encoder-cached one-shot, and
encoder-cached chunked. The reported speedup is the meaningful full-grid
comparison: cached chunked versus original chunked. The reference job uses one
A100 GPU.

Before timing, it performs a strict fast/original decoder comparison using
the same cached encoder output and Cartesian GTO coefficients. For the chunked
comparison it allows only the normal CUDA sparse-reduction rounding variation
caused by the original path recomputing the encoder for every block, then
independently requires the NMAE difference to be at most `1e-6`. It records
both max absolute differences and NMAE. A failed numerical check makes the job
fail rather than silently reporting a speedup with altered predictions.

## Full-grid inference

Full-FFT-grid inference for a single CHGCAR structure is provided by the
repository-level entry point:

```bash
python infer_full_grid.py \
    --system my-system \
    --input path/to/CHGCAR \
    --checkpoint checkpoints/aiden-pbe.pt \
    --elem-csv data/elem-des.csv \
    --output-dir results/my-system \
    --query-block-size 65536
```

The output contains a flattened `density_pred` array (VASP CHGCAR order, x
fastest) plus `grid_shape`; reshape the prediction with Fortran order to that
shape.
