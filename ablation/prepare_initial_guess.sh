#!/bin/bash

source ~/.bashrc
conda activate aiden

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export AIDEN_ABLATION_VARIANT=full
export AIDEN_PREPARE_SHARED_INITIAL_GUESS=1

shared_dir="ablation/shared_initial_guess_l4_2k"
shared_checkpoint="${shared_dir}/initial_guess.pt"

if [ -f "${shared_checkpoint}" ]; then
  python -c 'import sys, torch; payload = torch.load(sys.argv[1], map_location="cpu"); assert "initial_guess" in payload' "${shared_checkpoint}"
  checkpoint_status=$?
  if [ ${checkpoint_status} -eq 0 ]; then
    echo "Shared initial guess already exists and is readable: ${shared_checkpoint}"
    exit 0
  fi
  echo "Existing shared initial guess is invalid; refusing to overwrite it." >&2
  exit 3
fi

echo "Preparing shared one-center initialization on the fixed PBE-2k train subset"
python ablation/run_train.py \
  --data-root data/aiden-mp \
  --elem-csv data/elem-des.csv \
  --save-dir "${shared_dir}" \
  --max-files 150000 \
  --seed 42 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --num-queries-train 5000 \
  --num-queries-val 5000 \
  --batch-size 12 \
  --val-batch-size 6 \
  --val-max-structures 256 \
  --num-workers 4 \
  --val-num-workers 0 \
  --prefetch-factor 1 \
  --in-memory-cache-size 0 \
  --initial-guess-steps 1000 \
  --initial-guess-probes 5000 \
  --initial-guess-batch-size 12 \
  --initial-guess-lr 0.1 \
  --initial-guess-restart-steps 250 \
  --pretrain-log-interval 100 \
  --max-steps 1 \
  --encoder-lr 2e-4 \
  --decoder-lr 1e-3 \
  --weight-decay 0 \
  --grad-clip 0.5 \
  --ema-decay 0.995 \
  --log-interval 1 \
  --val-interval 1 \
  --checkpoint-interval 1 \
  --target-std 1.258352004595664 \
  --scale-factor 0.1 \
  --num-channels 48 \
  --num-encoder-layers 2 \
  --lmax 4 \
  --mmax 4 \
  --correlation 2 \
  --edge-correlation 2 \
  --num-heads 4 \
  --edge-hidden 48 \
  --atom-cutoff 4.0 \
  --max-atom-neighbors 100 \
  --num-atom-rbf 8 \
  --radial-hidden 128 \
  --atom-radial-type bessel \
  --dropout 0 \
  --num-orbitals 8 \
  --num-gto-radial 8 \
  --initial-guess-rank 1 \
  --orbital-cutoff 3.0 \
  --gto-min-exponent 0.15 \
  --gto-max-exponent 256.0 \
  --max-atomic-number 118 \
  --abs-scale 1000 \
  --gto-edge-chunk-size 32768 \
  --device cuda
