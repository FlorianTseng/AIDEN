#!/bin/bash

source ~/.bashrc
conda activate aiden

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export AIDEN_PREPARE_SHARED_INITIAL_GUESS=0

ablation_variant="$1"
case "${ablation_variant}" in
  full|no_gie|ace_ace|tece_tece|tece_ace|no_rra|no_rho_init|fixed_gto)
    ;;
  *)
    echo "Unknown ablation variant: ${ablation_variant}" >&2
    exit 2
    ;;
esac

export AIDEN_ABLATION_VARIANT="${ablation_variant}"
ablation_save_dir="ablation/runs_l4_2k_100k/${ablation_variant}"
ablation_result_dir="ablation/results_l4_2k_100k/${ablation_variant}"
ablation_initial_guess="ablation/shared_initial_guess_l4_2k/initial_guess.pt"
ablation_resume_args=()
ablation_initial_guess_args=(
  --initial-guess-steps 0
  --initial-guess-checkpoint "${ablation_initial_guess}"
)

if [ "${ablation_variant}" = "no_rho_init" ]; then
  ablation_initial_guess_args=(--initial-guess-steps 0)
elif [ ! -f "${ablation_initial_guess}" ]; then
  echo "Missing shared initial guess: ${ablation_initial_guess}" >&2
  echo "Submit ablation/job_prepare_initial_guess.sh first." >&2
  exit 4
fi

if [ -f "${ablation_save_dir}/last.pt" ]; then
  python -c 'import sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); assert archive.infolist(); archive.close()' "${ablation_save_dir}/last.pt" 2>/dev/null
  ablation_checkpoint_status=$?
  if [ ${ablation_checkpoint_status} -eq 0 ]; then
    ablation_resume_args=(--resume "${ablation_save_dir}/last.pt" --resume-mode full)
  fi
fi

if [ ${#ablation_resume_args[@]} -eq 0 ] && [ -f "${ablation_save_dir}/best_nmae.pt" ]; then
  python -c 'import sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); assert archive.infolist(); archive.close()' "${ablation_save_dir}/best_nmae.pt" 2>/dev/null
  ablation_checkpoint_status=$?
  if [ ${ablation_checkpoint_status} -eq 0 ]; then
    ablation_resume_args=(--resume "${ablation_save_dir}/best_nmae.pt" --resume-mode full)
  fi
fi

if [ ${#ablation_resume_args[@]} -eq 0 ] && { [ -e "${ablation_save_dir}/last.pt" ] || [ -e "${ablation_save_dir}/best_nmae.pt" ]; }; then
  echo "Existing checkpoints for ${ablation_variant} are invalid; refusing to overwrite them." >&2
  exit 3
fi

echo "Starting AIDEN L=4 PBE-2k/100k ablation: ${ablation_variant}"
python ablation/smoke_test.py \
  --data-root data/aiden-mp \
  --elem-csv data/elem-des.csv \
  --num-queries 5000 \
  --device cuda

ablation_smoke_status=$?
if [ ${ablation_smoke_status} -ne 0 ]; then
  echo "Smoke test failed for ${ablation_variant} with status ${ablation_smoke_status}." >&2
  exit ${ablation_smoke_status}
fi

python ablation/run_train.py \
  --data-root data/aiden-mp \
  --elem-csv data/elem-des.csv \
  --save-dir "${ablation_save_dir}" \
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
  --initial-guess-probes 5000 \
  --initial-guess-batch-size 12 \
  --initial-guess-lr 0.1 \
  --initial-guess-restart-steps 250 \
  --pretrain-log-interval 100 \
  --max-steps 100000 \
  --encoder-lr 2e-4 \
  --decoder-lr 1e-3 \
  --weight-decay 0 \
  --grad-clip 0.5 \
  --ema-decay 0.995 \
  --log-interval 100 \
  --val-interval 1000 \
  --checkpoint-interval 1000 \
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
  --device cuda \
  "${ablation_initial_guess_args[@]}" \
  "${ablation_resume_args[@]}"

ablation_train_status=$?
if [ ${ablation_train_status} -ne 0 ]; then
  echo "Training failed for ${ablation_variant} with status ${ablation_train_status}." >&2
  exit ${ablation_train_status}
fi

python ablation/evaluate.py \
  --checkpoint "${ablation_save_dir}/best_nmae.pt" \
  --data-root data/aiden-mp \
  --elem-csv data/elem-des.csv \
  --split test \
  --num-queries 5000 \
  --max-files 150000 \
  --max-structures 2000 \
  --batch-size 6 \
  --num-workers 4 \
  --device cuda \
  --output-dir "${ablation_result_dir}"
